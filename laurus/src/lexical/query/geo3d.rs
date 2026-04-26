//! 3D geographical search functionality (ECEF Cartesian queries).
//!
//! This module is the 3D counterpart of [`super::geo`]: where the 2D module
//! reasons about WGS84 latitude / longitude pairs and uses Haversine
//! distance, the 3D module reasons about Earth-Centered Earth-Fixed (ECEF)
//! Cartesian points and uses straight-line Euclidean distance. ECEF is
//! the right primitive for true 3D queries (drone proximity, satellite
//! tracking, indoor 3D positioning) where altitude is a first-class
//! dimension and pole-wrap or date-line concerns disappear.
//!
//! # Implemented queries
//!
//! - [`Geo3dDistanceQuery`] — sphere-radius search: every doc whose stored
//!   ECEF point lies within `radius_m` meters of the query center.
//! - [`Geo3dBoundingBoxQuery`] — 3D axis-aligned box search: every doc
//!   whose stored ECEF point falls inside the closed `[min, max]` box.
//!
//! The k-NN flavour arrives in #302.
//!
//! # How sphere queries reach the BKD
//!
//! [`Geo3dDistanceQuery::find_matches`] grabs the per-field
//! [`BKDTree`](crate::lexical::index::structures::bkd_tree::BKDTree) from
//! the reader and runs `tree.intersect(&mut SphereVisitor)`. The visitor's
//! [`compare`](IntersectVisitor::compare) implementation classifies each
//! cell as Inside / Outside / Crosses by comparing the cell's nearest /
//! farthest squared distance to the sphere's center against `radius²`,
//! enabling the BKD reader to skip whole subtrees that cannot intersect
//! the sphere.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::data::GeoEcefPoint;
use crate::error::Result;
use crate::lexical::index::structures::aabb::AABB;
use crate::lexical::index::structures::visitor::{CellRelation, IntersectVisitor};
use crate::lexical::query::Query;
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::scorer::Scorer;
use crate::lexical::reader::LexicalIndexReader;

/// A 3D distance (sphere) query against an ECEF-typed field.
///
/// Matches every document whose stored [`GeoEcefPoint`] lies within
/// `radius_m` meters of `center`. Distance is straight-line Euclidean in
/// ECEF space; a `radius_m` of 1000 means "within 1 km of `center`",
/// regardless of latitude or altitude.
///
/// Scoring follows the same shape as the 2D
/// [`GeoDistanceQuery`](super::geo::GeoDistanceQuery): score is
/// `1.0 - distance / radius`, clamped to `[0, 1]`. Documents that the BKD
/// reader classified as "fully inside" (their enclosing cell sits
/// entirely within the sphere) are reported with the maximum score `1.0`
/// because the visitor never observes their exact coordinates — see the
/// trade-off note on [`SphereVisitor::visit_inside`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geo3dDistanceQuery {
    /// Field containing 3D ECEF coordinates.
    field: String,
    /// Center point of the search sphere.
    center: GeoEcefPoint,
    /// Sphere radius in meters.
    radius_m: f64,
    /// Boost factor applied to the final score.
    boost: f32,
}

impl Geo3dDistanceQuery {
    /// Construct a new query.
    ///
    /// `radius_m` is interpreted as straight-line distance in ECEF meters.
    /// Negative or zero radii produce a query that matches no documents
    /// (see [`Geo3dDistanceQuery::is_empty`]).
    pub fn new<F: Into<String>>(field: F, center: GeoEcefPoint, radius_m: f64) -> Self {
        Self {
            field: field.into(),
            center,
            radius_m,
            boost: 1.0,
        }
    }

    /// Set the boost factor applied to every match's score.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Field name this query targets.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Search center.
    pub fn center(&self) -> GeoEcefPoint {
        self.center
    }

    /// Search radius in meters.
    pub fn radius_m(&self) -> f64 {
        self.radius_m
    }

    /// Run the query against `reader` and return the matches sorted by
    /// distance ascending (closest first).
    pub fn find_matches(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<Geo3dMatch>> {
        let mut matches: Vec<Geo3dMatch> = Vec::new();
        let Some(bkd) = reader.get_bkd_tree(&self.field)? else {
            return Ok(matches);
        };

        let mut visitor = SphereVisitor::new(self.center, self.radius_m);
        bkd.intersect(&mut visitor)?;

        let radius = self.radius_m;
        for hit in visitor.hits {
            let distance = hit.distance_sq.sqrt();
            let score = if hit.from_inside_cell {
                1.0
            } else if radius <= 0.0 {
                0.0
            } else {
                (1.0 - distance / radius).clamp(0.0, 1.0) as f32
            };
            matches.push(Geo3dMatch {
                doc_id: hit.doc_id,
                distance_m: distance,
                score,
            });
        }

        // Multi-segment readers can produce duplicates; keep the closest.
        matches.sort_by(|a, b| {
            a.doc_id.cmp(&b.doc_id).then_with(|| {
                a.distance_m
                    .partial_cmp(&b.distance_m)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        matches.dedup_by_key(|m| m.doc_id);
        // Final order: distance ascending.
        matches.sort_by(|a, b| {
            a.distance_m
                .partial_cmp(&b.distance_m)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(matches)
    }
}

/// A document matched by a [`Geo3dDistanceQuery`], with its distance to
/// the query center and a `[0, 1]` relevance score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geo3dMatch {
    /// Document id.
    pub doc_id: u64,
    /// Straight-line distance to the query center in meters. `0.0` for
    /// hits reported via [`SphereVisitor::visit_inside`] (the visitor
    /// does not observe the point's coordinates in that case).
    pub distance_m: f64,
    /// Final relevance score in `[0, 1]`, before the query's boost is
    /// applied by the [`Scorer`].
    pub score: f32,
}

/// `IntersectVisitor` that collects every doc whose stored point lies
/// within a sphere of radius `radius_m` centered at `center`.
///
/// The classification logic uses [`AABB::min_distance_sq_to_point`] (for
/// the Outside test) and [`AABB::max_distance_sq_to_point`] (for the
/// Inside test) so both checks happen in squared-distance space and
/// avoid `sqrt`.
struct SphereVisitor {
    center: [f64; 3],
    radius_sq: f64,
    hits: Vec<SphereHit>,
}

#[derive(Debug, Clone, Copy)]
struct SphereHit {
    doc_id: u64,
    /// Squared distance from `center` to the doc's stored point. `0.0`
    /// for hits arriving via `visit_inside` (the visitor does not
    /// observe the coordinates in that case).
    distance_sq: f64,
    /// `true` if the hit was reported via `visit_inside` rather than
    /// `visit`. The query layer treats Inside-cell hits as having a
    /// score of `1.0` regardless of their unknown distance.
    from_inside_cell: bool,
}

impl SphereVisitor {
    fn new(center: GeoEcefPoint, radius: f64) -> Self {
        let radius_clamped = radius.max(0.0);
        Self {
            center: [center.x, center.y, center.z],
            radius_sq: radius_clamped * radius_clamped,
            hits: Vec::new(),
        }
    }
}

impl IntersectVisitor for SphereVisitor {
    fn compare(&self, cell: &AABB) -> CellRelation {
        debug_assert_eq!(cell.num_dims(), 3, "SphereVisitor expects a 3D BKD");
        let min_d_sq = cell.min_distance_sq_to_point(&self.center);
        if min_d_sq > self.radius_sq {
            return CellRelation::Outside;
        }
        let max_d_sq = cell.max_distance_sq_to_point(&self.center);
        if max_d_sq <= self.radius_sq {
            return CellRelation::Inside;
        }
        CellRelation::Crosses
    }

    /// Trade-off note: `visit_inside` is invoked for every doc beneath a
    /// subtree whose AABB is fully inside the sphere, which by definition
    /// means the doc's point is inside the sphere too — but the visitor
    /// does not get the coordinates, so the exact distance is unknown.
    /// We mark the hit accordingly; the scorer surfaces it as `1.0`. If
    /// callers need exact per-doc distances they should consume
    /// [`Geo3dMatch::distance_m`] only when `score < 1.0`, or fall back
    /// to the `Geo3dNearestQuery` k-NN flavour from #302.
    fn visit_inside(&mut self, doc_id: u64) {
        self.hits.push(SphereHit {
            doc_id,
            distance_sq: 0.0,
            from_inside_cell: true,
        });
    }

    fn visit(&mut self, doc_id: u64, point: &[f64]) {
        debug_assert_eq!(point.len(), 3, "SphereVisitor expects a 3D BKD");
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        let dz = point[2] - self.center[2];
        let d_sq = dx * dx + dy * dy + dz * dz;
        if d_sq <= self.radius_sq {
            self.hits.push(SphereHit {
                doc_id,
                distance_sq: d_sq,
                from_inside_cell: false,
            });
        }
    }
}

/// Matcher for [`Geo3dDistanceQuery`]. Iterates over the matches in
/// distance-ascending order (mirrors [`super::geo::GeoMatcher`]).
#[derive(Debug)]
pub struct Geo3dMatcher {
    matches: Vec<Geo3dMatch>,
    cursor: usize,
}

impl Geo3dMatcher {
    /// Wrap a pre-sorted match list. Callers are expected to pass the
    /// distance-ascending list produced by
    /// [`Geo3dDistanceQuery::find_matches`].
    pub fn new(matches: Vec<Geo3dMatch>) -> Self {
        Self { matches, cursor: 0 }
    }
}

impl Matcher for Geo3dMatcher {
    fn doc_id(&self) -> u64 {
        if self.cursor >= self.matches.len() {
            u64::MAX
        } else {
            self.matches[self.cursor].doc_id
        }
    }

    fn next(&mut self) -> Result<bool> {
        self.cursor += 1;
        if self.cursor < self.matches.len() {
            Ok(true)
        } else {
            self.cursor = self.matches.len();
            Ok(false)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        // Linear scan over the distance-sorted list; matches the
        // semantics of `GeoMatcher::skip_to` in 2D.
        while self.cursor < self.matches.len() {
            if self.matches[self.cursor].doc_id >= target {
                return Ok(true);
            }
            self.cursor += 1;
        }
        Ok(false)
    }

    fn cost(&self) -> u64 {
        self.matches.len() as u64
    }

    fn is_exhausted(&self) -> bool {
        self.cursor >= self.matches.len()
    }
}

/// Scorer for [`Geo3dDistanceQuery`]. Looks up the precomputed score
/// per doc id and applies the query's boost.
#[derive(Debug)]
pub struct Geo3dScorer {
    doc_scores: HashMap<u64, f32>,
    boost: f32,
}

impl Geo3dScorer {
    /// Build a scorer from the matches returned by
    /// [`Geo3dDistanceQuery::find_matches`].
    pub fn new(matches: Vec<Geo3dMatch>, boost: f32) -> Self {
        let mut doc_scores = HashMap::with_capacity(matches.len());
        for m in matches {
            doc_scores.insert(m.doc_id, m.score);
        }
        Self { doc_scores, boost }
    }
}

impl Scorer for Geo3dScorer {
    fn score(&self, doc_id: u64, _term_freq: f32, _field_length: Option<f32>) -> f32 {
        self.doc_scores.get(&doc_id).copied().unwrap_or(0.0) * self.boost
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn max_score(&self) -> f32 {
        self.doc_scores.values().copied().fold(0.0_f32, f32::max) * self.boost
    }

    fn name(&self) -> &'static str {
        "Geo3dScorer"
    }
}

impl Query for Geo3dDistanceQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        Ok(Box::new(Geo3dMatcher::new(self.find_matches(reader)?)))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        Ok(Box::new(Geo3dScorer::new(
            self.find_matches(reader)?,
            self.boost,
        )))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn description(&self) -> String {
        format!(
            "Geo3dDistanceQuery(field: {}, center: {:?}, radius: {}m)",
            self.field, self.center, self.radius_m
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.radius_m <= 0.0)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        // Same shape as 2D GeoDistanceQuery::cost — moderate.
        let doc_count = reader.doc_count();
        Ok(doc_count.saturating_mul(2))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A 3D bounding-box query against an ECEF-typed field.
///
/// Matches every document whose stored [`GeoEcefPoint`] falls inside the
/// closed axis-aligned box `[min, max]` (inclusive on every axis). The
/// constructor enforces `min.x <= max.x` (and likewise for `y` / `z`); a
/// degenerate box where `min == max` on one or more axes is allowed and
/// matches points that lie exactly on that face / edge / corner.
///
/// Implemented as a thin wrapper over [`BKDTree::range_search`] with the
/// 3D bounds — no custom visitor is needed because the BKD's existing
/// axis-aligned range path already does the right thing for AABBs. As a
/// result every match comes back with `distance_m = 0.0` and `score =
/// 1.0` (the BKD only surfaces doc ids, not the points themselves, so
/// per-doc distance scoring would require an extra read pass; users who
/// want distance-based ranking should use [`Geo3dDistanceQuery`]).
///
/// [`BKDTree::range_search`]: crate::lexical::index::structures::bkd_tree::BKDTree::range_search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geo3dBoundingBoxQuery {
    /// Field containing 3D ECEF coordinates.
    field: String,
    /// South-west-bottom corner of the box (per-axis minimum).
    min: GeoEcefPoint,
    /// North-east-top corner of the box (per-axis maximum).
    max: GeoEcefPoint,
    /// Boost factor applied to the constant per-doc score.
    boost: f32,
}

impl Geo3dBoundingBoxQuery {
    /// Construct a new bounding-box query.
    ///
    /// # Errors
    /// Returns `LaurusError::other` when `min` exceeds `max` on any axis.
    /// A degenerate box where they are equal on one or more axes is
    /// accepted (it matches points lying exactly on that face / edge /
    /// corner).
    pub fn new<F: Into<String>>(field: F, min: GeoEcefPoint, max: GeoEcefPoint) -> Result<Self> {
        if min.x > max.x {
            return Err(crate::error::LaurusError::other(format!(
                "Geo3dBoundingBoxQuery: min.x ({}) must be <= max.x ({})",
                min.x, max.x
            )));
        }
        if min.y > max.y {
            return Err(crate::error::LaurusError::other(format!(
                "Geo3dBoundingBoxQuery: min.y ({}) must be <= max.y ({})",
                min.y, max.y
            )));
        }
        if min.z > max.z {
            return Err(crate::error::LaurusError::other(format!(
                "Geo3dBoundingBoxQuery: min.z ({}) must be <= max.z ({})",
                min.z, max.z
            )));
        }
        Ok(Self {
            field: field.into(),
            min,
            max,
            boost: 1.0,
        })
    }

    /// Set the boost factor applied to every match's score.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Field name this query targets.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// SW-bottom corner (per-axis minimum).
    pub fn min(&self) -> GeoEcefPoint {
        self.min
    }

    /// NE-top corner (per-axis maximum).
    pub fn max(&self) -> GeoEcefPoint {
        self.max
    }

    /// Run the query against `reader` and return the matches in
    /// doc-id-ascending order. Every match has `distance_m = 0.0` and
    /// `score = 1.0` — see the type docs for the rationale.
    pub fn find_matches(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<Geo3dMatch>> {
        let Some(bkd) = reader.get_bkd_tree(&self.field)? else {
            return Ok(Vec::new());
        };
        let mins = [Some(self.min.x), Some(self.min.y), Some(self.min.z)];
        let maxs = [Some(self.max.x), Some(self.max.y), Some(self.max.z)];
        // Closed bounds on both ends: a point sitting exactly on a face
        // of the box is a hit (consistent with the constructor's
        // tolerance for degenerate boxes).
        let doc_ids = bkd.range_search(&mins, &maxs, true, true)?;
        Ok(doc_ids
            .into_iter()
            .map(|doc_id| Geo3dMatch {
                doc_id,
                distance_m: 0.0,
                score: 1.0,
            })
            .collect())
    }
}

impl Query for Geo3dBoundingBoxQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        Ok(Box::new(Geo3dMatcher::new(self.find_matches(reader)?)))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        Ok(Box::new(Geo3dScorer::new(
            self.find_matches(reader)?,
            self.boost,
        )))
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn description(&self) -> String {
        format!(
            "Geo3dBoundingBoxQuery(field: {}, min: {:?}, max: {:?})",
            self.field, self.min, self.max
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        // The constructor guarantees min <= max on every axis, so the box
        // is always non-empty in the geometric sense (it contains at
        // least one point). Whether any document falls inside is a
        // runtime question; we leave that to the matcher.
        Ok(false)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        let doc_count = reader.doc_count();
        Ok(doc_count.saturating_mul(2))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(min: [f64; 3], max: [f64; 3]) -> AABB {
        AABB::new(min.to_vec(), max.to_vec()).unwrap()
    }

    fn visitor(cx: f64, cy: f64, cz: f64, radius: f64) -> SphereVisitor {
        SphereVisitor::new(GeoEcefPoint::new(cx, cy, cz), radius)
    }

    #[test]
    fn sphere_visitor_compare_outside() {
        // Sphere radius 5 around origin; cell sits 100 units away on x.
        let v = visitor(0.0, 0.0, 0.0, 5.0);
        let c = cell([100.0, 0.0, 0.0], [110.0, 10.0, 10.0]);
        assert_eq!(v.compare(&c), CellRelation::Outside);
    }

    #[test]
    fn sphere_visitor_compare_inside() {
        // Sphere radius 100 around origin; cell tightly centered fits
        // entirely inside (corner distance √(3) * 1 < 100).
        let v = visitor(0.0, 0.0, 0.0, 100.0);
        let c = cell([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert_eq!(v.compare(&c), CellRelation::Inside);
    }

    #[test]
    fn sphere_visitor_compare_crosses_at_boundary() {
        // Sphere radius 5; cell straddles the boundary.
        let v = visitor(0.0, 0.0, 0.0, 5.0);
        let c = cell([0.0, 0.0, 0.0], [10.0, 0.0, 0.0]);
        assert_eq!(v.compare(&c), CellRelation::Crosses);
    }

    #[test]
    fn sphere_visitor_visit_filters_by_radius() {
        let mut v = visitor(0.0, 0.0, 0.0, 5.0);
        // Inside the sphere
        v.visit(1, &[1.0, 2.0, 2.0]); // sqrt(9) = 3
        // On the boundary (5² + 0 + 0 = 25 == radius²)
        v.visit(2, &[5.0, 0.0, 0.0]);
        // Outside the sphere
        v.visit(3, &[10.0, 0.0, 0.0]);
        assert_eq!(v.hits.len(), 2);
        assert_eq!(v.hits[0].doc_id, 1);
        assert!(!v.hits[0].from_inside_cell);
        assert_eq!(v.hits[1].doc_id, 2);
    }

    #[test]
    fn sphere_visitor_visit_inside_marks_hit_uniformly() {
        let mut v = visitor(0.0, 0.0, 0.0, 100.0);
        v.visit_inside(42);
        assert_eq!(v.hits.len(), 1);
        assert_eq!(v.hits[0].doc_id, 42);
        assert_eq!(v.hits[0].distance_sq, 0.0);
        assert!(v.hits[0].from_inside_cell);
    }

    #[test]
    fn sphere_visitor_negative_radius_matches_nothing() {
        let mut v = visitor(0.0, 0.0, 0.0, -1.0);
        // radius is clamped to 0, so radius² = 0; only a hit at the
        // exact center qualifies.
        v.visit(1, &[0.0, 0.0, 0.0]);
        v.visit(2, &[1.0, 0.0, 0.0]);
        assert_eq!(v.hits.len(), 1);
        assert_eq!(v.hits[0].doc_id, 1);
    }

    #[test]
    fn geo3d_distance_query_basics() {
        let q = Geo3dDistanceQuery::new("position", GeoEcefPoint::new(1.0, 2.0, 3.0), 500.0)
            .with_boost(2.0);
        assert_eq!(q.field(), "position");
        assert_eq!(q.center(), GeoEcefPoint::new(1.0, 2.0, 3.0));
        assert_eq!(q.radius_m(), 500.0);
        assert_eq!(q.boost(), 2.0);
        let cloned = q.clone_box();
        assert!(cloned.description().contains("Geo3dDistanceQuery"));
    }

    #[test]
    fn geo3d_bbox_query_basics() {
        let q = Geo3dBoundingBoxQuery::new(
            "position",
            GeoEcefPoint::new(0.0, 0.0, 0.0),
            GeoEcefPoint::new(10.0, 20.0, 30.0),
        )
        .unwrap()
        .with_boost(3.0);
        assert_eq!(q.field(), "position");
        assert_eq!(q.min(), GeoEcefPoint::new(0.0, 0.0, 0.0));
        assert_eq!(q.max(), GeoEcefPoint::new(10.0, 20.0, 30.0));
        assert_eq!(q.boost(), 3.0);
        let cloned = q.clone_box();
        assert!(cloned.description().contains("Geo3dBoundingBoxQuery"));
    }

    #[test]
    fn geo3d_bbox_query_accepts_degenerate_box() {
        // Zero-volume box (single point); valid per the type docs.
        let q = Geo3dBoundingBoxQuery::new(
            "position",
            GeoEcefPoint::new(5.0, 5.0, 5.0),
            GeoEcefPoint::new(5.0, 5.0, 5.0),
        );
        assert!(q.is_ok(), "degenerate (zero-volume) box must be accepted");
    }

    #[test]
    fn geo3d_bbox_query_accepts_zero_volume_axis() {
        // Zero on a single axis (a face) — also valid.
        let q = Geo3dBoundingBoxQuery::new(
            "position",
            GeoEcefPoint::new(0.0, 0.0, 5.0),
            GeoEcefPoint::new(10.0, 10.0, 5.0),
        );
        assert!(
            q.is_ok(),
            "box with zero extent on one axis must be accepted"
        );
    }

    #[test]
    fn geo3d_bbox_query_rejects_inverted_box() {
        // min.x > max.x → reject.
        let err = Geo3dBoundingBoxQuery::new(
            "position",
            GeoEcefPoint::new(10.0, 0.0, 0.0),
            GeoEcefPoint::new(5.0, 10.0, 10.0),
        )
        .unwrap_err();
        assert!(format!("{err:?}").contains("min.x"));

        // min.y > max.y → reject.
        let err_y = Geo3dBoundingBoxQuery::new(
            "position",
            GeoEcefPoint::new(0.0, 10.0, 0.0),
            GeoEcefPoint::new(10.0, 5.0, 10.0),
        )
        .unwrap_err();
        assert!(format!("{err_y:?}").contains("min.y"));

        // min.z > max.z → reject.
        let err_z = Geo3dBoundingBoxQuery::new(
            "position",
            GeoEcefPoint::new(0.0, 0.0, 10.0),
            GeoEcefPoint::new(10.0, 10.0, 5.0),
        )
        .unwrap_err();
        assert!(format!("{err_z:?}").contains("min.z"));
    }
}
