//! Geographical search functionality for location-based queries.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// Canonical GeoPoint definition lives in `crate::data` alongside DataValue.
// We re-export it here so that `use laurus::lexical::query::GeoPoint` keeps
// working for downstream callers.
pub use crate::data::GeoPoint;

use crate::error::Result;
use crate::lexical::query::Query;
use crate::lexical::query::matcher::Matcher;
use crate::lexical::query::scorer::Scorer;
use crate::lexical::reader::LexicalIndexReader;

/// A 2D geographical bounding box, expressed as the south-west `min` corner
/// and the north-east `max` corner of an axis-aligned lat/lon rectangle.
///
/// Invariant: `min.lat <= max.lat` and `min.lon <= max.lon`. The constructor
/// validates this and returns `LaurusError::other` for inverted boxes. The
/// previous `top_left` / `bottom_right` representation was removed in #297;
/// `min` / `max` matches the convention used by the redesigned BKD AABB,
/// which makes `GeoBoundingBox` ↔ AABB conversion a direct copy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoBoundingBox {
    /// South-west corner: minimum latitude and minimum longitude.
    pub min: GeoPoint,
    /// North-east corner: maximum latitude and maximum longitude.
    pub max: GeoPoint,
}

impl GeoBoundingBox {
    /// Create a new bounding box from its south-west and north-east corners.
    ///
    /// Errors if `min.lat > max.lat` or `min.lon > max.lon`.
    pub fn new(min: GeoPoint, max: GeoPoint) -> Result<Self> {
        if min.lat > max.lat {
            return Err(crate::error::LaurusError::other(format!(
                "min latitude {} must be <= max latitude {}",
                min.lat, max.lat
            )));
        }
        if min.lon > max.lon {
            return Err(crate::error::LaurusError::other(format!(
                "min longitude {} must be <= max longitude {}",
                min.lon, max.lon
            )));
        }
        Ok(GeoBoundingBox { min, max })
    }

    /// Whether `point` lies inside the closed rectangle `[min, max]`.
    pub fn contains(&self, point: &GeoPoint) -> bool {
        point.within_bounds(self.min.lat, self.max.lat, self.min.lon, self.max.lon)
    }

    /// Center of the bounding box, clamped to the valid lat/lon ranges as
    /// a defensive measure (a malformed box could in theory carry
    /// out-of-range corners).
    pub fn center(&self) -> GeoPoint {
        let center_lat = ((self.min.lat + self.max.lat) / 2.0).clamp(-90.0, 90.0);
        let center_lon = ((self.min.lon + self.max.lon) / 2.0).clamp(-180.0, 180.0);
        // Clamped to valid ranges, so the infallible constructor is fine.
        GeoPoint::new(center_lat, center_lon)
    }

    /// Width (lon span) and height (lat span) in degrees.
    pub fn dimensions(&self) -> (f64, f64) {
        let width = self.max.lon - self.min.lon;
        let height = self.max.lat - self.min.lat;
        (width, height)
    }

    /// Greatest great-circle distance from the box's center to any of its
    /// four corners, in meters.
    pub fn max_distance_from_center(&self) -> f64 {
        let center = self.center();
        let sw = self.min;
        let ne = self.max;
        // The bounding box invariant guarantees that swapping corners
        // produces in-range GeoPoints, so the infallible constructor is
        // appropriate here.
        let nw = GeoPoint::new(self.max.lat, self.min.lon);
        let se = GeoPoint::new(self.min.lat, self.max.lon);
        [&sw, &se, &nw, &ne]
            .iter()
            .map(|corner| center.distance_to(corner))
            .fold(0.0, f64::max)
    }
}

/// A geographical distance query that finds documents within a certain distance of a point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoDistanceQuery {
    /// Field containing geographical coordinates
    field: String,
    /// Center point for the search
    center: GeoPoint,
    /// Maximum distance in meters
    distance_m: f64,
    /// Boost factor for the query
    boost: f32,
}

impl GeoDistanceQuery {
    /// Create a new geo distance query.
    pub fn new<F: Into<String>>(field: F, center: GeoPoint, distance_m: f64) -> Self {
        GeoDistanceQuery {
            field: field.into(),
            center,
            distance_m,
            boost: 1.0,
        }
    }

    /// Create a geo distance query from raw lat/lon coordinates.
    ///
    /// Validates the latitude / longitude bounds via [`GeoPoint::try_new`]
    /// (`-90 <= lat <= 90`, `-180 <= lon <= 180`) and returns
    /// `LaurusError::other` for out-of-range values.
    ///
    /// # Arguments
    ///
    /// * `field` - The field containing geographical coordinates.
    /// * `lat` - Center latitude in degrees (`-90` to `90`).
    /// * `lon` - Center longitude in degrees (`-180` to `180`).
    /// * `distance_m` - Maximum distance from the centre in meters.
    ///
    /// # Example
    ///
    /// ```rust
    /// use laurus::lexical::query::GeoDistanceQuery;
    ///
    /// let query =
    ///     GeoDistanceQuery::within_radius("location", 40.7128, -74.0060, 10_000.0).unwrap();
    /// ```
    pub fn within_radius<F: Into<String>>(
        field: F,
        lat: f64,
        lon: f64,
        distance_m: f64,
    ) -> Result<Self> {
        let center = GeoPoint::try_new(lat, lon)?;
        Ok(GeoDistanceQuery::new(field, center, distance_m))
    }

    /// Set the boost factor.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the center point.
    pub fn center(&self) -> GeoPoint {
        self.center
    }

    /// Get the search distance in meters.
    pub fn distance_m(&self) -> f64 {
        self.distance_m
    }

    /// Find matching documents and their distances using spatial indexing.
    pub fn find_matches(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<GeoMatch>> {
        let mut matches = Vec::new();
        let mut seen_docs = std::collections::HashSet::new();

        // Create a bounding box for efficient filtering
        let bounding_box = self.create_bounding_box();

        // Get candidates from the index
        let candidates = self.get_spatial_candidates(reader, &bounding_box)?;

        for (doc_id, point) in candidates {
            // Skip if we've already processed this document
            if seen_docs.contains(&doc_id) {
                continue;
            }
            seen_docs.insert(doc_id);

            let distance = self.center.distance_to(&point);
            if distance <= self.distance_m {
                let score = if distance == 0.0 {
                    1.0
                } else {
                    // Simple inverse distance scoring
                    (1.0 - (distance / self.distance_m)).max(0.0) as f32
                };

                matches.push(GeoMatch {
                    doc_id,
                    point,
                    distance_m: distance,
                    relevance_score: score,
                });
            }
        }

        // Sort by distance (closest first), then by relevance score
        matches.sort_by(|a, b| {
            a.distance_m
                .partial_cmp(&b.distance_m)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.relevance_score
                        .partial_cmp(&a.relevance_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        Ok(matches)
    }

    /// Create a bounding box for efficient spatial filtering.
    fn create_bounding_box(&self) -> GeoBoundingBox {
        // Approximate degree distance at the center latitude
        let lat_deg_m = 111_000.0; // ~111 km (= 111 000 m) per degree latitude
        // At poles cos(lat) ≈ 0 → clamp to avoid division by zero
        let lon_deg_m = (111_000.0 * self.center.lat.to_radians().cos()).max(1.0);

        let lat_delta = self.distance_m / lat_deg_m;
        let lon_delta = self.distance_m / lon_deg_m;

        // Clamp to the valid lat/lon ranges so the resulting GeoPoints are
        // always in-range (the infallible `GeoPoint::new` debug-asserts).
        let min = GeoPoint::new(
            (self.center.lat - lat_delta).clamp(-90.0, 90.0),
            (self.center.lon - lon_delta).clamp(-180.0, 180.0),
        );
        let max = GeoPoint::new(
            (self.center.lat + lat_delta).clamp(-90.0, 90.0),
            (self.center.lon + lon_delta).clamp(-180.0, 180.0),
        );

        GeoBoundingBox::new(min, max).unwrap_or_else(|_| {
            // Fallback to a tiny box around the center if validation fails
            // (extreme deltas at the poles can produce inverted boxes).
            let fmin = GeoPoint::new(
                (self.center.lat - 0.01).clamp(-90.0, 90.0),
                (self.center.lon - 0.01).clamp(-180.0, 180.0),
            );
            let fmax = GeoPoint::new(
                (self.center.lat + 0.01).clamp(-90.0, 90.0),
                (self.center.lon + 0.01).clamp(-180.0, 180.0),
            );
            GeoBoundingBox::new(fmin, fmax).unwrap_or(GeoBoundingBox {
                min: self.center,
                max: self.center,
            })
        })
    }

    /// Get spatial candidates from the index within the bounding box.
    fn get_spatial_candidates(
        &self,
        reader: &dyn LexicalIndexReader,
        bounding_box: &GeoBoundingBox,
    ) -> Result<Vec<(u64, GeoPoint)>> {
        let mut candidates = Vec::new();

        // Try to use BKD tree for efficient candidate retrieval
        if let Some(bkd_tree) = reader.get_bkd_tree(&self.field)? {
            // Lat range is [min_lat, max_lat], Lon range is [min_lon, max_lon]
            let mins = [
                Some(bounding_box.min.lat), // min_lat
                Some(bounding_box.min.lon), // min_lon
            ];
            let maxs = [
                Some(bounding_box.max.lat), // max_lat
                Some(bounding_box.max.lon), // max_lon
            ];

            let doc_ids = bkd_tree.range_search(&mins, &maxs, true, true)?;
            for doc_id in doc_ids {
                // Retrieve GeoPoint for exact distance calculation
                if let Some(doc) = reader.document(doc_id)?
                    && let Some(field_value) = doc.get_field(&self.field)
                    && let Some(geo_point) = field_value.as_geo()
                {
                    candidates.push((doc_id, geo_point));
                }
            }
            return Ok(candidates);
        }

        // Get the maximum document ID to iterate through all documents
        let max_doc = reader.max_doc();

        // Iterate through all documents in the index
        for doc_id in 0..max_doc {
            // Get the document
            if let Some(doc) = reader.document(doc_id)? {
                // Get the geo field value
                if let Some(field_value) = doc.get_field(&self.field) {
                    // Extract the GeoPoint from the field value
                    if let Some(geo_point) = field_value.as_geo() {
                        // First check bounding box for efficiency, then exact distance
                        if bounding_box.contains(&geo_point) {
                            let distance = self.center.distance_to(&geo_point);
                            // Double-check with exact distance calculation
                            if distance <= self.distance_m {
                                candidates.push((doc_id, geo_point));
                            }
                        }
                    }
                }
            }
        }

        Ok(candidates)
    }
}

#[cfg(test)]
impl GeoDistanceQuery {
    /// Calculate relevance score based on distance (closer = higher score).
    fn calculate_distance_score(&self, distance_m: f64) -> f32 {
        if distance_m > self.distance_m {
            return 0.0;
        }

        // Linear decay: score = 1.0 at center, 0.0 at max distance
        let normalized_distance = distance_m / self.distance_m;
        (1.0 - normalized_distance) as f32
    }

    /// Calculate enhanced relevance score with multiple factors.
    fn calculate_distance_score_enhanced(&self, distance_m: f64, point: &GeoPoint) -> f32 {
        if distance_m > self.distance_m {
            return 0.0;
        }

        // Base distance score (exponential decay for better distance weighting)
        let normalized_distance = distance_m / self.distance_m;
        let base_score = (-2.0 * normalized_distance).exp() as f32;

        // Precision bonus for exact location matches (within 100 m)
        let precision_bonus = if distance_m < 100.0 { 0.1 } else { 0.0 };

        // Geographic relevance bonus (e.g., prefer points in certain regions)
        let geo_bonus = self.calculate_geographic_relevance(point);

        // Population density estimation (simulated)
        let density_bonus = self.estimate_population_density(point) * 0.05;

        (base_score + precision_bonus + geo_bonus + density_bonus).min(1.0)
    }

    /// Calculate geographic relevance based on point characteristics.
    fn calculate_geographic_relevance(&self, point: &GeoPoint) -> f32 {
        // Bonus for points in certain latitudinal zones (e.g., temperate zones)
        let lat_abs = point.lat.abs();
        let temperate_bonus = if lat_abs > 23.5 && lat_abs < 66.5 {
            0.05
        } else {
            0.0
        };

        // Bonus for points near major meridians or equator
        let meridian_bonus = if point.lon.abs() % 15.0 < 1.0 {
            0.02
        } else {
            0.0
        };
        let equator_bonus = if point.lat.abs() < 5.0 { 0.03 } else { 0.0 };

        temperate_bonus + meridian_bonus + equator_bonus
    }

    /// Estimate population density bonus (simplified simulation).
    fn estimate_population_density(&self, point: &GeoPoint) -> f32 {
        // Simplified heuristic: higher density near major coordinates
        let lat_density = (1.0 - (point.lat.abs() / 90.0)) as f32;
        let lon_density = (1.0 - (point.lon.abs() / 180.0)) as f32;

        // Coastal bonus (simplified: points near 0° longitude or specific latitudes)
        let coastal_bonus = if point.lon.abs() < 10.0 || (point.lat.abs() - 40.0).abs() < 5.0 {
            0.2
        } else {
            0.0
        };

        ((lat_density + lon_density) / 2.0 + coastal_bonus).min(1.0)
    }
}

impl Query for GeoDistanceQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        let matches = self.find_matches(reader)?;
        Ok(Box::new(GeoMatcher::new(matches)))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        let matches = self.find_matches(reader)?;
        Ok(Box::new(GeoScorer::new(matches, self.boost)))
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
            "GeoDistanceQuery(field: {}, center: {:?}, distance: {}m)",
            self.field, self.center, self.distance_m
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        Ok(self.distance_m <= 0.0)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        // Geo queries can be expensive depending on the spatial index
        let doc_count = reader.doc_count() as u32;
        Ok(doc_count as u64 * 2) // Moderate cost
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A geographical bounding box query.
///
/// This query matches documents with coordinates within a rectangular bounding box
/// defined by top-left and bottom-right corners.
///
/// # Scoring Behavior
///
/// Documents within the bounding box are scored based on their distance from the
/// bounding box center. Documents closer to the center receive higher scores.
/// This allows for relevance ranking even when all results are within the box.
///
/// - Score range: 0.0 to 1.0
/// - Documents at the center: score ≈ 1.0
/// - Documents at the edges: score decreases based on distance from center
///
/// If you need all documents within the box to have equal scores, consider using
/// a constant score wrapper or filtering approach instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoBoundingBoxQuery {
    /// Field containing geographical coordinates
    field: String,
    /// Bounding box for the search
    bounding_box: GeoBoundingBox,
    /// Boost factor for the query
    boost: f32,
}

impl GeoBoundingBoxQuery {
    /// Create a new geo bounding box query.
    pub fn new<F: Into<String>>(field: F, bounding_box: GeoBoundingBox) -> Self {
        GeoBoundingBoxQuery {
            field: field.into(),
            bounding_box,
            boost: 1.0,
        }
    }

    /// Create a geo bounding box query from raw lat/lon corner coordinates.
    ///
    /// Validates that each corner falls within the latitude / longitude
    /// bounds (via [`GeoPoint::try_new`]) and that `min_lat <= max_lat`
    /// / `min_lon <= max_lon` (via [`GeoBoundingBox::new`]).
    ///
    /// # Arguments
    ///
    /// * `field` - The field containing geographical coordinates.
    /// * `min_lat` - Minimum (south) latitude.
    /// * `min_lon` - Minimum (west) longitude.
    /// * `max_lat` - Maximum (north) latitude.
    /// * `max_lon` - Maximum (east) longitude.
    ///
    /// # Example
    ///
    /// ```rust
    /// use laurus::lexical::query::GeoBoundingBoxQuery;
    ///
    /// let query = GeoBoundingBoxQuery::within_bounding_box(
    ///     "location", 40.0, -75.0, 41.0, -74.0,
    /// ).unwrap();
    /// ```
    pub fn within_bounding_box<F: Into<String>>(
        field: F,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Result<Self> {
        let min = GeoPoint::try_new(min_lat, min_lon)?;
        let max = GeoPoint::try_new(max_lat, max_lon)?;
        let bbox = GeoBoundingBox::new(min, max)?;
        Ok(GeoBoundingBoxQuery::new(field, bbox))
    }

    /// Set the boost factor.
    pub fn with_boost(mut self, boost: f32) -> Self {
        self.boost = boost;
        self
    }

    /// Get the field name.
    pub fn field(&self) -> &str {
        &self.field
    }

    /// Get the bounding box.
    pub fn bounding_box(&self) -> &GeoBoundingBox {
        &self.bounding_box
    }

    /// Find matching documents within the bounding box.
    pub fn find_matches(&self, reader: &dyn LexicalIndexReader) -> Result<Vec<GeoMatch>> {
        let mut matches = Vec::new();
        let mut seen_docs = std::collections::HashSet::new();

        // Get candidates from the spatial index
        let candidates = self.get_candidates_in_bounds(reader)?;

        for (doc_id, point) in candidates {
            // Skip if we've already processed this document
            if seen_docs.contains(&doc_id) {
                continue;
            }
            seen_docs.insert(doc_id);

            if self.bounding_box.contains(&point) {
                let center = self.bounding_box.center();
                let distance = center.distance_to(&point);

                // Simple scoring based on position within bounding box
                let relevance_score = if distance == 0.0 {
                    1.0
                } else {
                    // Closer to center gets higher score
                    let max_distance = self.bounding_box.max_distance_from_center();
                    ((max_distance - distance) / max_distance).max(0.0) as f32
                };

                matches.push(GeoMatch {
                    doc_id,
                    point,
                    distance_m: distance,
                    relevance_score,
                });
            }
        }

        // Sort by relevance score (highest first), then by distance to center
        matches.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap()
                .then_with(|| a.distance_m.partial_cmp(&b.distance_m).unwrap())
        });

        Ok(matches)
    }

    /// Get candidate points that might be within the bounding box.
    fn get_candidates_in_bounds(
        &self,
        reader: &dyn LexicalIndexReader,
    ) -> Result<Vec<(u64, GeoPoint)>> {
        let mut candidates = Vec::new();

        // Try to use BKD tree for efficient candidate retrieval
        if let Some(bkd_tree) = reader.get_bkd_tree(&self.field)? {
            let mins = [
                Some(self.bounding_box.min.lat), // min_lat
                Some(self.bounding_box.min.lon), // min_lon
            ];
            let maxs = [
                Some(self.bounding_box.max.lat), // max_lat
                Some(self.bounding_box.max.lon), // max_lon
            ];

            let doc_ids = bkd_tree.range_search(&mins, &maxs, true, true)?;
            for doc_id in doc_ids {
                // Retrieve GeoPoint for exact check
                if let Some(doc) = reader.document(doc_id)?
                    && let Some(field_value) = doc.get_field(&self.field)
                    && let Some(geo_point) = field_value.as_geo()
                {
                    candidates.push((doc_id, geo_point));
                }
            }
            return Ok(candidates);
        }

        // Get the maximum document ID to iterate through all documents
        let max_doc = reader.max_doc();

        // Iterate through all documents in the index
        for doc_id in 0..max_doc {
            // Get the document
            if let Some(doc) = reader.document(doc_id)? {
                // Get the geo field value
                if let Some(field_value) = doc.get_field(&self.field) {
                    // Extract the GeoPoint from the field value
                    if let Some(geo_point) = field_value.as_geo() {
                        // Check if the point is within the bounding box
                        if self.bounding_box.contains(&geo_point) {
                            candidates.push((doc_id, geo_point));
                        }
                    }
                }
            }
        }

        Ok(candidates)
    }
}

#[cfg(test)]
impl GeoBoundingBoxQuery {
    /// Generate candidates within and around the bounding box.
    fn generate_bounding_box_candidates(&self) -> Vec<(u64, GeoPoint)> {
        let mut candidates = Vec::new();
        let (width, height) = self.bounding_box.dimensions();

        // Generate grid points within the bounding box
        let grid_size = 20;
        for i in 0..grid_size {
            for j in 0..grid_size {
                let lat_ratio = i as f64 / (grid_size - 1) as f64;
                let lon_ratio = j as f64 / (grid_size - 1) as f64;

                let lat = self.bounding_box.min.lat + lat_ratio * height;
                let lon = self.bounding_box.min.lon + lon_ratio * width;

                if let Ok(point) = GeoPoint::try_new(lat, lon) {
                    let doc_id = (i * grid_size + j + 2000) as u64;
                    candidates.push((doc_id, point));
                }
            }
        }

        // Add some points outside the box for testing boundary conditions
        let expansion_factor = 0.1;
        let expanded_width = width * (1.0 + expansion_factor);
        let expanded_height = height * (1.0 + expansion_factor);

        for i in 0..10 {
            let angle = (i as f64 / 10.0) * 2.0 * std::f64::consts::PI;
            let lat_offset = angle.sin() * expanded_height / 2.0;
            let lon_offset = angle.cos() * expanded_width / 2.0;

            let center = self.bounding_box.center();
            if let Ok(point) = GeoPoint::try_new(center.lat + lat_offset, center.lon + lon_offset) {
                let doc_id = (i + 3000) as u64;
                candidates.push((doc_id, point));
            }
        }

        candidates
    }

    /// Calculate relevance score for points within the bounding box.
    fn calculate_bounding_box_score(&self, point: &GeoPoint) -> f32 {
        let center = self.bounding_box.center();
        let (width, height) = self.bounding_box.dimensions();

        // Distance from center as a fraction of the bounding box diagonal
        let distance_to_center = center.distance_to(point);
        let diagonal_m = ((width * 111_000.0).powi(2) + (height * 111_000.0).powi(2)).sqrt();
        let normalized_distance = distance_to_center / diagonal_m;

        // Base score: higher for points closer to center
        let base_score = (1.0 - normalized_distance.min(1.0)) as f32;

        // Bonus for points near edges or corners (depending on use case)
        let edge_bonus = self.calculate_edge_proximity_bonus(point);

        // Corner bonus for strategic locations
        let corner_bonus = self.calculate_corner_bonus(point);

        (base_score + edge_bonus + corner_bonus).min(1.0)
    }

    /// Calculate bonus for points near edges of the bounding box.
    fn calculate_edge_proximity_bonus(&self, point: &GeoPoint) -> f32 {
        let (width, height) = self.bounding_box.dimensions();
        let edge_threshold = 0.1; // 10% of dimension

        let lat_distance_from_edge =
            (point.lat - self.bounding_box.min.lat).min(self.bounding_box.max.lat - point.lat);
        let lon_distance_from_edge =
            (point.lon - self.bounding_box.min.lon).min(self.bounding_box.max.lon - point.lon);

        let lat_edge_proximity = if lat_distance_from_edge < height * edge_threshold {
            0.05
        } else {
            0.0
        };
        let lon_edge_proximity = if lon_distance_from_edge < width * edge_threshold {
            0.05
        } else {
            0.0
        };

        lat_edge_proximity + lon_edge_proximity
    }

    /// Calculate bonus for points near corners of the bounding box.
    fn calculate_corner_bonus(&self, point: &GeoPoint) -> f32 {
        let bbox = &self.bounding_box;
        let corners = [
            bbox.min,                                  // SW
            GeoPoint::new(bbox.max.lat, bbox.min.lon), // NW
            bbox.max,                                  // NE
            GeoPoint::new(bbox.min.lat, bbox.max.lon), // SE
        ];

        let corner_threshold_m = 1_000.0; // Within 1 km (= 1 000 m) of corner
        let mut min_corner_distance = f64::INFINITY;

        for corner in &corners {
            let distance = point.distance_to(corner);
            min_corner_distance = min_corner_distance.min(distance);
        }

        if min_corner_distance < corner_threshold_m {
            0.1 // Corner bonus
        } else {
            0.0
        }
    }
}

impl Query for GeoBoundingBoxQuery {
    fn matcher(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Matcher>> {
        let matches = self.find_matches(reader)?;
        Ok(Box::new(GeoMatcher::new(matches)))
    }

    fn scorer(&self, reader: &dyn LexicalIndexReader) -> Result<Box<dyn Scorer>> {
        let matches = self.find_matches(reader)?;
        Ok(Box::new(GeoScorer::new(matches, self.boost)))
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
            "GeoBoundingBoxQuery(field: {}, bounds: {:?})",
            self.field, self.bounding_box
        )
    }

    fn is_empty(&self, _reader: &dyn LexicalIndexReader) -> Result<bool> {
        let (width, height) = self.bounding_box.dimensions();
        Ok(width <= 0.0 || height <= 0.0)
    }

    fn cost(&self, reader: &dyn LexicalIndexReader) -> Result<u64> {
        let doc_count = reader.doc_count() as u32;
        Ok(doc_count as u64)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A match found by geographical search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoMatch {
    /// Document ID
    pub doc_id: u64,
    /// Geographical point of the document
    pub point: GeoPoint,
    /// Distance from query center in meters
    pub distance_m: f64,
    /// Relevance score based on distance
    pub relevance_score: f32,
}

/// Matcher for geographical queries.
#[derive(Debug)]
pub struct GeoMatcher {
    /// Matching documents in order of relevance (distance-sorted)
    matches: Vec<GeoMatch>,
    /// Current iteration position
    current_index: usize,
}

impl GeoMatcher {
    /// Create a new geo matcher.
    pub fn new(mut matches: Vec<GeoMatch>) -> Self {
        // Sort matches by distance (closest first)
        matches.sort_by(|a, b| {
            a.distance_m
                .partial_cmp(&b.distance_m)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        GeoMatcher {
            matches,
            current_index: 0,
        }
    }
}

impl Matcher for GeoMatcher {
    fn doc_id(&self) -> u64 {
        if self.current_index >= self.matches.len() {
            u64::MAX
        } else {
            self.matches[self.current_index].doc_id
        }
    }

    fn next(&mut self) -> Result<bool> {
        self.current_index += 1;
        if self.current_index < self.matches.len() {
            Ok(true)
        } else {
            self.current_index = self.matches.len();
            Ok(false)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        // Find first document ID >= target in order
        while self.current_index < self.matches.len() {
            let doc_id = self.matches[self.current_index].doc_id;
            if doc_id >= target {
                return Ok(true);
            }
            self.current_index += 1;
        }
        Ok(false)
    }

    fn cost(&self) -> u64 {
        self.matches.len() as u64
    }

    fn is_exhausted(&self) -> bool {
        self.current_index >= self.matches.len()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Scorer for geographical queries.
#[derive(Debug)]
pub struct GeoScorer {
    /// Document scores based on geographical relevance
    doc_scores: HashMap<u64, f32>,
    /// Query boost factor
    boost: f32,
}

impl GeoScorer {
    /// Create a new geo scorer.
    pub fn new(matches: Vec<GeoMatch>, boost: f32) -> Self {
        let mut doc_scores = HashMap::new();

        for geo_match in matches {
            doc_scores.insert(geo_match.doc_id, geo_match.relevance_score);
        }

        GeoScorer { doc_scores, boost }
    }
}

impl Scorer for GeoScorer {
    fn score(&self, doc_id: u64, _term_freq: f32, _field_length: Option<f32>) -> f32 {
        self.doc_scores.get(&doc_id).unwrap_or(&0.0) * self.boost
    }

    fn boost(&self) -> f32 {
        self.boost
    }

    fn set_boost(&mut self, boost: f32) {
        self.boost = boost;
    }

    fn max_score(&self) -> f32 {
        self.doc_scores
            .values()
            .fold(0.0_f32, |max, &score| max.max(score))
            * self.boost
    }

    fn name(&self) -> &'static str {
        "GeoScorer"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_point_creation() {
        let point = GeoPoint::try_new(40.7128, -74.0060).unwrap(); // New York City
        assert_eq!(point.lat, 40.7128);
        assert_eq!(point.lon, -74.0060);

        // Test invalid coordinates
        assert!(GeoPoint::try_new(91.0, 0.0).is_err()); // Invalid latitude
        assert!(GeoPoint::try_new(0.0, 181.0).is_err()); // Invalid longitude
    }

    #[test]
    fn test_geo_distance_calculation() {
        let nyc = GeoPoint::try_new(40.7128, -74.0060).unwrap();
        let la = GeoPoint::try_new(34.0522, -118.2437).unwrap();

        let distance = nyc.distance_to(&la);
        // Distance between NYC and LA is approximately 3 944 km (3 944 000 m)
        assert!((distance - 3_944_000.0).abs() < 100_000.0); // Allow ~100 km tolerance
    }

    #[test]
    fn test_geo_bearing() {
        let nyc = GeoPoint::try_new(40.7128, -74.0060).unwrap();
        let la = GeoPoint::try_new(34.0522, -118.2437).unwrap();

        let bearing = nyc.bearing_to(&la);
        // Bearing from NYC to LA should be roughly west (around 270 degrees)
        assert!(bearing > 200.0 && bearing < 300.0);
    }

    #[test]
    fn test_geo_bounding_box() {
        let min = GeoPoint::try_new(40.0, -75.0).unwrap();
        let max = GeoPoint::try_new(41.0, -74.0).unwrap();
        let bbox = GeoBoundingBox::new(min, max).unwrap();

        let inside_point = GeoPoint::try_new(40.5, -74.5).unwrap();
        let outside_point = GeoPoint::try_new(42.0, -73.0).unwrap();

        assert!(bbox.contains(&inside_point));
        assert!(!bbox.contains(&outside_point));

        let center = bbox.center();
        assert_eq!(center.lat, 40.5);
        assert_eq!(center.lon, -74.5);
    }

    #[test]
    fn test_geo_distance_query() {
        let center = GeoPoint::try_new(40.7128, -74.0060).unwrap();
        let query = GeoDistanceQuery::new("location", center, 10_000.0).with_boost(1.5);

        assert_eq!(query.field(), "location");
        assert_eq!(query.center(), center);
        assert_eq!(query.distance_m(), 10_000.0);
        assert_eq!(query.boost(), 1.5);
    }

    #[test]
    fn test_geo_distance_scoring() {
        let center = GeoPoint::try_new(0.0, 0.0).unwrap();
        let query = GeoDistanceQuery::new("location", center, 10_000.0);

        // Test scoring at different distances (meters)
        assert_eq!(query.calculate_distance_score(0.0), 1.0); // At center
        assert_eq!(query.calculate_distance_score(5_000.0), 0.5); // Half distance
        assert_eq!(query.calculate_distance_score(10_000.0), 0.0); // At max distance
        assert_eq!(query.calculate_distance_score(15_000.0), 0.0); // Beyond max distance
    }

    #[test]
    fn test_geo_bounding_box_query() {
        let min = GeoPoint::try_new(40.0, -75.0).unwrap();
        let max = GeoPoint::try_new(41.0, -74.0).unwrap();
        let bbox = GeoBoundingBox::new(min, max).unwrap();
        let query = GeoBoundingBoxQuery::new("location", bbox);

        assert_eq!(query.field(), "location");
        assert_eq!(query.bounding_box().min, min);
        assert_eq!(query.bounding_box().max, max);
    }

    #[test]
    fn test_geo_distance_query_within_radius_factory() {
        let query =
            GeoDistanceQuery::within_radius("location", 40.7128, -74.0060, 10_000.0).unwrap();

        assert_eq!(query.field(), "location");
        assert_eq!(query.center().lat, 40.7128);
        assert_eq!(query.center().lon, -74.0060);
        assert_eq!(query.distance_m(), 10_000.0);
    }

    #[test]
    fn test_geo_distance_query_within_radius_invalid_lat() {
        // Latitude outside [-90, 90] is rejected by GeoPoint::try_new.
        let err = GeoDistanceQuery::within_radius("location", 95.0, 0.0, 10_000.0).unwrap_err();
        assert!(format!("{err}").to_lowercase().contains("lat"));
    }

    #[test]
    fn test_geo_bounding_box_query_within_bounding_box_factory() {
        let query =
            GeoBoundingBoxQuery::within_bounding_box("location", 40.0, -75.0, 41.0, -74.0).unwrap();

        assert_eq!(query.field(), "location");
        assert_eq!(query.bounding_box().min.lat, 40.0);
        assert_eq!(query.bounding_box().min.lon, -75.0);
        assert_eq!(query.bounding_box().max.lat, 41.0);
        assert_eq!(query.bounding_box().max.lon, -74.0);
    }

    #[test]
    fn test_geo_bounding_box_query_within_bounding_box_inverted() {
        // min > max is rejected by GeoBoundingBox::new.
        let err = GeoBoundingBoxQuery::within_bounding_box("location", 50.0, 0.0, 40.0, 10.0)
            .unwrap_err();
        // Error mentions either bounding-box or min/max.
        let msg = format!("{err}").to_lowercase();
        assert!(msg.contains("bounding") || msg.contains("min") || msg.contains("max"));
    }

    #[test]
    fn test_geo_matcher() {
        let matches = vec![
            GeoMatch {
                doc_id: 3,
                point: GeoPoint::try_new(0.0, 0.0).unwrap(),
                distance_m: 1_000.0,
                relevance_score: 0.9,
            },
            GeoMatch {
                doc_id: 1,
                point: GeoPoint::try_new(0.0, 0.0).unwrap(),
                distance_m: 2_000.0,
                relevance_score: 0.8,
            },
        ];

        let mut matcher = GeoMatcher::new(matches);

        // Should return documents in distance-sorted order (closest first)
        // After sorting: doc_id: 3 (1 km) comes before doc_id: 1 (2 km)
        assert_eq!(matcher.doc_id(), 3); // Initial position: closest document

        assert!(matcher.next().unwrap()); // Move to next
        assert_eq!(matcher.doc_id(), 1); // Second document (farther)

        assert!(!matcher.next().unwrap()); // No more documents
    }

    #[test]
    fn test_geo_scorer() {
        let matches = vec![GeoMatch {
            doc_id: 1,
            point: GeoPoint::try_new(0.0, 0.0).unwrap(),
            distance_m: 1_000.0,
            relevance_score: 0.9,
        }];

        let scorer = GeoScorer::new(matches, 2.0);

        assert_eq!(scorer.score(1, 1.0, None), 0.9 * 2.0);
        assert_eq!(scorer.score(999, 1.0, None), 0.0); // Non-existent document
        assert_eq!(scorer.max_score(), 0.9 * 2.0);
        assert_eq!(scorer.name(), "GeoScorer");
    }

    #[test]
    fn test_enhanced_distance_scoring() {
        let center = GeoPoint::try_new(40.7128, -74.0060).unwrap(); // NYC
        let query = GeoDistanceQuery::new("location", center, 10_000.0);

        // Test point very close to centre (50 m away)
        let close_point = GeoPoint::try_new(40.7130, -74.0062).unwrap();
        let close_score = query.calculate_distance_score_enhanced(50.0, &close_point);

        // Test point at moderate distance (1 km)
        let mid_point = GeoPoint::try_new(40.7200, -74.0100).unwrap();
        let mid_score = query.calculate_distance_score_enhanced(1_000.0, &mid_point);

        // Test point at almost max distance (9 km)
        let far_point = GeoPoint::try_new(40.8000, -74.1000).unwrap();
        let far_score = query.calculate_distance_score_enhanced(9_000.0, &far_point);

        // Scores should decrease with distance, and close points should get precision bonus
        assert!(close_score > mid_score);
        assert!(mid_score > far_score);
        assert!(close_score > 0.9); // Should get precision bonus
    }

    #[test]
    fn test_bounding_box_enhanced_functionality() {
        let min = GeoPoint::try_new(40.0, -75.0).unwrap();
        let max = GeoPoint::try_new(41.0, -74.0).unwrap();
        let bbox = GeoBoundingBox::new(min, max).unwrap();
        let query = GeoBoundingBoxQuery::new("location", bbox);

        // Test that generated candidates include points within bounds
        let candidates = query.generate_bounding_box_candidates();
        assert!(!candidates.is_empty());

        // Test that some candidates are within the bounding box
        let within_count = candidates
            .iter()
            .filter(|(_, point)| query.bounding_box().contains(point))
            .count();
        assert!(within_count > 0);

        // Test scoring for points within bounding box
        let center_point = query.bounding_box().center();
        let center_score = query.calculate_bounding_box_score(&center_point);

        let corner_point = query.bounding_box().min;
        let corner_score = query.calculate_bounding_box_score(&corner_point);

        // Center should generally score higher than corners
        assert!(center_score >= corner_score);
    }

    #[test]
    fn test_spatial_bounding_box_creation() {
        let center = GeoPoint::try_new(40.7128, -74.0060).unwrap(); // NYC
        let query = GeoDistanceQuery::new("location", center, 5_000.0); // 5 km radius

        let bbox = query.create_bounding_box();

        // Check that the bounding box contains the center
        assert!(bbox.contains(&center));

        // Check that the bounding box is roughly the right size
        let (width, height) = bbox.dimensions();
        assert!(width > 0.0 && width < 1.0); // Should be less than 1 degree
        assert!(height > 0.0 && height < 1.0);

        // The center should be approximately in the middle of the bounding box
        // (within ~1 km).
        let bbox_center = bbox.center();
        let center_distance = center.distance_to(&bbox_center);
        assert!(center_distance < 1_000.0);
    }

    #[test]
    fn test_geographic_relevance_calculation() {
        let center = GeoPoint::try_new(40.7128, -74.0060).unwrap();
        let query = GeoDistanceQuery::new("location", center, 10_000.0);

        // Test temperate zone bonus
        let temperate_point = GeoPoint::try_new(45.0, 0.0).unwrap(); // Temperate zone
        let tropical_point = GeoPoint::try_new(10.0, 0.0).unwrap(); // Tropical zone

        let temperate_bonus = query.calculate_geographic_relevance(&temperate_point);
        let tropical_bonus = query.calculate_geographic_relevance(&tropical_point);

        assert!(temperate_bonus > tropical_bonus);

        // Test equator bonus
        let equator_point = GeoPoint::try_new(2.0, 0.0).unwrap(); // Near equator
        let non_equator_point = GeoPoint::try_new(45.0, 0.0).unwrap();

        let equator_geo_bonus = query.calculate_geographic_relevance(&equator_point);
        let non_equator_geo_bonus = query.calculate_geographic_relevance(&non_equator_point);

        // Both should have some bonus, but for different reasons
        assert!(equator_geo_bonus > 0.0);
        assert!(non_equator_geo_bonus > 0.0);
    }
}
