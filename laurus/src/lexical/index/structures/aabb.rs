//! Axis-aligned bounding box (AABB) used by the BKD tree's
//! `IntersectVisitor` API.
//!
//! All bounds are interpreted as **closed** intervals — `[min[d], max[d]]` —
//! and unbounded coordinates are represented with `f64::NEG_INFINITY` /
//! `f64::INFINITY`. Callers that need half-open semantics (e.g. range queries
//! with `>` rather than `>=`) layer that on top of `AABB`, in their visitor
//! implementation, rather than in the AABB itself.

use crate::error::{LaurusError, Result};

/// Axis-aligned bounding box with `min` and `max` per dimension.
///
/// Both `min` and `max` slices have the same length, which equals the
/// dimensionality of the box (`num_dims`). For every dimension `d`,
/// `min[d] <= max[d]` must hold; the constructor validates this.
#[derive(Debug, Clone, PartialEq)]
pub struct AABB {
    min: Vec<f64>,
    max: Vec<f64>,
}

impl AABB {
    /// Create a new AABB from per-dimension `min` / `max` vectors.
    ///
    /// # Errors
    /// - `min.len() != max.len()` (dimensionality mismatch).
    /// - `min.is_empty()` (a 0-dimensional box has no meaning).
    /// - any `min[d] > max[d]` (degenerate box).
    /// - any coordinate is `NaN`.
    pub fn new(min: Vec<f64>, max: Vec<f64>) -> Result<Self> {
        if min.len() != max.len() {
            return Err(LaurusError::index(format!(
                "AABB dimension mismatch: min has {} dims, max has {} dims",
                min.len(),
                max.len()
            )));
        }
        if min.is_empty() {
            return Err(LaurusError::index(
                "AABB requires at least one dimension".to_string(),
            ));
        }
        for d in 0..min.len() {
            if min[d].is_nan() || max[d].is_nan() {
                return Err(LaurusError::index(format!(
                    "AABB contains NaN at dimension {d}"
                )));
            }
            if min[d] > max[d] {
                return Err(LaurusError::index(format!(
                    "AABB invalid at dimension {d}: min={} > max={}",
                    min[d], max[d]
                )));
            }
        }
        Ok(AABB { min, max })
    }

    /// Construct an AABB that spans the entire `f64` range on every
    /// dimension — `[NEG_INFINITY, INFINITY]`. Useful as the initial
    /// "match everything" query.
    pub fn unbounded(num_dims: usize) -> Self {
        AABB {
            min: vec![f64::NEG_INFINITY; num_dims],
            max: vec![f64::INFINITY; num_dims],
        }
    }

    /// Number of dimensions covered by this AABB.
    #[inline]
    pub fn num_dims(&self) -> usize {
        self.min.len()
    }

    /// Per-dimension lower bounds.
    #[inline]
    pub fn min(&self) -> &[f64] {
        &self.min
    }

    /// Per-dimension upper bounds.
    #[inline]
    pub fn max(&self) -> &[f64] {
        &self.max
    }

    /// Whether `point` lies on or inside this AABB on every dimension.
    /// Returns `false` if `point` has a different dimensionality.
    pub fn contains_point(&self, point: &[f64]) -> bool {
        if point.len() != self.min.len() {
            return false;
        }
        for (d, &v) in point.iter().enumerate() {
            if v < self.min[d] || v > self.max[d] {
                return false;
            }
        }
        true
    }

    /// Whether `other` is entirely inside `self` on every dimension.
    /// Returns `false` if dimensionalities differ.
    pub fn contains_aabb(&self, other: &AABB) -> bool {
        if other.num_dims() != self.num_dims() {
            return false;
        }
        for d in 0..self.min.len() {
            if other.min[d] < self.min[d] || other.max[d] > self.max[d] {
                return false;
            }
        }
        true
    }

    /// Whether `self` and `other` share at least one point.
    /// Returns `false` if dimensionalities differ.
    pub fn intersects(&self, other: &AABB) -> bool {
        if other.num_dims() != self.num_dims() {
            return false;
        }
        for d in 0..self.min.len() {
            if self.max[d] < other.min[d] || self.min[d] > other.max[d] {
                return false;
            }
        }
        true
    }

    /// Squared Euclidean distance from `point` to the *nearest* point inside
    /// (or on) the AABB. Returns `0.0` when `point` itself lies inside the
    /// box. The squared form avoids a `sqrt` in tight pruning loops; callers
    /// that need a real distance can apply `.sqrt()` to the result.
    ///
    /// Returns `f64::INFINITY` if `point.len()` does not match
    /// [`AABB::num_dims`].
    pub fn min_distance_sq_to_point(&self, point: &[f64]) -> f64 {
        if point.len() != self.min.len() {
            return f64::INFINITY;
        }
        let mut acc = 0.0;
        for (d, &p) in point.iter().enumerate() {
            let lo = self.min[d];
            let hi = self.max[d];
            let delta = if p < lo {
                lo - p
            } else if p > hi {
                p - hi
            } else {
                0.0
            };
            acc += delta * delta;
        }
        acc
    }

    /// Squared Euclidean distance from `point` to the *farthest* point
    /// inside the AABB. For each axis we pick whichever corner (`min[d]`
    /// or `max[d]`) is farther from `point[d]`. Pairs with
    /// [`AABB::min_distance_sq_to_point`] for the BKD's sphere-vs-cell
    /// containment test: a cell whose `max_distance_sq_to_point(center)
    /// <= radius²` lies fully inside the sphere.
    ///
    /// Returns `f64::NEG_INFINITY` if `point.len()` does not match
    /// [`AABB::num_dims`] (a sentinel that will never satisfy any
    /// `<= radius²` check, so misuse never produces a false-positive
    /// Inside classification).
    pub fn max_distance_sq_to_point(&self, point: &[f64]) -> f64 {
        if point.len() != self.min.len() {
            return f64::NEG_INFINITY;
        }
        let mut acc = 0.0;
        for (d, &p) in point.iter().enumerate() {
            let dlo = (p - self.min[d]).abs();
            let dhi = (p - self.max[d]).abs();
            let far = dlo.max(dhi);
            acc += far * far;
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_validates_dimension_mismatch() {
        let err = AABB::new(vec![0.0, 0.0], vec![1.0]).unwrap_err();
        assert!(format!("{err:?}").contains("dimension mismatch"));
    }

    #[test]
    fn new_validates_empty() {
        let err = AABB::new(vec![], vec![]).unwrap_err();
        assert!(format!("{err:?}").contains("at least one dimension"));
    }

    #[test]
    fn new_validates_min_greater_than_max() {
        let err = AABB::new(vec![5.0], vec![3.0]).unwrap_err();
        assert!(format!("{err:?}").contains("min=5 > max=3"));
    }

    #[test]
    fn new_rejects_nan() {
        let err = AABB::new(vec![f64::NAN], vec![1.0]).unwrap_err();
        assert!(format!("{err:?}").contains("NaN"));
    }

    #[test]
    fn unbounded_uses_infinities() {
        let aabb = AABB::unbounded(3);
        assert_eq!(aabb.num_dims(), 3);
        for d in 0..3 {
            assert_eq!(aabb.min()[d], f64::NEG_INFINITY);
            assert_eq!(aabb.max()[d], f64::INFINITY);
        }
    }

    #[test]
    fn contains_point_handles_boundary_inclusively() {
        let aabb = AABB::new(vec![0.0, 0.0], vec![10.0, 10.0]).unwrap();
        assert!(aabb.contains_point(&[5.0, 5.0]));
        assert!(aabb.contains_point(&[0.0, 10.0])); // boundary
        assert!(!aabb.contains_point(&[10.1, 5.0]));
        assert!(!aabb.contains_point(&[5.0]));
    }

    #[test]
    fn contains_aabb_strict_subset() {
        let outer = AABB::new(vec![0.0, 0.0], vec![10.0, 10.0]).unwrap();
        let inner = AABB::new(vec![1.0, 1.0], vec![9.0, 9.0]).unwrap();
        let touching = AABB::new(vec![0.0, 0.0], vec![10.0, 10.0]).unwrap();
        let outside = AABB::new(vec![5.0, 5.0], vec![15.0, 15.0]).unwrap();
        assert!(outer.contains_aabb(&inner));
        assert!(outer.contains_aabb(&touching));
        assert!(!outer.contains_aabb(&outside));
    }

    #[test]
    fn intersects_disjoint_and_overlapping() {
        let a = AABB::new(vec![0.0, 0.0], vec![5.0, 5.0]).unwrap();
        let overlapping = AABB::new(vec![3.0, 3.0], vec![8.0, 8.0]).unwrap();
        let touching = AABB::new(vec![5.0, 5.0], vec![10.0, 10.0]).unwrap();
        let disjoint = AABB::new(vec![6.0, 6.0], vec![10.0, 10.0]).unwrap();
        assert!(a.intersects(&overlapping));
        // Touching boxes share their corner — closed intervals intersect.
        assert!(a.intersects(&touching));
        assert!(!a.intersects(&disjoint));
    }

    #[test]
    fn min_distance_sq_to_point_is_zero_when_inside() {
        let aabb = AABB::new(vec![0.0, 0.0, 0.0], vec![10.0, 10.0, 10.0]).unwrap();
        assert_eq!(aabb.min_distance_sq_to_point(&[5.0, 5.0, 5.0]), 0.0);
        // On the boundary: still inside, distance 0.
        assert_eq!(aabb.min_distance_sq_to_point(&[0.0, 10.0, 5.0]), 0.0);
    }

    #[test]
    fn min_distance_sq_to_point_outside_axes() {
        // 3D AABB at origin, point 3 units beyond each axis.
        let aabb = AABB::new(vec![0.0, 0.0, 0.0], vec![10.0, 10.0, 10.0]).unwrap();
        // Point 3 units below the min corner on x only: dist² = 9.
        assert_eq!(aabb.min_distance_sq_to_point(&[-3.0, 5.0, 5.0]), 9.0);
        // Point 3 units past max on every axis: dist² = 3² + 3² + 3² = 27.
        assert_eq!(aabb.min_distance_sq_to_point(&[13.0, 13.0, 13.0]), 27.0);
    }

    #[test]
    fn min_distance_sq_to_point_dim_mismatch_is_infinity() {
        let aabb = AABB::new(vec![0.0, 0.0, 0.0], vec![10.0, 10.0, 10.0]).unwrap();
        assert!(aabb.min_distance_sq_to_point(&[5.0]).is_infinite());
    }

    #[test]
    fn max_distance_sq_to_point_picks_far_corner() {
        // 1D AABB [0, 10]; point at 1 → far corner is 10, dist 9.
        let aabb = AABB::new(vec![0.0], vec![10.0]).unwrap();
        assert_eq!(aabb.max_distance_sq_to_point(&[1.0]), 81.0);
        // Point at 6 → far corner is 0, dist 6.
        assert_eq!(aabb.max_distance_sq_to_point(&[6.0]), 36.0);

        // 3D AABB [0, 10]³; center query (5, 5, 5) → every corner 5√3 away
        // → squared = 75.
        let aabb3 = AABB::new(vec![0.0, 0.0, 0.0], vec![10.0, 10.0, 10.0]).unwrap();
        assert_eq!(aabb3.max_distance_sq_to_point(&[5.0, 5.0, 5.0]), 75.0);
    }

    #[test]
    fn min_le_max_distance_sq() {
        // Sanity: min_distance_sq is always <= max_distance_sq for a
        // matching-dim point.
        let aabb = AABB::new(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]).unwrap();
        for point in [
            [0.0, 0.0, 0.0],
            [2.5, 3.5, 4.5],
            [10.0, 0.0, -5.0],
            [4.0, 5.0, 6.0],
        ] {
            let lo = aabb.min_distance_sq_to_point(&point);
            let hi = aabb.max_distance_sq_to_point(&point);
            assert!(
                lo <= hi,
                "min_dist_sq ({lo}) must be <= max_dist_sq ({hi}) for point {point:?}"
            );
        }
    }
}
