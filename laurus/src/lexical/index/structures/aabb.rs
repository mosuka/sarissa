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
}
