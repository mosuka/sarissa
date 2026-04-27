//! WGS84 ↔ ECEF coordinate conversions.
//!
//! [`GeoEcefPoint`] (3D Cartesian, `(x, y, z)` meters) is the storage
//! format for `Geo3d` fields, but most user input arrives as WGS84
//! geographic coordinates (`(lat°, lon°, height_m)`). This module
//! supplies the canonical pair of conversions:
//!
//! - [`wgs84_to_ecef`] — geographic to Cartesian, closed-form.
//! - [`ecef_to_wgs84`] — Cartesian back to geographic, using Bowring's
//!   1985 closed-form as the seed for three Newton-Raphson refinement
//!   iterations. The seed alone has ~1 mm error at LEO altitudes; one
//!   iteration cuts that to ~1 nm and three iterations converge to
//!   floating-point precision well past geosynchronous orbit. A
//!   meridian-formula fallback handles the `1 / cos(lat)` singularity
//!   at the poles.
//!
//! # WGS84 ellipsoid constants
//!
//! Defined in NIMA TR 8350.2 (and reproduced by EPSG:4326). Exposed as
//! `pub const`s so external code (tests, tooling) can reference the
//! exact values used by laurus rather than re-deriving them.
//!
//! # Numeric properties
//!
//! At any point from sub-surface up to far beyond LEO altitudes the
//! implementation round-trips with sub-µm error on every axis (well
//! inside the "sub-mm" precision the project promises). The unit tests
//! sweep a coarse lat / lon / height grid and assert a 10 nm tolerance.

use crate::data::GeoEcefPoint;

/// WGS84 semi-major axis (equatorial radius), meters.
pub const WGS84_A: f64 = 6_378_137.0;

/// WGS84 flattening, dimensionless. Exactly `1 / 298.257_223_563`.
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;

/// WGS84 semi-minor axis (polar radius), meters. Equal to `a * (1 - f)`.
pub const WGS84_B: f64 = WGS84_A * (1.0 - WGS84_F);

/// WGS84 first eccentricity squared, `e² = f * (2 - f)`.
pub const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);

/// WGS84 second eccentricity squared, `e'² = e² / (1 - e²)`.
/// Used by Bowring's reverse formula.
pub const WGS84_E_PRIME_SQ: f64 = WGS84_E2 / (1.0 - WGS84_E2);

/// Convert geographic (WGS84) coordinates to ECEF Cartesian.
///
/// # Arguments
/// - `lat_deg` — latitude in degrees, `[-90, 90]`. Values outside the
///   range are not rejected; they are handled by the trig identities,
///   but produce unusual ECEF positions that callers should validate
///   upstream.
/// - `lon_deg` — longitude in degrees. Any real value is accepted;
///   `lon_deg + 360.0` produces the same Cartesian point as `lon_deg`.
/// - `height_m` — height above the WGS84 ellipsoid in meters
///   (positive away from the ellipsoid, negative below it).
///
/// # Returns
/// A `GeoEcefPoint` whose `(x, y, z)` are the Cartesian coordinates
/// (meters) of the input.
///
/// # Formula
/// ```text
/// N = a / sqrt(1 - e² sin²(lat))     (prime vertical radius)
/// x = (N + h) cos(lat) cos(lon)
/// y = (N + h) cos(lat) sin(lon)
/// z = (N (1 - e²) + h) sin(lat)
/// ```
pub fn wgs84_to_ecef(lat_deg: f64, lon_deg: f64, height_m: f64) -> GeoEcefPoint {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_lon = lon.sin();
    let cos_lon = lon.cos();

    let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
    let x = (n + height_m) * cos_lat * cos_lon;
    let y = (n + height_m) * cos_lat * sin_lon;
    let z = (n * (1.0 - WGS84_E2) + height_m) * sin_lat;
    GeoEcefPoint::new(x, y, z)
}

/// Convert ECEF Cartesian coordinates back to geographic (WGS84).
///
/// # Returns
/// A tuple `(lat_deg, lon_deg, height_m)`:
/// - `lat_deg` ∈ `[-90, 90]`
/// - `lon_deg` ∈ `(-180, 180]` (the value returned by `f64::atan2`,
///   no manual normalisation is performed beyond what `atan2` does)
/// - `height_m` is the height above the WGS84 ellipsoid in meters
///
/// # Algorithm
/// Bowring's 1985 closed-form gives the seed latitude, then a few
/// Newton-Raphson refinements take it to floating-point precision —
/// the closed-form alone has ~1 mm error at LEO altitudes; a single
/// iteration cuts that to ~1 nm. The implementation runs three
/// iterations unconditionally; each is cheap (one `sqrt`, two trig
/// calls) and keeps sub-mm precision well past geosynchronous orbit.
///
/// Near the poles (`p ≈ 0`) the standard `h = p / cos(lat) - N`
/// formula diverges, so the height falls back to the meridian form
/// `h = z / sin(lat) - N (1 - e²)`.
pub fn ecef_to_wgs84(p: &GeoEcefPoint) -> (f64, f64, f64) {
    let GeoEcefPoint { x, y, z } = *p;

    // Origin: undefined geodetic coordinate. Return (0, 0, -a) so the
    // result is at least round-trippable to the centre point of the
    // ellipsoid (height = -a places the point at the origin).
    if x == 0.0 && y == 0.0 && z == 0.0 {
        return (0.0, 0.0, -WGS84_A);
    }

    let p_xy = (x * x + y * y).sqrt();
    let lon = y.atan2(x);

    // Bowring's parametric-latitude seed.
    let theta = (z * WGS84_A).atan2(p_xy * WGS84_B);
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();
    let lat_num = z + WGS84_E_PRIME_SQ * WGS84_B * sin_theta.powi(3);
    let lat_den = p_xy - WGS84_E2 * WGS84_A * cos_theta.powi(3);
    let mut lat = lat_num.atan2(lat_den);

    // Newton-Raphson refinement: the seed has ~1 mm error at LEO
    // altitudes; each iteration improves by several decimal digits.
    // Three iterations comfortably cover everything from sub-surface
    // to far beyond geosynchronous orbit.
    for _ in 0..3 {
        let sin_lat = lat.sin();
        let cos_lat = lat.cos();
        let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
        // Height needed to refine lat — use the meridian form near the
        // poles to dodge the `1 / cos(lat)` blow-up.
        let h = if p_xy > 1.0 {
            p_xy / cos_lat - n
        } else {
            z / sin_lat - n * (1.0 - WGS84_E2)
        };
        // Update latitude using the corrected `N + h` term.
        let denom = 1.0 - WGS84_E2 * n / (n + h);
        lat = z.atan2(p_xy * denom);
    }

    // Final height with the converged lat.
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
    let height = if p_xy > 1.0 {
        p_xy / cos_lat - n
    } else {
        z / sin_lat - n * (1.0 - WGS84_E2)
    };

    (lat.to_degrees(), lon.to_degrees(), height)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip tolerance: 10 nanometres on each axis — well inside
    /// the "sub-mm" figure quoted in the module-level docs and tight
    /// enough to catch any algorithmic regression. The Newton-Raphson
    /// refinement loop in `ecef_to_wgs84` converges to floating-point
    /// precision after three iterations, so this passes everywhere
    /// from sub-surface to far beyond LEO.
    const ROUNDTRIP_TOL_M: f64 = 1.0e-8;
    /// Latitude / longitude tolerance: `1e-12` degrees ≈ 1 µm at the
    /// surface — a few orders of magnitude tighter than the height
    /// tolerance, again to catch regressions early.
    const ANGLE_TOL_DEG: f64 = 1.0e-12;

    fn approx_eq(a: f64, b: f64, tol: f64, label: &str) {
        let diff = (a - b).abs();
        assert!(diff <= tol, "{label}: {a} vs {b} (diff {diff} > tol {tol})");
    }

    #[test]
    fn wgs84_to_ecef_at_equator_prime_meridian() {
        // (lat=0, lon=0, h=0) → the equatorial radius along +x.
        let p = wgs84_to_ecef(0.0, 0.0, 0.0);
        approx_eq(p.x, WGS84_A, 1e-9, "x");
        approx_eq(p.y, 0.0, 1e-9, "y");
        approx_eq(p.z, 0.0, 1e-9, "z");
    }

    #[test]
    fn wgs84_to_ecef_at_north_pole() {
        // (lat=90, lon=anything, h=0) → on the polar axis at z = b.
        let p = wgs84_to_ecef(90.0, 42.0, 0.0);
        approx_eq(p.x, 0.0, 1e-6, "x");
        approx_eq(p.y, 0.0, 1e-6, "y");
        approx_eq(p.z, WGS84_B, 1e-6, "z");
    }

    #[test]
    fn wgs84_to_ecef_at_south_pole() {
        let p = wgs84_to_ecef(-90.0, -123.0, 0.0);
        approx_eq(p.x, 0.0, 1e-6, "x");
        approx_eq(p.y, 0.0, 1e-6, "y");
        approx_eq(p.z, -WGS84_B, 1e-6, "z");
    }

    #[test]
    fn wgs84_to_ecef_height_on_equator() {
        // At (0, 0, h), x = a + h, y = z = 0.
        let p = wgs84_to_ecef(0.0, 0.0, 10_000.0);
        approx_eq(p.x, WGS84_A + 10_000.0, 1e-9, "x");
        approx_eq(p.y, 0.0, 1e-9, "y");
        approx_eq(p.z, 0.0, 1e-9, "z");
    }

    #[test]
    fn round_trip_equator_prime_meridian() {
        let (lat, lon, h) = ecef_to_wgs84(&wgs84_to_ecef(0.0, 0.0, 0.0));
        approx_eq(lat, 0.0, ANGLE_TOL_DEG, "lat");
        approx_eq(lon, 0.0, ANGLE_TOL_DEG, "lon");
        approx_eq(h, 0.0, ROUNDTRIP_TOL_M, "h");
    }

    #[test]
    fn round_trip_tokyo() {
        // Tokyo Tower-ish coordinates: 35.6586° N, 139.7454° E, 250 m.
        let lat0 = 35.6586;
        let lon0 = 139.7454;
        let h0 = 250.0;
        let (lat, lon, h) = ecef_to_wgs84(&wgs84_to_ecef(lat0, lon0, h0));
        approx_eq(lat, lat0, ANGLE_TOL_DEG, "lat");
        approx_eq(lon, lon0, ANGLE_TOL_DEG, "lon");
        approx_eq(h, h0, ROUNDTRIP_TOL_M, "h");
    }

    #[test]
    fn round_trip_high_altitude_satellite_orbit() {
        // 400 km altitude (rough ISS orbit) at mid-lat.
        let lat0 = 51.6;
        let lon0 = -0.1; // London-ish ground track
        let h0 = 400_000.0;
        let (lat, lon, h) = ecef_to_wgs84(&wgs84_to_ecef(lat0, lon0, h0));
        approx_eq(lat, lat0, ANGLE_TOL_DEG, "lat");
        approx_eq(lon, lon0, ANGLE_TOL_DEG, "lon");
        approx_eq(h, h0, ROUNDTRIP_TOL_M, "h");
    }

    #[test]
    fn round_trip_below_ellipsoid() {
        // Mariana-Trench-ish depth: -10 km below the ellipsoid.
        let lat0 = 11.35;
        let lon0 = 142.20;
        let h0 = -10_000.0;
        let (lat, lon, h) = ecef_to_wgs84(&wgs84_to_ecef(lat0, lon0, h0));
        approx_eq(lat, lat0, ANGLE_TOL_DEG, "lat");
        approx_eq(lon, lon0, ANGLE_TOL_DEG, "lon");
        approx_eq(h, h0, ROUNDTRIP_TOL_M, "h");
    }

    #[test]
    fn round_trip_north_pole() {
        // The polar singularity branch: longitude is undefined at the
        // pole, so we only check lat and h.
        let lat0 = 90.0;
        let h0 = 100.0;
        let p = wgs84_to_ecef(lat0, 0.0, h0);
        let (lat, _lon, h) = ecef_to_wgs84(&p);
        approx_eq(lat, lat0, ANGLE_TOL_DEG, "lat");
        approx_eq(h, h0, 1e-3, "h"); // 1 mm tolerance at the pole
    }

    #[test]
    fn round_trip_south_pole() {
        let lat0 = -90.0;
        let h0 = 50.0;
        let p = wgs84_to_ecef(lat0, 17.0, h0);
        let (lat, _lon, h) = ecef_to_wgs84(&p);
        approx_eq(lat, lat0, ANGLE_TOL_DEG, "lat");
        approx_eq(h, h0, 1e-3, "h");
    }

    #[test]
    fn ecef_to_wgs84_origin_returns_centre_point() {
        // Defensive: (0, 0, 0) → defined as (lat=0, lon=0, h=-a).
        let (lat, lon, h) = ecef_to_wgs84(&GeoEcefPoint::new(0.0, 0.0, 0.0));
        approx_eq(lat, 0.0, ANGLE_TOL_DEG, "lat");
        approx_eq(lon, 0.0, ANGLE_TOL_DEG, "lon");
        approx_eq(h, -WGS84_A, 1e-9, "h");
    }

    #[test]
    fn round_trip_random_grid() {
        // Sweep a coarse lat / lon / height grid and assert sub-µm
        // round-trip on every sample. Catches any axis-specific
        // regression that point tests might miss.
        for &lat0 in &[-89.0, -45.0, -10.0, 0.0, 10.0, 45.0, 89.0] {
            for &lon0 in &[-179.0, -120.0, -45.0, 0.0, 45.0, 120.0, 179.0] {
                for &h0 in &[-5_000.0, 0.0, 100.0, 5_000.0, 100_000.0] {
                    let p = wgs84_to_ecef(lat0, lon0, h0);
                    let (lat, lon, h) = ecef_to_wgs84(&p);
                    approx_eq(lat, lat0, ANGLE_TOL_DEG, "lat");
                    approx_eq(lon, lon0, ANGLE_TOL_DEG, "lon");
                    approx_eq(h, h0, ROUNDTRIP_TOL_M, "h");
                }
            }
        }
    }
}
