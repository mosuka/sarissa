//! Geographical search functionality for location-based queries.

use crate::error::Result;
use crate::index::reader::IndexReader;
use crate::query::{Matcher, Query, Scorer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A geographical point with latitude and longitude.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeoPoint {
    /// Latitude in degrees (-90 to 90)
    pub lat: f64,
    /// Longitude in degrees (-180 to 180)
    pub lon: f64,
}

impl GeoPoint {
    /// Create a new geographical point.
    pub fn new(lat: f64, lon: f64) -> Result<Self> {
        if !(-90.0..=90.0).contains(&lat) {
            return Err(crate::error::SarissaError::other(format!(
                "Invalid latitude: {lat} (must be between -90 and 90)"
            )));
        }
        if !(-180.0..=180.0).contains(&lon) {
            return Err(crate::error::SarissaError::other(format!(
                "Invalid longitude: {lon} (must be between -180 and 180)"
            )));
        }

        Ok(GeoPoint { lat, lon })
    }

    /// Calculate the Haversine distance to another point in kilometers.
    pub fn distance_to(&self, other: &GeoPoint) -> f64 {
        const EARTH_RADIUS_KM: f64 = 6371.0;

        let lat1_rad = self.lat.to_radians();
        let lat2_rad = other.lat.to_radians();
        let delta_lat = (other.lat - self.lat).to_radians();
        let delta_lon = (other.lon - self.lon).to_radians();

        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

        EARTH_RADIUS_KM * c
    }

    /// Calculate the bearing (direction) to another point in degrees.
    pub fn bearing_to(&self, other: &GeoPoint) -> f64 {
        let lat1_rad = self.lat.to_radians();
        let lat2_rad = other.lat.to_radians();
        let delta_lon = (other.lon - self.lon).to_radians();

        let y = delta_lon.sin() * lat2_rad.cos();
        let x = lat1_rad.cos() * lat2_rad.sin() - lat1_rad.sin() * lat2_rad.cos() * delta_lon.cos();

        let bearing_rad = y.atan2(x);
        (bearing_rad.to_degrees() + 360.0) % 360.0
    }

    /// Check if this point is within a rectangular bounding box.
    pub fn within_bounds(&self, min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> bool {
        self.lat >= min_lat && self.lat <= max_lat && self.lon >= min_lon && self.lon <= max_lon
    }
}

/// A geographical bounding box defined by minimum and maximum coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoBoundingBox {
    /// Top-left corner
    pub top_left: GeoPoint,
    /// Bottom-right corner  
    pub bottom_right: GeoPoint,
}

impl GeoBoundingBox {
    /// Create a new bounding box.
    pub fn new(top_left: GeoPoint, bottom_right: GeoPoint) -> Result<Self> {
        if top_left.lat < bottom_right.lat {
            return Err(crate::error::SarissaError::other(
                "Top-left latitude must be greater than bottom-right latitude",
            ));
        }
        if top_left.lon > bottom_right.lon {
            return Err(crate::error::SarissaError::other(
                "Top-left longitude must be less than bottom-right longitude",
            ));
        }

        Ok(GeoBoundingBox {
            top_left,
            bottom_right,
        })
    }

    /// Check if a point is within this bounding box.
    pub fn contains(&self, point: &GeoPoint) -> bool {
        point.within_bounds(
            self.bottom_right.lat,  // min_lat
            self.top_left.lat,      // max_lat  
            self.top_left.lon,      // min_lon
            self.bottom_right.lon,  // max_lon
        )
    }

    /// Get the center point of this bounding box.
    pub fn center(&self) -> GeoPoint {
        let center_lat = (self.top_left.lat + self.bottom_right.lat) / 2.0;
        let center_lon = (self.top_left.lon + self.bottom_right.lon) / 2.0;
        GeoPoint::new(center_lat, center_lon).unwrap() // Should always be valid
    }

    /// Get the width and height of the bounding box in degrees.
    pub fn dimensions(&self) -> (f64, f64) {
        let width = self.bottom_right.lon - self.top_left.lon;
        let height = self.top_left.lat - self.bottom_right.lat;
        (width, height)
    }

    /// Get the maximum distance from the center to any corner of the bounding box.
    pub fn max_distance_from_center(&self) -> f64 {
        let center = self.center();
        let corners = [
            &self.top_left,
            &self.bottom_right,
            &GeoPoint::new(self.top_left.lat, self.bottom_right.lon).unwrap(),
            &GeoPoint::new(self.bottom_right.lat, self.top_left.lon).unwrap(),
        ];
        
        corners
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
    /// Maximum distance in kilometers
    distance_km: f64,
    /// Boost factor for the query
    boost: f32,
}

impl GeoDistanceQuery {
    /// Create a new geo distance query.
    pub fn new<F: Into<String>>(field: F, center: GeoPoint, distance_km: f64) -> Self {
        GeoDistanceQuery {
            field: field.into(),
            center,
            distance_km,
            boost: 1.0,
        }
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

    /// Get the search distance.
    pub fn distance_km(&self) -> f64 {
        self.distance_km
    }

    /// Find matching documents and their distances using spatial indexing.
    pub fn find_matches(&self, reader: &dyn IndexReader) -> Result<Vec<GeoMatch>> {
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
            if distance <= self.distance_km {
                let score = if distance == 0.0 {
                    1.0
                } else {
                    // Simple inverse distance scoring
                    (1.0 - (distance / self.distance_km)).max(0.0) as f32
                };

                matches.push(GeoMatch {
                    doc_id,
                    point,
                    distance_km: distance,
                    relevance_score: score,
                });
            }
        }

        // Sort by distance (closest first), then by relevance score
        matches.sort_by(|a, b| {
            a.distance_km
                .partial_cmp(&b.distance_km)
                .unwrap()
                .then_with(|| b.relevance_score.partial_cmp(&a.relevance_score).unwrap())
        });

        Ok(matches)
    }

    /// Create a bounding box for efficient spatial filtering.
    fn create_bounding_box(&self) -> GeoBoundingBox {
        // Approximate degree distance at the center latitude
        let lat_deg_km = 111.0; // ~111 km per degree latitude
        let lon_deg_km = 111.0 * self.center.lat.to_radians().cos(); // Longitude varies by latitude

        let lat_delta = self.distance_km / lat_deg_km;
        let lon_delta = self.distance_km / lon_deg_km;

        let top_left = GeoPoint::new(
            (self.center.lat + lat_delta).min(90.0),
            (self.center.lon - lon_delta).max(-180.0),
        )
        .unwrap_or(self.center);

        let bottom_right = GeoPoint::new(
            (self.center.lat - lat_delta).max(-90.0),
            (self.center.lon + lon_delta).min(180.0),
        )
        .unwrap_or(self.center);

        let bbox = GeoBoundingBox::new(top_left, bottom_right).unwrap_or_else(|_| {
            // Fallback to a small box around center
            let fallback_top_left =
                GeoPoint::new(self.center.lat + 0.01, self.center.lon - 0.01).unwrap();
            let fallback_bottom_right =
                GeoPoint::new(self.center.lat - 0.01, self.center.lon + 0.01).unwrap();
            GeoBoundingBox::new(fallback_top_left, fallback_bottom_right).unwrap()
        });


        bbox
    }

    /// Get spatial candidates from the index within the bounding box.
    fn get_spatial_candidates(
        &self,
        reader: &dyn IndexReader,
        bounding_box: &GeoBoundingBox,
    ) -> Result<Vec<(u32, GeoPoint)>> {
        let mut candidates = Vec::new();

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
                        if bounding_box.contains(geo_point) {
                            let distance = self.center.distance_to(geo_point);
                            // Double-check with exact distance calculation
                            if distance <= self.distance_km {
                                candidates.push((doc_id as u32, geo_point.clone()));
                            }
                        }
                    }
                }
            }
        }

        Ok(candidates)
    }


    /// Calculate relevance score based on distance (closer = higher score).
    #[allow(dead_code)]
    fn calculate_distance_score(&self, distance_km: f64) -> f32 {
        if distance_km > self.distance_km {
            return 0.0;
        }

        // Linear decay: score = 1.0 at center, 0.0 at max distance
        let normalized_distance = distance_km / self.distance_km;
        (1.0 - normalized_distance) as f32
    }

    /// Calculate enhanced relevance score with multiple factors.
    #[allow(dead_code)]
    fn calculate_distance_score_enhanced(&self, distance_km: f64, point: &GeoPoint) -> f32 {
        if distance_km > self.distance_km {
            return 0.0;
        }

        // Base distance score (exponential decay for better distance weighting)
        let normalized_distance = distance_km / self.distance_km;
        let base_score = (-2.0 * normalized_distance).exp() as f32;

        // Precision bonus for exact location matches
        let precision_bonus = if distance_km < 0.1 { 0.1 } else { 0.0 };

        // Geographic relevance bonus (e.g., prefer points in certain regions)
        let geo_bonus = self.calculate_geographic_relevance(point);

        // Population density estimation (simulated)
        let density_bonus = self.estimate_population_density(point) * 0.05;

        (base_score + precision_bonus + geo_bonus + density_bonus).min(1.0)
    }

    /// Calculate geographic relevance based on point characteristics.
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    fn matcher(&self, reader: &dyn IndexReader) -> Result<Box<dyn Matcher>> {
        let matches = self.find_matches(reader)?;
        Ok(Box::new(GeoMatcher::new(matches)))
    }

    fn scorer(&self, reader: &dyn IndexReader) -> Result<Box<dyn Scorer>> {
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
            "GeoDistanceQuery(field: {}, center: {:?}, distance: {}km)",
            self.field, self.center, self.distance_km
        )
    }

    fn is_empty(&self, _reader: &dyn IndexReader) -> Result<bool> {
        Ok(self.distance_km <= 0.0)
    }

    fn cost(&self, reader: &dyn IndexReader) -> Result<u64> {
        // Geo queries can be expensive depending on the spatial index
        let doc_count = reader.doc_count() as u32;
        Ok(doc_count as u64 * 2) // Moderate cost
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A geographical bounding box query.
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
    pub fn find_matches(&self, reader: &dyn IndexReader) -> Result<Vec<GeoMatch>> {
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
                    distance_km: distance,
                    relevance_score,
                });
            }
        }

        // Sort by relevance score (highest first), then by distance to center
        matches.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap()
                .then_with(|| a.distance_km.partial_cmp(&b.distance_km).unwrap())
        });

        Ok(matches)
    }

    /// Get candidate points that might be within the bounding box.
    fn get_candidates_in_bounds(&self, reader: &dyn IndexReader) -> Result<Vec<(u32, GeoPoint)>> {
        let mut candidates = Vec::new();

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
                        if self.bounding_box.contains(geo_point) {
                            candidates.push((doc_id as u32, geo_point.clone()));
                        }
                    }
                }
            }
        }

        Ok(candidates)
    }

    /// Generate candidates within and around the bounding box.
    #[allow(dead_code)]
    fn generate_bounding_box_candidates(&self) -> Vec<(u32, GeoPoint)> {
        let mut candidates = Vec::new();
        let (width, height) = self.bounding_box.dimensions();

        // Generate grid points within the bounding box
        let grid_size = 20;
        for i in 0..grid_size {
            for j in 0..grid_size {
                let lat_ratio = i as f64 / (grid_size - 1) as f64;
                let lon_ratio = j as f64 / (grid_size - 1) as f64;

                let lat = self.bounding_box.bottom_right.lat + lat_ratio * height;
                let lon = self.bounding_box.top_left.lon + lon_ratio * width;

                if let Ok(point) = GeoPoint::new(lat, lon) {
                    let doc_id = (i * grid_size + j + 2000) as u32;
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
            if let Ok(point) = GeoPoint::new(center.lat + lat_offset, center.lon + lon_offset) {
                let doc_id = (i + 3000) as u32;
                candidates.push((doc_id, point));
            }
        }

        candidates
    }

    /// Calculate relevance score for points within the bounding box.
    #[allow(dead_code)]
    fn calculate_bounding_box_score(&self, point: &GeoPoint) -> f32 {
        let center = self.bounding_box.center();
        let (width, height) = self.bounding_box.dimensions();

        // Distance from center as a fraction of the bounding box diagonal
        let distance_to_center = center.distance_to(point);
        let diagonal_km = ((width * 111.0).powi(2) + (height * 111.0).powi(2)).sqrt();
        let normalized_distance = distance_to_center / diagonal_km;

        // Base score: higher for points closer to center
        let base_score = (1.0 - normalized_distance.min(1.0)) as f32;

        // Bonus for points near edges or corners (depending on use case)
        let edge_bonus = self.calculate_edge_proximity_bonus(point);

        // Corner bonus for strategic locations
        let corner_bonus = self.calculate_corner_bonus(point);

        (base_score + edge_bonus + corner_bonus).min(1.0)
    }

    /// Calculate bonus for points near edges of the bounding box.
    #[allow(dead_code)]
    fn calculate_edge_proximity_bonus(&self, point: &GeoPoint) -> f32 {
        let (width, height) = self.bounding_box.dimensions();
        let edge_threshold = 0.1; // 10% of dimension

        let lat_distance_from_edge = (point.lat - self.bounding_box.bottom_right.lat)
            .min(self.bounding_box.top_left.lat - point.lat);
        let lon_distance_from_edge = (point.lon - self.bounding_box.top_left.lon)
            .min(self.bounding_box.bottom_right.lon - point.lon);

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
    #[allow(dead_code)]
    fn calculate_corner_bonus(&self, point: &GeoPoint) -> f32 {
        let corners = [
            self.bounding_box.top_left,
            GeoPoint::new(
                self.bounding_box.top_left.lat,
                self.bounding_box.bottom_right.lon,
            )
            .unwrap(),
            self.bounding_box.bottom_right,
            GeoPoint::new(
                self.bounding_box.bottom_right.lat,
                self.bounding_box.top_left.lon,
            )
            .unwrap(),
        ];

        let corner_threshold_km = 1.0; // Within 1km of corner
        let mut min_corner_distance = f64::INFINITY;

        for corner in &corners {
            let distance = point.distance_to(corner);
            min_corner_distance = min_corner_distance.min(distance);
        }

        if min_corner_distance < corner_threshold_km {
            0.1 // Corner bonus
        } else {
            0.0
        }
    }
}

impl Query for GeoBoundingBoxQuery {
    fn matcher(&self, reader: &dyn IndexReader) -> Result<Box<dyn Matcher>> {
        let matches = self.find_matches(reader)?;
        Ok(Box::new(GeoMatcher::new(matches)))
    }

    fn scorer(&self, reader: &dyn IndexReader) -> Result<Box<dyn Scorer>> {
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

    fn is_empty(&self, _reader: &dyn IndexReader) -> Result<bool> {
        let (width, height) = self.bounding_box.dimensions();
        Ok(width <= 0.0 || height <= 0.0)
    }

    fn cost(&self, reader: &dyn IndexReader) -> Result<u64> {
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
    pub doc_id: u32,
    /// Geographical point of the document
    pub point: GeoPoint,
    /// Distance from query center in kilometers
    pub distance_km: f64,
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
    /// Current document ID
    current_doc_id: u64,
}

impl GeoMatcher {
    /// Create a new geo matcher.
    pub fn new(mut matches: Vec<GeoMatch>) -> Self {
        // Sort matches by distance (closest first)
        matches.sort_by(|a, b| a.distance_km.partial_cmp(&b.distance_km).unwrap_or(std::cmp::Ordering::Equal));
        
        let current_doc_id = if matches.is_empty() {
            u64::MAX  // Invalid state when no matches
        } else {
            matches[0].doc_id as u64  // Position at first match
        };
        
        GeoMatcher {
            matches,
            current_index: 0,
            current_doc_id,
        }
    }
}

impl Matcher for GeoMatcher {
    fn doc_id(&self) -> u64 {
        if self.current_index >= self.matches.len() {
            u64::MAX  // Invalid state when exhausted
        } else {
            self.current_doc_id
        }
    }

    fn next(&mut self) -> Result<bool> {
        if self.current_index >= self.matches.len() {
            return Ok(false);
        }
        
        self.current_index += 1;
        if self.current_index < self.matches.len() {
            self.current_doc_id = self.matches[self.current_index].doc_id as u64;
            Ok(true)
        } else {
            self.current_doc_id = u64::MAX; // Invalid state
            Ok(false)
        }
    }

    fn skip_to(&mut self, target: u64) -> Result<bool> {
        // Find first document ID >= target in order
        while self.current_index < self.matches.len() {
            let doc_id = self.matches[self.current_index].doc_id as u64;
            if doc_id >= target {
                self.current_doc_id = doc_id;
                return Ok(true);
            }
            self.current_index += 1;
        }
        self.current_doc_id = u64::MAX; // No match found
        Ok(false)
    }

    fn cost(&self) -> u64 {
        self.matches.len() as u64
    }

    fn is_exhausted(&self) -> bool {
        self.current_index >= self.matches.len()
    }
}

/// Scorer for geographical queries.
#[derive(Debug)]
pub struct GeoScorer {
    /// Document scores based on geographical relevance
    doc_scores: HashMap<u32, f32>,
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
    fn score(&self, doc_id: u64, _term_freq: f32) -> f32 {
        self.doc_scores.get(&(doc_id as u32)).unwrap_or(&0.0) * self.boost
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
}

/// Unified geo query API for convenient geographical searching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeoQuery {
    /// Distance-based query (within radius)
    Distance(GeoDistanceQuery),
    /// Bounding box query (within rectangular area)
    BoundingBox(GeoBoundingBoxQuery),
}

impl GeoQuery {
    /// Create a distance-based geo query (search within radius).
    ///
    /// # Arguments
    /// * `field` - The field containing geographical coordinates
    /// * `lat` - Center latitude in degrees (-90 to 90)
    /// * `lon` - Center longitude in degrees (-180 to 180)
    /// * `radius_km` - Search radius in kilometers
    ///
    /// # Example
    /// ```rust
    /// use sarissa::query::geo::GeoQuery;
    ///
    /// let query = GeoQuery::within_radius("location", 40.7128, -74.0060, 10.0).unwrap();
    /// ```
    pub fn within_radius<F: Into<String>>(
        field: F,
        lat: f64,
        lon: f64,
        radius_km: f64,
    ) -> Result<Self> {
        let center = GeoPoint::new(lat, lon)?;
        Ok(GeoQuery::Distance(GeoDistanceQuery::new(
            field, center, radius_km,
        )))
    }

    /// Create a bounding box geo query (search within rectangular area).
    ///
    /// # Arguments
    /// * `field` - The field containing geographical coordinates
    /// * `min_lat` - Minimum latitude (bottom edge)
    /// * `min_lon` - Minimum longitude (left edge)
    /// * `max_lat` - Maximum latitude (top edge)
    /// * `max_lon` - Maximum longitude (right edge)
    ///
    /// # Example
    /// ```rust
    /// use sarissa::query::geo::GeoQuery;
    ///
    /// let query = GeoQuery::within_bounding_box("location", 40.0, -75.0, 41.0, -74.0).unwrap();
    /// ```
    pub fn within_bounding_box<F: Into<String>>(
        field: F,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Result<Self> {
        let top_left = GeoPoint::new(max_lat, min_lon)?;
        let bottom_right = GeoPoint::new(min_lat, max_lon)?;
        let bbox = GeoBoundingBox::new(top_left, bottom_right)?;
        Ok(GeoQuery::BoundingBox(GeoBoundingBoxQuery::new(field, bbox)))
    }

    /// Create a geo query from a center point and radius.
    pub fn from_center_and_radius<F: Into<String>>(
        field: F,
        center: GeoPoint,
        radius_km: f64,
    ) -> Self {
        GeoQuery::Distance(GeoDistanceQuery::new(field, center, radius_km))
    }

    /// Create a geo query from a bounding box.
    pub fn from_bounding_box<F: Into<String>>(field: F, bbox: GeoBoundingBox) -> Self {
        GeoQuery::BoundingBox(GeoBoundingBoxQuery::new(field, bbox))
    }

    /// Set the boost factor for this query.
    pub fn with_boost(mut self, boost: f32) -> Self {
        match &mut self {
            GeoQuery::Distance(query) => {
                *query = query.clone().with_boost(boost);
            }
            GeoQuery::BoundingBox(query) => {
                *query = query.clone().with_boost(boost);
            }
        }
        self
    }

    /// Get the field name for this query.
    pub fn field(&self) -> &str {
        match self {
            GeoQuery::Distance(query) => query.field(),
            GeoQuery::BoundingBox(query) => query.field(),
        }
    }

    /// Get the boost factor for this query.
    pub fn boost(&self) -> f32 {
        match self {
            GeoQuery::Distance(query) => query.boost(),
            GeoQuery::BoundingBox(query) => query.boost(),
        }
    }
}

impl Query for GeoQuery {
    fn matcher(&self, reader: &dyn IndexReader) -> Result<Box<dyn Matcher>> {
        match self {
            GeoQuery::Distance(query) => query.matcher(reader),
            GeoQuery::BoundingBox(query) => query.matcher(reader),
        }
    }

    fn scorer(&self, reader: &dyn IndexReader) -> Result<Box<dyn Scorer>> {
        match self {
            GeoQuery::Distance(query) => query.scorer(reader),
            GeoQuery::BoundingBox(query) => query.scorer(reader),
        }
    }

    fn boost(&self) -> f32 {
        match self {
            GeoQuery::Distance(query) => query.boost(),
            GeoQuery::BoundingBox(query) => query.boost(),
        }
    }

    fn set_boost(&mut self, boost: f32) {
        match self {
            GeoQuery::Distance(query) => query.set_boost(boost),
            GeoQuery::BoundingBox(query) => query.set_boost(boost),
        }
    }

    fn clone_box(&self) -> Box<dyn Query> {
        Box::new(self.clone())
    }

    fn description(&self) -> String {
        match self {
            GeoQuery::Distance(query) => query.description(),
            GeoQuery::BoundingBox(query) => query.description(),
        }
    }

    fn is_empty(&self, reader: &dyn IndexReader) -> Result<bool> {
        match self {
            GeoQuery::Distance(query) => query.is_empty(reader),
            GeoQuery::BoundingBox(query) => query.is_empty(reader),
        }
    }

    fn cost(&self, reader: &dyn IndexReader) -> Result<u64> {
        match self {
            GeoQuery::Distance(query) => query.cost(reader),
            GeoQuery::BoundingBox(query) => query.cost(reader),
        }
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
        let point = GeoPoint::new(40.7128, -74.0060).unwrap(); // New York City
        assert_eq!(point.lat, 40.7128);
        assert_eq!(point.lon, -74.0060);

        // Test invalid coordinates
        assert!(GeoPoint::new(91.0, 0.0).is_err()); // Invalid latitude
        assert!(GeoPoint::new(0.0, 181.0).is_err()); // Invalid longitude
    }

    #[test]
    fn test_geo_distance_calculation() {
        let nyc = GeoPoint::new(40.7128, -74.0060).unwrap();
        let la = GeoPoint::new(34.0522, -118.2437).unwrap();

        let distance = nyc.distance_to(&la);
        // Distance between NYC and LA is approximately 3,944 km
        assert!((distance - 3944.0).abs() < 100.0); // Allow some tolerance
    }

    #[test]
    fn test_geo_bearing() {
        let nyc = GeoPoint::new(40.7128, -74.0060).unwrap();
        let la = GeoPoint::new(34.0522, -118.2437).unwrap();

        let bearing = nyc.bearing_to(&la);
        // Bearing from NYC to LA should be roughly west (around 270 degrees)
        assert!(bearing > 200.0 && bearing < 300.0);
    }

    #[test]
    fn test_geo_bounding_box() {
        let top_left = GeoPoint::new(41.0, -75.0).unwrap();
        let bottom_right = GeoPoint::new(40.0, -74.0).unwrap();
        let bbox = GeoBoundingBox::new(top_left, bottom_right).unwrap();

        let inside_point = GeoPoint::new(40.5, -74.5).unwrap();
        let outside_point = GeoPoint::new(42.0, -73.0).unwrap();

        assert!(bbox.contains(&inside_point));
        assert!(!bbox.contains(&outside_point));

        let center = bbox.center();
        assert_eq!(center.lat, 40.5);
        assert_eq!(center.lon, -74.5);
    }

    #[test]
    fn test_geo_distance_query() {
        let center = GeoPoint::new(40.7128, -74.0060).unwrap();
        let query = GeoDistanceQuery::new("location", center, 10.0).with_boost(1.5);

        assert_eq!(query.field(), "location");
        assert_eq!(query.center(), center);
        assert_eq!(query.distance_km(), 10.0);
        assert_eq!(query.boost(), 1.5);
    }

    #[test]
    fn test_geo_distance_scoring() {
        let center = GeoPoint::new(0.0, 0.0).unwrap();
        let query = GeoDistanceQuery::new("location", center, 10.0);

        // Test scoring at different distances
        assert_eq!(query.calculate_distance_score(0.0), 1.0); // At center
        assert_eq!(query.calculate_distance_score(5.0), 0.5); // Half distance
        assert_eq!(query.calculate_distance_score(10.0), 0.0); // At max distance
        assert_eq!(query.calculate_distance_score(15.0), 0.0); // Beyond max distance
    }

    #[test]
    fn test_geo_bounding_box_query() {
        let top_left = GeoPoint::new(41.0, -75.0).unwrap();
        let bottom_right = GeoPoint::new(40.0, -74.0).unwrap();
        let bbox = GeoBoundingBox::new(top_left, bottom_right).unwrap();
        let query = GeoBoundingBoxQuery::new("location", bbox);

        assert_eq!(query.field(), "location");
        assert_eq!(query.bounding_box().top_left, top_left);
        assert_eq!(query.bounding_box().bottom_right, bottom_right);
    }

    #[test]
    fn test_geo_matcher() {
        let matches = vec![
            GeoMatch {
                doc_id: 3,
                point: GeoPoint::new(0.0, 0.0).unwrap(),
                distance_km: 1.0,
                relevance_score: 0.9,
            },
            GeoMatch {
                doc_id: 1,
                point: GeoPoint::new(0.0, 0.0).unwrap(),
                distance_km: 2.0,
                relevance_score: 0.8,
            },
        ];

        let mut matcher = GeoMatcher::new(matches);

        // Should return documents in distance-sorted order (closest first)
        // After sorting: doc_id: 3 (1.0km) comes before doc_id: 1 (2.0km)
        // Initial state: pointing to first document (doc_id: 3)
        assert_eq!(matcher.doc_id(), 3);  // Initial position: closest document
        
        assert!(matcher.next().unwrap());  // Move to next
        assert_eq!(matcher.doc_id(), 1);   // Second document (farther)
        
        assert!(!matcher.next().unwrap()); // No more documents
    }

    #[test]
    fn test_geo_scorer() {
        let matches = vec![GeoMatch {
            doc_id: 1,
            point: GeoPoint::new(0.0, 0.0).unwrap(),
            distance_km: 1.0,
            relevance_score: 0.9,
        }];

        let scorer = GeoScorer::new(matches, 2.0);

        assert_eq!(scorer.score(1, 1.0), 0.9 * 2.0);
        assert_eq!(scorer.score(999, 1.0), 0.0); // Non-existent document
        assert_eq!(scorer.max_score(), 0.9 * 2.0);
        assert_eq!(scorer.name(), "GeoScorer");
    }

    #[test]
    fn test_enhanced_distance_scoring() {
        let center = GeoPoint::new(40.7128, -74.0060).unwrap(); // NYC
        let query = GeoDistanceQuery::new("location", center, 10.0);

        // Test point very close to center
        let close_point = GeoPoint::new(40.7130, -74.0062).unwrap();
        let close_score = query.calculate_distance_score_enhanced(0.05, &close_point);

        // Test point at moderate distance
        let mid_point = GeoPoint::new(40.7200, -74.0100).unwrap();
        let mid_score = query.calculate_distance_score_enhanced(1.0, &mid_point);

        // Test point at max distance
        let far_point = GeoPoint::new(40.8000, -74.1000).unwrap();
        let far_score = query.calculate_distance_score_enhanced(9.0, &far_point);

        // Scores should decrease with distance, and close points should get precision bonus
        assert!(close_score > mid_score);
        assert!(mid_score > far_score);
        assert!(close_score > 0.9); // Should get precision bonus
    }

    #[test]
    fn test_bounding_box_enhanced_functionality() {
        let top_left = GeoPoint::new(41.0, -75.0).unwrap();
        let bottom_right = GeoPoint::new(40.0, -74.0).unwrap();
        let bbox = GeoBoundingBox::new(top_left, bottom_right).unwrap();
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

        let corner_point = query.bounding_box().top_left;
        let corner_score = query.calculate_bounding_box_score(&corner_point);

        // Center should generally score higher than corners
        assert!(center_score >= corner_score);
    }

    #[test]
    fn test_spatial_bounding_box_creation() {
        let center = GeoPoint::new(40.7128, -74.0060).unwrap(); // NYC
        let query = GeoDistanceQuery::new("location", center, 5.0); // 5km radius

        let bbox = query.create_bounding_box();

        // Check that the bounding box contains the center
        assert!(bbox.contains(&center));

        // Check that the bounding box is roughly the right size
        let (width, height) = bbox.dimensions();
        assert!(width > 0.0 && width < 1.0); // Should be less than 1 degree
        assert!(height > 0.0 && height < 1.0);

        // The center should be approximately in the middle of the bounding box
        let bbox_center = bbox.center();
        let center_distance = center.distance_to(&bbox_center);
        assert!(center_distance < 1.0); // Should be very close
    }

    #[test]
    fn test_geographic_relevance_calculation() {
        let center = GeoPoint::new(40.7128, -74.0060).unwrap();
        let query = GeoDistanceQuery::new("location", center, 10.0);

        // Test temperate zone bonus
        let temperate_point = GeoPoint::new(45.0, 0.0).unwrap(); // Temperate zone
        let tropical_point = GeoPoint::new(10.0, 0.0).unwrap(); // Tropical zone

        let temperate_bonus = query.calculate_geographic_relevance(&temperate_point);
        let tropical_bonus = query.calculate_geographic_relevance(&tropical_point);

        assert!(temperate_bonus > tropical_bonus);

        // Test equator bonus
        let equator_point = GeoPoint::new(2.0, 0.0).unwrap(); // Near equator
        let non_equator_point = GeoPoint::new(45.0, 0.0).unwrap();

        let equator_geo_bonus = query.calculate_geographic_relevance(&equator_point);
        let non_equator_geo_bonus = query.calculate_geographic_relevance(&non_equator_point);

        // Both should have some bonus, but for different reasons
        assert!(equator_geo_bonus > 0.0);
        assert!(non_equator_geo_bonus > 0.0);
    }
}
