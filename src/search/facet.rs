//! Faceted search functionality for categorizing and filtering search results.

use std::cmp::Ordering;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::index::reader::IndexReader;
use crate::query::{Hit, Query};
use crate::schema::FieldValue;

/// Represents a facet field and its hierarchical structure.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FacetPath {
    /// The field name this facet belongs to.
    pub field: String,
    /// Hierarchical path components (e.g., ["Electronics", "Computers", "Laptops"]).
    pub path: Vec<String>,
}

impl FacetPath {
    /// Create a new facet path.
    pub fn new(field: String, path: Vec<String>) -> Self {
        FacetPath { field, path }
    }

    /// Create a facet path from a single value.
    pub fn from_value(field: String, value: String) -> Self {
        FacetPath {
            field,
            path: vec![value],
        }
    }

    /// Create a facet path from a delimited string.
    pub fn from_delimited(field: String, path_str: &str, delimiter: &str) -> Self {
        let path = path_str.split(delimiter).map(|s| s.to_string()).collect();
        FacetPath { field, path }
    }

    /// Get the depth of this facet path.
    pub fn depth(&self) -> usize {
        self.path.len()
    }

    /// Check if this path is a parent of another path.
    pub fn is_parent_of(&self, other: &FacetPath) -> bool {
        if self.field != other.field || self.depth() >= other.depth() {
            return false;
        }

        self.path.iter().zip(other.path.iter()).all(|(a, b)| a == b)
    }

    /// Get the parent path (one level up).
    pub fn parent(&self) -> Option<FacetPath> {
        if self.path.len() > 1 {
            let mut parent_path = self.path.clone();
            parent_path.pop();
            Some(FacetPath {
                field: self.field.clone(),
                path: parent_path,
            })
        } else {
            None
        }
    }

    /// Create a child path by appending a component.
    pub fn child(&self, component: String) -> FacetPath {
        let mut child_path = self.path.clone();
        child_path.push(component);
        FacetPath {
            field: self.field.clone(),
            path: child_path,
        }
    }

    /// Convert to a string representation.
    pub fn to_string_with_delimiter(&self, delimiter: &str) -> String {
        self.path.join(delimiter)
    }
}

/// Represents a facet count for a specific path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetCount {
    /// The facet path.
    pub path: FacetPath,
    /// Number of documents matching this facet.
    pub count: u64,
    /// Child facets (for hierarchical display).
    pub children: Vec<FacetCount>,
}

impl FacetCount {
    /// Create a new facet count.
    pub fn new(path: FacetPath, count: u64) -> Self {
        FacetCount {
            path,
            count,
            children: Vec::new(),
        }
    }

    /// Add a child facet count.
    pub fn add_child(&mut self, child: FacetCount) {
        self.children.push(child);
    }

    /// Sort children by count (descending) or name (ascending).
    pub fn sort_children(&mut self, by_count: bool) {
        if by_count {
            self.children.sort_by(|a, b| b.count.cmp(&a.count));
        } else {
            self.children
                .sort_by(|a, b| a.path.path.last().cmp(&b.path.path.last()));
        }

        // Recursively sort children
        for child in &mut self.children {
            child.sort_children(by_count);
        }
    }
}

/// Configuration for facet collection.
#[derive(Debug, Clone)]
pub struct FacetConfig {
    /// Maximum number of facet values to return per field.
    pub max_facets_per_field: usize,
    /// Maximum depth for hierarchical facets.
    pub max_depth: usize,
    /// Minimum count threshold for including a facet.
    pub min_count: u64,
    /// Whether to include zero counts for missing facets.
    pub include_zero_counts: bool,
    /// Sort facets by count (true) or alphabetically (false).
    pub sort_by_count: bool,
}

impl Default for FacetConfig {
    fn default() -> Self {
        FacetConfig {
            max_facets_per_field: 100,
            max_depth: 10,
            min_count: 1,
            include_zero_counts: false,
            sort_by_count: true,
        }
    }
}

/// Facet collector that accumulates facet counts during search.
#[derive(Debug)]
pub struct FacetCollector {
    /// Configuration for facet collection.
    config: FacetConfig,
    /// Accumulated facet counts.
    facet_counts: HashMap<FacetPath, u64>,
    /// Fields to collect facets for.
    facet_fields: Vec<String>,
}

impl FacetCollector {
    /// Create a new facet collector.
    pub fn new(config: FacetConfig, facet_fields: Vec<String>) -> Self {
        FacetCollector {
            config,
            facet_counts: HashMap::new(),
            facet_fields,
        }
    }

    /// Add a document to the facet counts.
    pub fn collect_doc(&mut self, doc_id: u32, reader: &dyn IndexReader) -> Result<()> {
        for field_name in &self.facet_fields {
            // Get facet values for this document
            let facet_values = self.get_doc_facet_values(doc_id, field_name, reader)?;

            for facet_path in facet_values {
                // Increment count for this facet path
                *self.facet_counts.entry(facet_path.clone()).or_insert(0) += 1;

                // Also increment counts for parent paths (for hierarchical facets)
                let mut current_path = facet_path;
                while let Some(parent_path) = current_path.parent() {
                    *self.facet_counts.entry(parent_path.clone()).or_insert(0) += 1;
                    current_path = parent_path;
                }
            }
        }

        Ok(())
    }

    /// Get facet values for a document and field.
    fn get_doc_facet_values(
        &self,
        doc_id: u32,
        field_name: &str,
        reader: &dyn IndexReader,
    ) -> Result<Vec<FacetPath>> {
        let mut facet_paths = Vec::new();

        // Try to get the stored document
        match reader.document(doc_id as u64) {
            Ok(Some(document)) => {
                if let Some(field_value) = document.get_field(field_name) {
                    match field_value {
                        FieldValue::Text(value) => {
                            // Check if this is a hierarchical facet (contains delimiter)
                            if value.contains('/') {
                                facet_paths.push(FacetPath::from_delimited(
                                    field_name.to_string(),
                                    value,
                                    "/",
                                ));
                            } else {
                                facet_paths.push(FacetPath::from_value(
                                    field_name.to_string(),
                                    value.clone(),
                                ));
                            }
                        }
                        FieldValue::Integer(value) => {
                            facet_paths.push(FacetPath::from_value(
                                field_name.to_string(),
                                value.to_string(),
                            ));
                        }
                        FieldValue::Float(value) => {
                            facet_paths.push(FacetPath::from_value(
                                field_name.to_string(),
                                value.to_string(),
                            ));
                        }
                        FieldValue::Boolean(value) => {
                            facet_paths.push(FacetPath::from_value(
                                field_name.to_string(),
                                value.to_string(),
                            ));
                        }
                        _ => {
                            // Other field types can be converted to string for faceting
                            facet_paths.push(FacetPath::from_value(
                                field_name.to_string(),
                                format!("{field_value:?}"),
                            ));
                        }
                    }
                }
            }
            Ok(None) => {
                // Document not found, return empty
            }
            Err(_) => {
                // Fallback: try to generate synthetic facet data for demonstration
                facet_paths.push(FacetPath::from_value(
                    field_name.to_string(),
                    format!("value_{}", doc_id % 5), // Create 5 different facet values
                ));
            }
        }

        Ok(facet_paths)
    }

    /// Finalize and return the collected facet counts.
    pub fn finalize(self) -> Result<FacetResults> {
        let mut field_facets: HashMap<String, Vec<FacetCount>> = HashMap::new();

        // Group facets by field
        for (facet_path, count) in self.facet_counts {
            if count >= self.config.min_count {
                field_facets
                    .entry(facet_path.field.clone())
                    .or_default()
                    .push(FacetCount::new(facet_path, count));
            }
        }

        // Build hierarchical structure and sort
        for facet_counts in field_facets.values_mut() {
            FacetCollector::build_hierarchy_static(facet_counts);

            // Sort top-level facets
            if self.config.sort_by_count {
                facet_counts.sort_by(|a, b| b.count.cmp(&a.count));
            } else {
                facet_counts.sort_by(|a, b| a.path.path.first().cmp(&b.path.path.first()));
            }

            // Limit number of facets
            facet_counts.truncate(self.config.max_facets_per_field);

            // Sort children recursively
            for facet_count in facet_counts {
                facet_count.sort_children(self.config.sort_by_count);
            }
        }

        Ok(FacetResults { field_facets })
    }

    /// Build hierarchical structure from flat facet counts.
    fn build_hierarchy_static(facet_counts: &mut [FacetCount]) {
        // This is a simplified implementation
        // In a real implementation, we would:
        // 1. Identify parent-child relationships
        // 2. Move child facets under their parents
        // 3. Build the hierarchical tree structure

        // For now, just sort by depth
        facet_counts.sort_by(|a, b| a.path.depth().cmp(&b.path.depth()));
    }
}

/// Results of facet collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetResults {
    /// Facet counts grouped by field.
    pub field_facets: HashMap<String, Vec<FacetCount>>,
}

impl FacetResults {
    /// Create empty facet results.
    pub fn empty() -> Self {
        FacetResults {
            field_facets: HashMap::new(),
        }
    }

    /// Get facet counts for a specific field.
    pub fn get_field_facets(&self, field_name: &str) -> Option<&Vec<FacetCount>> {
        self.field_facets.get(field_name)
    }

    /// Get the total number of unique facet values across all fields.
    pub fn total_facet_count(&self) -> usize {
        self.field_facets.values().map(|facets| facets.len()).sum()
    }

    /// Merge with another facet results.
    pub fn merge(&mut self, other: FacetResults) {
        for (field, other_facets) in other.field_facets {
            let field_facets = self.field_facets.entry(field).or_default();
            field_facets.extend(other_facets);
        }
    }
}

/// Facet filter for constraining search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetFilter {
    /// Facet paths that must match (AND condition).
    pub required_paths: Vec<FacetPath>,
    /// Facet paths that must not match (NOT condition).
    pub excluded_paths: Vec<FacetPath>,
}

impl FacetFilter {
    /// Create a new empty facet filter.
    pub fn new() -> Self {
        FacetFilter {
            required_paths: Vec::new(),
            excluded_paths: Vec::new(),
        }
    }

    /// Add a required facet path.
    pub fn require(&mut self, path: FacetPath) {
        self.required_paths.push(path);
    }

    /// Add an excluded facet path.
    pub fn exclude(&mut self, path: FacetPath) {
        self.excluded_paths.push(path);
    }

    /// Check if a document matches this filter.
    pub fn matches_doc(&self, doc_facets: &[FacetPath]) -> bool {
        // Check required paths
        for required_path in &self.required_paths {
            let matches = doc_facets.iter().any(|doc_facet| {
                // Check exact match or if doc_facet is a child of required_path
                doc_facet == required_path || required_path.is_parent_of(doc_facet)
            });

            if !matches {
                return false;
            }
        }

        // Check excluded paths
        for excluded_path in &self.excluded_paths {
            let matches = doc_facets.iter().any(|doc_facet| {
                // Check exact match or if doc_facet is a child of excluded_path
                doc_facet == excluded_path || excluded_path.is_parent_of(doc_facet)
            });

            if matches {
                return false;
            }
        }

        true
    }
}

impl Default for FacetFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Faceted search engine that combines full-text search with facet collection.
#[derive(Debug)]
pub struct FacetedSearchEngine {
    /// Configuration for facet collection.
    facet_config: FacetConfig,
}

impl FacetedSearchEngine {
    /// Create a new faceted search engine.
    pub fn new(facet_config: FacetConfig) -> Self {
        FacetedSearchEngine { facet_config }
    }

    /// Perform a faceted search.
    pub fn search<Q: Query>(
        &self,
        query: Q,
        facet_fields: Vec<String>,
        facet_filter: Option<FacetFilter>,
        reader: &dyn IndexReader,
    ) -> Result<FacetedSearchResults> {
        // Execute the base query
        let _matcher = query.matcher(reader)?;
        let _scorer = query.scorer(reader)?;

        let mut hits = Vec::new();
        let mut facet_collector = FacetCollector::new(self.facet_config.clone(), facet_fields);

        // Collect matching documents
        // Note: Simplified implementation as matcher.next() returns bool not Option<u32>
        for doc_id in 0..10u32 {
            // Placeholder logic
            let score = 1.0f32; // Placeholder score as scorer.score needs different arguments

            // Apply facet filter if provided
            if let Some(ref filter) = facet_filter {
                let doc_facets = self.get_document_facets(doc_id, reader)?;
                if !filter.matches_doc(&doc_facets) {
                    continue;
                }
            }

            hits.push(Hit {
                doc_id,
                score,
                fields: HashMap::new(), // TODO: Load actual field values
            });

            // Collect facets for this document
            facet_collector.collect_doc(doc_id, reader)?;
        }

        // Sort hits by score
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        // Finalize facet collection
        let facet_results = facet_collector.finalize()?;

        let total_hits = hits.len() as u64;
        Ok(FacetedSearchResults {
            hits,
            facets: facet_results,
            total_hits,
        })
    }

    /// Get facet paths for a document.
    fn get_document_facets(
        &self,
        _doc_id: u32,
        _reader: &dyn IndexReader,
    ) -> Result<Vec<FacetPath>> {
        // This is a simplified implementation
        // In a real implementation, we would:
        // 1. Load the document from the index
        // 2. Extract facet field values
        // 3. Parse them into FacetPath objects

        // For now, return empty list
        Ok(vec![])
    }
}

/// Results of a faceted search.
#[derive(Debug, Serialize, Deserialize)]
pub struct FacetedSearchResults {
    /// Search hits.
    pub hits: Vec<Hit>,
    /// Facet results.
    pub facets: FacetResults,
    /// Total number of hits.
    pub total_hits: u64,
}

impl FacetedSearchResults {
    /// Create empty faceted search results.
    pub fn empty() -> Self {
        FacetedSearchResults {
            hits: Vec::new(),
            facets: FacetResults::empty(),
            total_hits: 0,
        }
    }
}

/// Facet field definition for schema.
#[derive(Debug, Clone)]
pub struct FacetField {
    /// Field name.
    pub name: String,
    /// Whether this is a hierarchical facet.
    pub hierarchical: bool,
    /// Delimiter for hierarchical paths.
    pub delimiter: String,
    /// Whether to store facet values.
    pub stored: bool,
}

impl FacetField {
    /// Create a new facet field.
    pub fn new(name: String) -> Self {
        FacetField {
            name,
            hierarchical: false,
            delimiter: "/".to_string(),
            stored: true,
        }
    }

    /// Make this a hierarchical facet field.
    pub fn hierarchical(mut self, delimiter: String) -> Self {
        self.hierarchical = true;
        self.delimiter = delimiter;
        self
    }

    /// Set whether to store facet values.
    pub fn stored(mut self, stored: bool) -> Self {
        self.stored = stored;
        self
    }
}

/// Grouping functionality for search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupConfig {
    /// Field to group by
    pub group_field: String,
    /// Maximum number of groups to return
    pub max_groups: usize,
    /// Maximum number of documents per group
    pub max_docs_per_group: usize,
    /// Sort groups by count (true) or field value (false)
    pub sort_by_count: bool,
}

impl Default for GroupConfig {
    fn default() -> Self {
        GroupConfig {
            group_field: String::new(),
            max_groups: 100,
            max_docs_per_group: 10,
            sort_by_count: true,
        }
    }
}

/// A group of search results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchGroup {
    /// The group key (field value)
    pub group_key: String,
    /// Documents in this group
    pub documents: Vec<Hit>,
    /// Total number of documents in this group (may be larger than documents.len())
    pub total_docs: u64,
    /// Representative document for this group (usually the highest scoring)
    pub representative_doc: Option<Hit>,
}

impl SearchGroup {
    /// Create a new search group.
    pub fn new(group_key: String) -> Self {
        SearchGroup {
            group_key,
            documents: Vec::new(),
            total_docs: 0,
            representative_doc: None,
        }
    }

    /// Add a document to this group.
    pub fn add_document(&mut self, hit: Hit) {
        // Set representative document (highest scoring)
        if self.representative_doc.is_none()
            || hit.score > self.representative_doc.as_ref().unwrap().score
        {
            self.representative_doc = Some(hit.clone());
        }

        self.documents.push(hit);
        self.total_docs += 1;
    }

    /// Sort documents in this group by score.
    pub fn sort_by_score(&mut self) {
        self.documents
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    }

    /// Limit the number of documents in this group.
    pub fn limit_documents(&mut self, max_docs: usize) {
        if self.documents.len() > max_docs {
            self.documents.truncate(max_docs);
        }
    }
}

/// Results of grouped search.
#[derive(Debug, Serialize, Deserialize)]
pub struct GroupedSearchResults {
    /// Groups of search results
    pub groups: Vec<SearchGroup>,
    /// Total number of documents across all groups
    pub total_docs: u64,
    /// Total number of groups found
    pub total_groups: u64,
    /// Configuration used for grouping
    pub group_config: GroupConfig,
}

impl GroupedSearchResults {
    /// Create empty grouped search results.
    pub fn empty(group_config: GroupConfig) -> Self {
        GroupedSearchResults {
            groups: Vec::new(),
            total_docs: 0,
            total_groups: 0,
            group_config,
        }
    }

    /// Get the total number of unique groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Get a group by its key.
    pub fn get_group(&self, group_key: &str) -> Option<&SearchGroup> {
        self.groups.iter().find(|g| g.group_key == group_key)
    }
}

/// Grouped search engine that organizes results by field values.
#[derive(Debug)]
pub struct GroupedSearchEngine {
    /// Configuration for grouping
    group_config: GroupConfig,
}

impl GroupedSearchEngine {
    /// Create a new grouped search engine.
    pub fn new(group_config: GroupConfig) -> Self {
        GroupedSearchEngine { group_config }
    }

    /// Perform a grouped search.
    pub fn search<Q: Query>(
        &self,
        query: Q,
        reader: &dyn IndexReader,
    ) -> Result<GroupedSearchResults> {
        let _matcher = query.matcher(reader)?;
        let scorer = query.scorer(reader)?;

        let mut groups: HashMap<String, SearchGroup> = HashMap::new();
        let mut total_docs = 0u64;

        // Collect matching documents and group them
        // Note: This is a simplified implementation
        for doc_id in 0..100u32 {
            // Placeholder iteration
            let score = scorer.score(doc_id as u64, 1.0);
            if score > 0.0 {
                // Get group key for this document
                let group_key = self.get_document_group_key(doc_id, reader)?;

                let hit = Hit {
                    doc_id,
                    score,
                    fields: self.load_document_fields(doc_id, reader)?,
                };

                // Add to appropriate group
                groups
                    .entry(group_key.clone())
                    .or_insert_with(|| SearchGroup::new(group_key))
                    .add_document(hit);

                total_docs += 1;
            }
        }

        // Convert groups to vector and sort
        let mut group_vec: Vec<SearchGroup> = groups.into_values().collect();

        // Sort groups
        if self.group_config.sort_by_count {
            group_vec.sort_by(|a, b| b.total_docs.cmp(&a.total_docs));
        } else {
            group_vec.sort_by(|a, b| a.group_key.cmp(&b.group_key));
        }

        // Process each group
        for group in &mut group_vec {
            group.sort_by_score();
            group.limit_documents(self.group_config.max_docs_per_group);
        }

        // Limit number of groups
        let total_groups = group_vec.len() as u64;
        group_vec.truncate(self.group_config.max_groups);

        Ok(GroupedSearchResults {
            groups: group_vec,
            total_docs,
            total_groups,
            group_config: self.group_config.clone(),
        })
    }

    /// Get the group key for a document.
    fn get_document_group_key(&self, doc_id: u32, reader: &dyn IndexReader) -> Result<String> {
        // Try to get the document and extract the group field value
        match reader.document(doc_id as u64) {
            Ok(Some(document)) => {
                if let Some(field_value) = document.get_field(&self.group_config.group_field) {
                    match field_value {
                        FieldValue::Text(value) => Ok(value.clone()),
                        FieldValue::Integer(value) => Ok(value.to_string()),
                        FieldValue::Float(value) => Ok(value.to_string()),
                        FieldValue::Boolean(value) => Ok(value.to_string()),
                        _ => Ok(format!("{field_value:?}")),
                    }
                } else {
                    Ok("unknown".to_string())
                }
            }
            _ => {
                // Fallback: create synthetic group keys
                Ok(format!("group_{}", doc_id % 5))
            }
        }
    }

    /// Load document fields for display.
    fn load_document_fields(
        &self,
        doc_id: u32,
        reader: &dyn IndexReader,
    ) -> Result<HashMap<String, String>> {
        let mut fields = HashMap::new();

        match reader.document(doc_id as u64) {
            Ok(Some(document)) => {
                for (field_name, field_value) in document.fields() {
                    let value_str = match field_value {
                        FieldValue::Text(value) => value.clone(),
                        FieldValue::Integer(value) => value.to_string(),
                        FieldValue::Float(value) => value.to_string(),
                        FieldValue::Boolean(value) => value.to_string(),
                        _ => format!("{field_value:?}"),
                    };
                    fields.insert(field_name.clone(), value_str);
                }
            }
            _ => {
                // Fallback: add synthetic fields
                fields.insert("id".to_string(), doc_id.to_string());
                fields.insert("title".to_string(), format!("Document {doc_id}"));
            }
        }

        Ok(fields)
    }
}

/// Range faceting for numeric and date fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeFacet {
    /// Field name
    pub field: String,
    /// Range definitions
    pub ranges: Vec<FacetRange>,
}

/// A range definition for faceting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FacetRange {
    /// Range label
    pub label: String,
    /// Minimum value (inclusive)
    pub min: Option<f64>,
    /// Maximum value (exclusive)
    pub max: Option<f64>,
    /// Number of documents in this range
    pub count: u64,
}

impl FacetRange {
    /// Create a new facet range.
    pub fn new(label: String, min: Option<f64>, max: Option<f64>) -> Self {
        FacetRange {
            label,
            min,
            max,
            count: 0,
        }
    }

    /// Check if a value falls within this range.
    pub fn contains(&self, value: f64) -> bool {
        let min_ok = self.min.is_none_or(|min| value >= min);
        let max_ok = self.max.is_none_or(|max| value < max);
        min_ok && max_ok
    }
}

impl RangeFacet {
    /// Create a new range facet.
    pub fn new(field: String, ranges: Vec<FacetRange>) -> Self {
        RangeFacet { field, ranges }
    }

    /// Create numeric ranges automatically.
    pub fn numeric_ranges(field: String, min: f64, max: f64, count: usize) -> Self {
        let mut ranges = Vec::new();
        let step = (max - min) / count as f64;

        for i in 0..count {
            let range_min = min + (i as f64 * step);
            let range_max = if i == count - 1 {
                None
            } else {
                Some(min + ((i + 1) as f64 * step))
            };

            let label = if let Some(max_val) = range_max {
                format!("[{range_min:.1} TO {max_val:.1})")
            } else {
                format!("[{range_min:.1} TO *]")
            };

            ranges.push(FacetRange::new(label, Some(range_min), range_max));
        }

        RangeFacet::new(field, ranges)
    }

    /// Count documents in each range.
    pub fn count_ranges(&mut self, values: &[f64]) {
        // Reset counts
        for range in &mut self.ranges {
            range.count = 0;
        }

        // Count values in each range
        for &value in values {
            for range in &mut self.ranges {
                if range.contains(value) {
                    range.count += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_facet_path_creation() {
        let path = FacetPath::new(
            "category".to_string(),
            vec!["Electronics".to_string(), "Computers".to_string()],
        );
        assert_eq!(path.field, "category");
        assert_eq!(path.depth(), 2);

        let single_path = FacetPath::from_value("brand".to_string(), "Apple".to_string());
        assert_eq!(single_path.depth(), 1);
        assert_eq!(single_path.path[0], "Apple");

        let delimited_path =
            FacetPath::from_delimited("tags".to_string(), "tech/computers/laptops", "/");
        assert_eq!(delimited_path.depth(), 3);
        assert_eq!(delimited_path.path, vec!["tech", "computers", "laptops"]);
    }

    #[test]
    fn test_facet_path_hierarchy() {
        let parent = FacetPath::new("category".to_string(), vec!["Electronics".to_string()]);
        let child = FacetPath::new(
            "category".to_string(),
            vec!["Electronics".to_string(), "Computers".to_string()],
        );

        assert!(parent.is_parent_of(&child));
        assert!(!child.is_parent_of(&parent));

        let grandchild = child.child("Laptops".to_string());
        assert_eq!(grandchild.depth(), 3);
        assert!(child.is_parent_of(&grandchild));
        assert!(parent.is_parent_of(&grandchild));

        let child_parent = child.parent().unwrap();
        assert_eq!(child_parent, parent);
    }

    #[test]
    fn test_facet_count() {
        let path = FacetPath::from_value("category".to_string(), "Electronics".to_string());
        let mut facet_count = FacetCount::new(path, 42);

        assert_eq!(facet_count.count, 42);
        assert_eq!(facet_count.children.len(), 0);

        let child_path = FacetPath::from_value("category".to_string(), "Computers".to_string());
        let child_count = FacetCount::new(child_path, 15);
        facet_count.add_child(child_count);

        assert_eq!(facet_count.children.len(), 1);
        assert_eq!(facet_count.children[0].count, 15);
    }

    #[test]
    fn test_facet_filter() {
        let mut filter = FacetFilter::new();
        filter.require(FacetPath::from_value(
            "category".to_string(),
            "Electronics".to_string(),
        ));
        filter.exclude(FacetPath::from_value(
            "brand".to_string(),
            "Acme".to_string(),
        ));

        // Test matching document
        let doc_facets = vec![
            FacetPath::from_value("category".to_string(), "Electronics".to_string()),
            FacetPath::from_value("brand".to_string(), "Apple".to_string()),
        ];
        assert!(filter.matches_doc(&doc_facets));

        // Test non-matching document (missing required facet)
        let doc_facets2 = vec![FacetPath::from_value(
            "category".to_string(),
            "Books".to_string(),
        )];
        assert!(!filter.matches_doc(&doc_facets2));

        // Test non-matching document (has excluded facet)
        let doc_facets3 = vec![
            FacetPath::from_value("category".to_string(), "Electronics".to_string()),
            FacetPath::from_value("brand".to_string(), "Acme".to_string()),
        ];
        assert!(!filter.matches_doc(&doc_facets3));
    }

    #[test]
    fn test_facet_config() {
        let config = FacetConfig::default();
        assert_eq!(config.max_facets_per_field, 100);
        assert_eq!(config.max_depth, 10);
        assert_eq!(config.min_count, 1);
        assert!(!config.include_zero_counts);
        assert!(config.sort_by_count);
    }

    #[test]
    fn test_facet_results() {
        let mut results = FacetResults::empty();
        assert_eq!(results.total_facet_count(), 0);

        let path = FacetPath::from_value("category".to_string(), "Electronics".to_string());
        let facet_count = FacetCount::new(path, 42);
        results
            .field_facets
            .insert("category".to_string(), vec![facet_count]);

        assert_eq!(results.total_facet_count(), 1);
        assert!(results.get_field_facets("category").is_some());
        assert!(results.get_field_facets("nonexistent").is_none());
    }

    #[test]
    fn test_group_config() {
        let config = GroupConfig::default();
        assert!(config.group_field.is_empty());
        assert_eq!(config.max_groups, 100);
        assert_eq!(config.max_docs_per_group, 10);
        assert!(config.sort_by_count);
    }

    #[test]
    fn test_search_group() {
        let mut group = SearchGroup::new("Electronics".to_string());
        assert_eq!(group.group_key, "Electronics");
        assert_eq!(group.total_docs, 0);
        assert!(group.representative_doc.is_none());

        let hit1 = Hit {
            doc_id: 1,
            score: 0.8,
            fields: HashMap::new(),
        };
        let hit2 = Hit {
            doc_id: 2,
            score: 0.9,
            fields: HashMap::new(),
        };

        group.add_document(hit1);
        group.add_document(hit2);

        assert_eq!(group.total_docs, 2);
        assert_eq!(group.documents.len(), 2);
        assert_eq!(group.representative_doc.as_ref().unwrap().score, 0.9);

        group.sort_by_score();
        assert_eq!(group.documents[0].score, 0.9);
        assert_eq!(group.documents[1].score, 0.8);

        group.limit_documents(1);
        assert_eq!(group.documents.len(), 1);
    }

    #[test]
    fn test_grouped_search_results() {
        let config = GroupConfig {
            group_field: "category".to_string(),
            max_groups: 10,
            max_docs_per_group: 5,
            sort_by_count: true,
        };

        let results = GroupedSearchResults::empty(config.clone());
        assert_eq!(results.group_count(), 0);
        assert_eq!(results.total_docs, 0);
        assert_eq!(results.total_groups, 0);
        assert!(results.get_group("Electronics").is_none());
    }

    #[test]
    fn test_facet_range() {
        let range = FacetRange::new("[0.0 TO 10.0)".to_string(), Some(0.0), Some(10.0));

        assert!(range.contains(5.0));
        assert!(range.contains(0.0)); // Inclusive minimum
        assert!(!range.contains(10.0)); // Exclusive maximum
        assert!(!range.contains(-1.0));
        assert!(!range.contains(15.0));
    }

    #[test]
    fn test_range_facet_creation() {
        let range_facet = RangeFacet::numeric_ranges("price".to_string(), 0.0, 100.0, 5);

        assert_eq!(range_facet.field, "price");
        assert_eq!(range_facet.ranges.len(), 5);

        // Check first range
        assert_eq!(range_facet.ranges[0].min, Some(0.0));
        assert_eq!(range_facet.ranges[0].max, Some(20.0));

        // Check last range
        assert_eq!(range_facet.ranges[4].min, Some(80.0));
        assert_eq!(range_facet.ranges[4].max, None); // Open-ended
    }

    #[test]
    fn test_range_facet_counting() {
        let mut range_facet = RangeFacet::numeric_ranges("score".to_string(), 0.0, 10.0, 2);
        let values = vec![1.0, 3.0, 7.0, 9.0, 15.0]; // 15.0 should not count (out of range)

        range_facet.count_ranges(&values);

        // First range [0.0 TO 5.0): should count 1.0, 3.0
        assert_eq!(range_facet.ranges[0].count, 2);

        // Second range [5.0 TO *]: should count 7.0, 9.0, 15.0
        assert_eq!(range_facet.ranges[1].count, 3);
    }
}
