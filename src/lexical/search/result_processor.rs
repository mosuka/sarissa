//! Advanced result processing including aggregation, highlighting, and faceting.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::document::field_value::FieldValue;
use crate::error::Result;
use crate::lexical::index::inverted::query::Query;
use crate::lexical::index::inverted::query::QueryResult;
use crate::lexical::reader::LexicalIndexReader;
use crate::lexical::search::features::facet::{FacetCollector, FacetResults};
use crate::lexical::search::features::highlight::{HighlightConfig, Highlighter};

/// Configuration for result processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultProcessorConfig {
    /// Maximum number of results to return.
    pub max_results: usize,

    /// Enable highlighting.
    pub enable_highlighting: bool,

    /// Enable faceting.
    pub enable_faceting: bool,

    /// Enable snippets.
    pub enable_snippets: bool,

    /// Enable field retrieval.
    pub retrieve_fields: bool,

    /// Fields to retrieve.
    pub fields_to_retrieve: Vec<String>,

    /// Fields to highlight.
    pub fields_to_highlight: Vec<String>,

    /// Facet fields.
    pub facet_fields: Vec<String>,

    /// Snippet length.
    pub snippet_length: usize,

    /// Enable result grouping.
    pub enable_grouping: bool,

    /// Group by field.
    pub group_by_field: Option<String>,
}

impl Default for ResultProcessorConfig {
    fn default() -> Self {
        ResultProcessorConfig {
            max_results: 10,
            enable_highlighting: false,
            enable_faceting: false,
            enable_snippets: false,
            retrieve_fields: true,
            fields_to_retrieve: vec!["*".to_string()],
            fields_to_highlight: Vec::new(),
            facet_fields: Vec::new(),
            snippet_length: 200,
            enable_grouping: false,
            group_by_field: None,
        }
    }
}

/// Processed search results with additional metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedSearchResults {
    /// The search hits.
    pub hits: Vec<ProcessedHit>,

    /// Total number of matching documents.
    pub total_hits: u64,

    /// Maximum score in the results.
    pub max_score: f32,

    /// Facet results.
    pub facets: FacetResults,

    /// Aggregation results.
    pub aggregations: HashMap<String, AggregationResult>,

    /// Query suggestions.
    pub suggestions: Vec<String>,

    /// Grouped results.
    pub groups: Option<Vec<ResultGroup>>,

    /// Processing time in milliseconds.
    pub processing_time_ms: u64,
}

/// Enhanced search hit with processed data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedHit {
    /// The document ID.
    pub doc_id: u32,

    /// The relevance score.
    pub score: f32,

    /// The document fields.
    pub fields: HashMap<String, String>,

    /// Highlighted snippets.
    pub highlights: HashMap<String, Vec<String>>,

    /// Snippets for preview.
    pub snippets: HashMap<String, String>,

    /// Explanation of score (if requested).
    pub explanation: Option<ScoreExplanation>,
}

/// Group of results for grouping functionality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultGroup {
    /// Group key.
    pub group_key: String,

    /// Documents in this group.
    pub hits: Vec<ProcessedHit>,

    /// Total hits in this group.
    pub total_hits: u64,

    /// Maximum score in this group.
    pub max_score: f32,
}

/// Explanation of how a score was calculated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreExplanation {
    /// Score value.
    pub value: f32,

    /// Description of how score was calculated.
    pub description: String,

    /// Sub-explanations.
    pub details: Vec<ScoreExplanation>,
}

/// Aggregation result types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AggregationResult {
    /// Count aggregation.
    Count { count: u64 },

    /// Sum aggregation.
    Sum { sum: f64 },

    /// Average aggregation.
    Average { avg: f64, count: u64 },

    /// Min/Max aggregation.
    MinMax { min: f64, max: f64 },

    /// Terms aggregation (top terms).
    Terms { buckets: Vec<TermsBucket> },

    /// Range aggregation.
    Range { buckets: Vec<RangeBucket> },

    /// Date histogram aggregation.
    DateHistogram { buckets: Vec<DateHistogramBucket> },
}

/// Terms aggregation bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TermsBucket {
    /// Term key.
    pub key: String,

    /// Document count.
    pub doc_count: u64,
}

/// Range aggregation bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeBucket {
    /// Range key.
    pub key: String,

    /// From value (inclusive).
    pub from: Option<f64>,

    /// To value (exclusive).
    pub to: Option<f64>,

    /// Document count.
    pub doc_count: u64,
}

/// Date histogram bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateHistogramBucket {
    /// Date key (timestamp).
    pub key: u64,

    /// Date key as string.
    pub key_as_string: String,

    /// Document count.
    pub doc_count: u64,
}

/// Advanced result processor with multiple processing capabilities.
#[derive(Debug)]
pub struct ResultProcessor {
    /// Configuration.
    config: ResultProcessorConfig,

    /// Highlighter for text highlighting.
    highlighter: Option<Highlighter>,

    /// Facet collector.
    facet_collector: Option<FacetCollector>,

    /// Index reader for document retrieval.
    reader: Arc<dyn LexicalIndexReader>,
}

impl ResultProcessor {
    /// Create a new result processor.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for result processing behavior
    /// * `reader` - Index reader for document retrieval
    ///
    /// # Returns
    ///
    /// Result containing the new processor instance
    pub fn new(config: ResultProcessorConfig, reader: Arc<dyn LexicalIndexReader>) -> Result<Self> {
        let highlighter = if config.enable_highlighting {
            Some(Highlighter::new(HighlightConfig::default()))
        } else {
            None
        };

        let facet_collector = if config.enable_faceting {
            let facet_config = crate::lexical::search::features::facet::FacetConfig::default(); // Use default config
            Some(FacetCollector::new(
                facet_config,
                config.facet_fields.clone(),
            ))
        } else {
            None
        };

        Ok(ResultProcessor {
            config,
            highlighter,
            facet_collector,
            reader,
        })
    }

    /// Process raw query results into enriched results.
    ///
    /// Applies highlighting, faceting, snippets, and other enhancements.
    ///
    /// # Arguments
    ///
    /// * `raw_results` - Raw search results from query execution
    /// * `query` - Query object for highlighting
    ///
    /// # Returns
    ///
    /// Enriched search results with highlights, facets, aggregations, etc.
    pub fn process_results<Q: Query>(
        &mut self,
        raw_results: Vec<QueryResult>,
        query: &Q,
    ) -> Result<ProcessedSearchResults> {
        let start_time = std::time::Instant::now();

        // Limit results
        let limited_results: Vec<_> = raw_results
            .into_iter()
            .take(self.config.max_results)
            .collect();

        let total_hits = limited_results.len() as u64;
        let max_score = limited_results
            .iter()
            .map(|r| r.score)
            .fold(0.0f32, f32::max);

        // Process each hit
        let mut processed_hits = Vec::new();
        for result in &limited_results {
            let processed_hit = self.process_hit(result, query)?;
            processed_hits.push(processed_hit);
        }

        // Collect facets
        let facets = if let Some(ref mut collector) = self.facet_collector {
            // Note: We cannot use finalize here as it consumes the collector
            // In a real implementation, you would need to redesign this API
            for result in &limited_results {
                collector.collect_doc(result.doc_id, self.reader.as_ref())?;
            }
            FacetResults::empty() // Placeholder until API is fixed
        } else {
            FacetResults::empty()
        };

        // Collect aggregations
        let aggregations = self.collect_aggregations(&limited_results)?;

        // Group results if requested
        let groups = if self.config.enable_grouping {
            Some(self.group_results(&processed_hits)?)
        } else {
            None
        };

        // Generate suggestions (placeholder)
        let suggestions = Vec::new();

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(ProcessedSearchResults {
            hits: processed_hits,
            total_hits,
            max_score,
            facets,
            aggregations,
            suggestions,
            groups,
            processing_time_ms,
        })
    }

    /// Process a single hit.
    fn process_hit<Q: Query>(&self, result: &QueryResult, query: &Q) -> Result<ProcessedHit> {
        // Retrieve document fields
        let fields = if self.config.retrieve_fields {
            self.retrieve_document_fields(result.doc_id)?
        } else {
            HashMap::new()
        };

        // Generate highlights
        let highlights = if self.config.enable_highlighting && self.highlighter.is_some() {
            self.generate_highlights(result.doc_id, &fields, query)?
        } else {
            HashMap::new()
        };

        // Generate snippets
        let snippets = if self.config.enable_snippets {
            self.generate_snippets(result.doc_id, &fields)?
        } else {
            HashMap::new()
        };

        // Score explanation (placeholder)
        let explanation = None;

        Ok(ProcessedHit {
            doc_id: result.doc_id,
            score: result.score,
            fields,
            highlights,
            snippets,
            explanation,
        })
    }

    /// Retrieve document fields.
    fn retrieve_document_fields(&self, doc_id: u32) -> Result<HashMap<String, String>> {
        let mut fields = HashMap::new();

        if let Some(document) = self.reader.document(doc_id as u64)? {
            for (field_name, field_value) in document.fields() {
                // Check if field should be retrieved
                if self.should_retrieve_field(field_name) {
                    let value_str = self.field_value_to_string(field_value);
                    fields.insert(field_name.clone(), value_str);
                }
            }
        }

        Ok(fields)
    }

    /// Check if a field should be retrieved.
    fn should_retrieve_field(&self, field_name: &str) -> bool {
        if self.config.fields_to_retrieve.contains(&"*".to_string()) {
            return true;
        }

        self.config
            .fields_to_retrieve
            .contains(&field_name.to_string())
    }

    /// Convert field value to string.
    fn field_value_to_string(&self, field_value: &FieldValue) -> String {
        match field_value {
            FieldValue::Text(s) => s.clone(),
            FieldValue::Integer(i) => i.to_string(),
            FieldValue::Float(f) => f.to_string(),
            FieldValue::Boolean(b) => b.to_string(),
            FieldValue::Binary(_) => "[binary data]".to_string(),
            FieldValue::DateTime(dt) => dt.to_rfc3339(),
            FieldValue::Geo(point) => format!("{},{}", point.lat, point.lon),
            FieldValue::Null => "null".to_string(),
        }
    }

    /// Generate highlights for a document.
    fn generate_highlights<Q: Query>(
        &self,
        _doc_id: u32,
        fields: &HashMap<String, String>,
        query: &Q,
    ) -> Result<HashMap<String, Vec<String>>> {
        let mut highlights = HashMap::new();

        if let Some(ref highlighter) = self.highlighter {
            for field_name in &self.config.fields_to_highlight {
                if let Some(field_text) = fields.get(field_name) {
                    let highlighted = highlighter.highlight(query, field_name, field_text)?;
                    let highlighted_texts: Vec<String> = highlighted
                        .fragments
                        .iter()
                        .map(|f| f.text.clone())
                        .collect();
                    if !highlighted_texts.is_empty() {
                        highlights.insert(field_name.clone(), highlighted_texts);
                    }
                }
            }
        }

        Ok(highlights)
    }

    /// Generate snippets for a document.
    fn generate_snippets(
        &self,
        _doc_id: u32,
        fields: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>> {
        let mut snippets = HashMap::new();

        for (field_name, field_text) in fields {
            if field_text.len() > self.config.snippet_length {
                let snippet = field_text
                    .chars()
                    .take(self.config.snippet_length)
                    .collect::<String>()
                    + "...";
                snippets.insert(field_name.clone(), snippet);
            } else {
                snippets.insert(field_name.clone(), field_text.clone());
            }
        }

        Ok(snippets)
    }

    /// Collect facets from results.
    #[allow(dead_code)]
    fn collect_facets(
        &self,
        collector: &mut FacetCollector,
        results: &[QueryResult],
    ) -> Result<FacetResults> {
        Self::collect_facets_static(collector, results, &self.reader)
    }

    /// Static helper for collecting facets to avoid borrowing issues.
    #[allow(dead_code)]
    fn collect_facets_static(
        collector: &mut FacetCollector,
        results: &[QueryResult],
        reader: &Arc<dyn LexicalIndexReader>,
    ) -> Result<FacetResults> {
        for result in results {
            collector.collect_doc(result.doc_id, reader.as_ref())?;
        }

        // Note: Cannot call finalize here because it takes self by value
        Ok(FacetResults::empty()) // Placeholder
    }

    /// Static helper for finalizing facet collection.
    #[allow(dead_code)]
    fn collect_facets_finalize(
        mut collector: FacetCollector,
        results: &[QueryResult],
        reader: &Arc<dyn LexicalIndexReader>,
    ) -> Result<FacetResults> {
        for result in results {
            collector.collect_doc(result.doc_id, reader.as_ref())?;
        }

        collector.finalize()
    }

    /// Collect aggregations from results.
    fn collect_aggregations(
        &self,
        results: &[QueryResult],
    ) -> Result<HashMap<String, AggregationResult>> {
        let mut aggregations = HashMap::new();

        // Count aggregation
        aggregations.insert(
            "total_count".to_string(),
            AggregationResult::Count {
                count: results.len() as u64,
            },
        );

        // Score statistics
        if !results.is_empty() {
            let scores: Vec<f64> = results.iter().map(|r| r.score as f64).collect();
            let sum: f64 = scores.iter().sum();
            let avg = sum / scores.len() as f64;
            let min = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            aggregations.insert(
                "score_stats".to_string(),
                AggregationResult::Average {
                    avg,
                    count: scores.len() as u64,
                },
            );

            aggregations.insert(
                "score_range".to_string(),
                AggregationResult::MinMax { min, max },
            );
        }

        Ok(aggregations)
    }

    /// Group results by field.
    fn group_results(&self, hits: &[ProcessedHit]) -> Result<Vec<ResultGroup>> {
        let group_field = match &self.config.group_by_field {
            Some(field) => field,
            None => return Ok(Vec::new()),
        };

        let mut groups: BTreeMap<String, Vec<ProcessedHit>> = BTreeMap::new();

        for hit in hits {
            let group_key = hit
                .fields
                .get(group_field)
                .unwrap_or(&"[no value]".to_string())
                .clone();

            groups.entry(group_key).or_default().push(hit.clone());
        }

        let result_groups = groups
            .into_iter()
            .map(|(group_key, group_hits)| {
                let total_hits = group_hits.len() as u64;
                let max_score = group_hits.iter().map(|h| h.score).fold(0.0f32, f32::max);

                ResultGroup {
                    group_key,
                    hits: group_hits,
                    total_hits,
                    max_score,
                }
            })
            .collect();

        Ok(result_groups)
    }
}

/// Builder for result processor configuration.
#[derive(Debug)]
pub struct ResultProcessorBuilder {
    config: ResultProcessorConfig,
}

impl ResultProcessorBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        ResultProcessorBuilder {
            config: ResultProcessorConfig::default(),
        }
    }

    /// Set maximum results.
    pub fn max_results(mut self, max_results: usize) -> Self {
        self.config.max_results = max_results;
        self
    }

    /// Enable highlighting.
    pub fn enable_highlighting(mut self, fields: Vec<String>) -> Self {
        self.config.enable_highlighting = true;
        self.config.fields_to_highlight = fields;
        self
    }

    /// Enable faceting.
    pub fn enable_faceting(mut self, fields: Vec<String>) -> Self {
        self.config.enable_faceting = true;
        self.config.facet_fields = fields;
        self
    }

    /// Enable snippets.
    pub fn enable_snippets(mut self, length: usize) -> Self {
        self.config.enable_snippets = true;
        self.config.snippet_length = length;
        self
    }

    /// Set fields to retrieve.
    pub fn retrieve_fields(mut self, fields: Vec<String>) -> Self {
        self.config.retrieve_fields = true;
        self.config.fields_to_retrieve = fields;
        self
    }

    /// Enable grouping.
    pub fn enable_grouping(mut self, field: String) -> Self {
        self.config.enable_grouping = true;
        self.config.group_by_field = Some(field);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> ResultProcessorConfig {
        self.config
    }
}

impl Default for ResultProcessorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    #[test]
    fn test_result_processor_config() {
        let config = ResultProcessorConfig {
            max_results: 20,
            enable_highlighting: true,
            fields_to_highlight: vec!["title".to_string()],
            ..Default::default()
        };

        assert_eq!(config.max_results, 20);
        assert!(config.enable_highlighting);
        assert_eq!(config.fields_to_highlight, vec!["title"]);
    }

    #[test]
    fn test_result_processor_builder() {
        let config = ResultProcessorBuilder::new()
            .max_results(50)
            .enable_highlighting(vec!["title".to_string(), "content".to_string()])
            .enable_faceting(vec!["category".to_string()])
            .enable_snippets(300)
            .build();

        assert_eq!(config.max_results, 50);
        assert!(config.enable_highlighting);
        assert!(config.enable_faceting);
        assert!(config.enable_snippets);
        assert_eq!(config.snippet_length, 300);
    }

    #[test]
    fn test_processed_hit_structure() {
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "Test Title".to_string());

        let hit = ProcessedHit {
            doc_id: 1,
            score: 0.95,
            fields,
            highlights: HashMap::new(),
            snippets: HashMap::new(),
            explanation: None,
        };

        assert_eq!(hit.doc_id, 1);
        assert_eq!(hit.score, 0.95);
        assert_eq!(hit.fields.get("title"), Some(&"Test Title".to_string()));
    }

    #[test]
    fn test_aggregation_results() {
        let count_agg = AggregationResult::Count { count: 100 };
        let avg_agg = AggregationResult::Average {
            avg: 0.75,
            count: 50,
        };

        match count_agg {
            AggregationResult::Count { count } => assert_eq!(count, 100),
            _ => panic!("Expected Count aggregation"),
        }

        match avg_agg {
            AggregationResult::Average { avg, count } => {
                assert_eq!(avg, 0.75);
                assert_eq!(count, 50);
            }
            _ => panic!("Expected Average aggregation"),
        }
    }
}
