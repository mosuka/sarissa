//! Document partitioning strategies for distributing documents across multiple indices.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use ahash::AHasher;

use crate::document::Document;
use crate::error::{Result, SageError};

/// Trait for partitioning documents across multiple indices.
pub trait DocumentPartitioner: Send + Sync {
    /// Determine which partition a document should be assigned to.
    /// Returns the partition index (0-based).
    fn partition(&self, doc: &Document) -> Result<usize>;

    /// Partition a batch of documents efficiently.
    /// Returns a vector of (partition_index, document) pairs.
    fn partition_batch(&self, docs: Vec<Document>) -> Result<Vec<(usize, Document)>> {
        let mut result = Vec::with_capacity(docs.len());

        for doc in docs {
            let partition_index = self.partition(&doc)?;
            result.push((partition_index, doc));
        }

        Ok(result)
    }

    /// Get the total number of partitions this partitioner supports.
    fn partition_count(&self) -> usize;

    /// Get a human-readable description of this partitioner.
    fn description(&self) -> String;

    /// Validate that the partitioner is properly configured.
    fn validate(&self) -> Result<()> {
        if self.partition_count() == 0 {
            return Err(SageError::invalid_argument(
                "Partition count cannot be zero",
            ));
        }
        Ok(())
    }
}

/// Hash-based document partitioner.
/// Distributes documents based on the hash value of a specified field.
#[derive(Debug, Clone)]
pub struct HashPartitioner {
    /// Field name to hash for partitioning.
    field_name: String,

    /// Number of partitions.
    partition_count: usize,

    /// Optional seed for hash function.
    hash_seed: Option<u64>,
}

impl HashPartitioner {
    /// Create a new hash partitioner.
    pub fn new(field_name: String, partition_count: usize) -> Self {
        Self {
            field_name,
            partition_count,
            hash_seed: None,
        }
    }

    /// Create a hash partitioner with a custom hash seed.
    pub fn with_seed(field_name: String, partition_count: usize, seed: u64) -> Self {
        Self {
            field_name,
            partition_count,
            hash_seed: Some(seed),
        }
    }

    /// Get the field name used for hashing.
    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    /// Calculate hash for a field value.
    fn calculate_hash(&self, value: &str) -> u64 {
        let mut hasher = AHasher::default();

        if let Some(seed) = self.hash_seed {
            hasher.write_u64(seed);
        }

        value.hash(&mut hasher);
        hasher.finish()
    }
}

impl DocumentPartitioner for HashPartitioner {
    fn partition(&self, doc: &Document) -> Result<usize> {
        let field_value = doc.get_field(&self.field_name).ok_or_else(|| {
            SageError::field(format!("Field '{}' not found in document", self.field_name))
        })?;

        let value_str = format!("{field_value:?}");
        let hash = self.calculate_hash(&value_str);
        let partition_index = (hash % self.partition_count as u64) as usize;

        Ok(partition_index)
    }

    fn partition_count(&self) -> usize {
        self.partition_count
    }

    fn description(&self) -> String {
        format!(
            "HashPartitioner(field='{}', partitions={}, seed={:?})",
            self.field_name, self.partition_count, self.hash_seed
        )
    }
}

/// Range-based document partitioner.
/// Distributes documents based on numeric or date ranges.
#[derive(Debug, Clone)]
pub struct RangePartitioner {
    /// Field name to use for range partitioning.
    field_name: String,

    /// Range boundaries (must be sorted).
    boundaries: Vec<i64>,
}

impl RangePartitioner {
    /// Create a new range partitioner with integer boundaries.
    pub fn new(field_name: String, boundaries: Vec<i64>) -> Result<Self> {
        if boundaries.is_empty() {
            return Err(SageError::invalid_argument(
                "Range boundaries cannot be empty",
            ));
        }

        let mut sorted_boundaries = boundaries;
        sorted_boundaries.sort_unstable();

        Ok(Self {
            field_name,
            boundaries: sorted_boundaries,
        })
    }

    /// Create a range partitioner for date ranges.
    pub fn new_date_ranges(field_name: String, date_ranges: Vec<&str>) -> Result<Self> {
        let mut boundaries = Vec::new();

        for date_str in date_ranges {
            let timestamp = chrono::DateTime::parse_from_rfc3339(date_str)
                .map_err(|e| {
                    SageError::invalid_argument(format!("Invalid date format '{date_str}': {e}"))
                })?
                .timestamp();
            boundaries.push(timestamp);
        }

        Self::new(field_name, boundaries)
    }

    /// Get the field name used for range partitioning.
    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    /// Find the appropriate partition for a numeric value.
    fn find_partition(&self, value: i64) -> usize {
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            if value < boundary {
                return i;
            }
        }
        // If value is greater than all boundaries, goes to the last partition
        self.boundaries.len()
    }
}

impl DocumentPartitioner for RangePartitioner {
    fn partition(&self, doc: &Document) -> Result<usize> {
        let field_value = doc.get_field(&self.field_name).ok_or_else(|| {
            SageError::field(format!("Field '{}' not found in document", self.field_name))
        })?;

        // Try to convert field value to i64
        let numeric_value = match field_value {
            crate::document::FieldValue::Integer(i) => *i,
            crate::document::FieldValue::Text(s) => s.parse::<i64>().map_err(|_| {
                SageError::field(format!("Cannot convert field value '{s}' to integer"))
            })?,
            _ => {
                return Err(SageError::field(format!(
                    "Field '{}' is not numeric or text",
                    self.field_name
                )));
            }
        };

        Ok(self.find_partition(numeric_value))
    }

    fn partition_count(&self) -> usize {
        self.boundaries.len() + 1
    }

    fn description(&self) -> String {
        format!(
            "RangePartitioner(field='{}', boundaries={:?})",
            self.field_name, self.boundaries
        )
    }
}

/// Value-based document partitioner.
/// Maps specific field values to specific partitions.
#[derive(Debug, Clone)]
pub struct ValuePartitioner {
    /// Field name to use for value mapping.
    field_name: String,

    /// Mapping from field values to partition indices.
    value_mapping: HashMap<String, usize>,

    /// Default partition for unmapped values.
    default_partition: Option<usize>,

    /// Total number of partitions.
    partition_count: usize,
}

impl ValuePartitioner {
    /// Create a new value partitioner.
    pub fn new(field_name: String, partition_count: usize) -> Self {
        Self {
            field_name,
            value_mapping: HashMap::new(),
            default_partition: None,
            partition_count,
        }
    }

    /// Add a value mapping.
    pub fn add_mapping(mut self, value: String, partition: usize) -> Result<Self> {
        if partition >= self.partition_count {
            return Err(SageError::invalid_argument(format!(
                "Partition index {} is out of range (max: {})",
                partition,
                self.partition_count - 1
            )));
        }

        self.value_mapping.insert(value, partition);
        Ok(self)
    }

    /// Set the default partition for unmapped values.
    pub fn with_default_partition(mut self, partition: usize) -> Result<Self> {
        if partition >= self.partition_count {
            return Err(SageError::invalid_argument(format!(
                "Default partition index {} is out of range (max: {})",
                partition,
                self.partition_count - 1
            )));
        }

        self.default_partition = Some(partition);
        Ok(self)
    }

    /// Create a value partitioner from a mapping.
    pub fn from_mapping(
        field_name: String,
        mapping: HashMap<String, usize>,
        default_partition: Option<usize>,
    ) -> Result<Self> {
        if mapping.is_empty() {
            return Err(SageError::invalid_argument(
                "Value mapping cannot be empty",
            ));
        }

        let max_partition = mapping.values().max().copied().unwrap_or(0);
        let partition_count = if let Some(default) = default_partition {
            max_partition.max(default) + 1
        } else {
            max_partition + 1
        };

        Ok(Self {
            field_name,
            value_mapping: mapping,
            default_partition,
            partition_count,
        })
    }

    /// Get the field name used for value mapping.
    pub fn field_name(&self) -> &str {
        &self.field_name
    }
}

impl DocumentPartitioner for ValuePartitioner {
    fn partition(&self, doc: &Document) -> Result<usize> {
        let field_value = doc.get_field(&self.field_name).ok_or_else(|| {
            SageError::field(format!("Field '{}' not found in document", self.field_name))
        })?;

        // Extract the actual string value from FieldValue
        let value_str = match field_value {
            crate::document::FieldValue::Text(s) => s.clone(),
            crate::document::FieldValue::Integer(i) => i.to_string(),
            crate::document::FieldValue::Float(f) => f.to_string(),
            crate::document::FieldValue::Boolean(b) => b.to_string(),
            crate::document::FieldValue::Binary(b) => format!("{b:?}"),
            crate::document::FieldValue::DateTime(dt) => dt.to_rfc3339(),
            crate::document::FieldValue::Geo(point) => format!("{},{}", point.lat, point.lon),
            crate::document::FieldValue::Null => "null".to_string(),
        };

        if let Some(&partition) = self.value_mapping.get(&value_str) {
            Ok(partition)
        } else if let Some(default) = self.default_partition {
            Ok(default)
        } else {
            Err(SageError::field(format!(
                "No mapping found for value '{value_str}' and no default partition configured"
            )))
        }
    }

    fn partition_count(&self) -> usize {
        self.partition_count
    }

    fn description(&self) -> String {
        format!(
            "ValuePartitioner(field='{}', mappings={}, default={:?})",
            self.field_name,
            self.value_mapping.len(),
            self.default_partition
        )
    }
}

/// Round-robin document partitioner.
/// Distributes documents evenly across partitions in sequence.
#[derive(Debug)]
pub struct RoundRobinPartitioner {
    /// Number of partitions.
    partition_count: usize,

    /// Current partition counter (thread-safe).
    counter: std::sync::atomic::AtomicUsize,
}

impl RoundRobinPartitioner {
    /// Create a new round-robin partitioner.
    pub fn new(partition_count: usize) -> Self {
        Self {
            partition_count,
            counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl DocumentPartitioner for RoundRobinPartitioner {
    fn partition(&self, _doc: &Document) -> Result<usize> {
        let next = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(next % self.partition_count)
    }

    fn partition_count(&self) -> usize {
        self.partition_count
    }

    fn description(&self) -> String {
        format!("RoundRobinPartitioner(partitions={})", self.partition_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Document, FieldValue};

    fn create_test_document(field_name: &str, value: FieldValue) -> Document {
        let mut doc = Document::new();
        doc.add_field(field_name.to_string(), value);
        doc
    }

    #[test]
    fn test_hash_partitioner() {
        let partitioner = HashPartitioner::new("category".to_string(), 3);

        let doc1 = create_test_document("category", FieldValue::Text("electronics".to_string()));
        let doc2 = create_test_document("category", FieldValue::Text("books".to_string()));
        let doc3 = create_test_document("category", FieldValue::Text("electronics".to_string()));

        let partition1 = partitioner.partition(&doc1).unwrap();
        let partition2 = partitioner.partition(&doc2).unwrap();
        let partition3 = partitioner.partition(&doc3).unwrap();

        assert!(partition1 < 3);
        assert!(partition2 < 3);
        assert_eq!(partition1, partition3); // Same value should go to same partition

        assert_eq!(partitioner.partition_count(), 3);
        assert!(partitioner.description().contains("HashPartitioner"));
    }

    #[test]
    fn test_range_partitioner() {
        let partitioner = RangePartitioner::new("price".to_string(), vec![100, 500, 1000]).unwrap();

        let doc1 = create_test_document("price", FieldValue::Integer(50)); // Should go to partition 0
        let doc2 = create_test_document("price", FieldValue::Integer(250)); // Should go to partition 1
        let doc3 = create_test_document("price", FieldValue::Integer(750)); // Should go to partition 2
        let doc4 = create_test_document("price", FieldValue::Integer(1500)); // Should go to partition 3

        assert_eq!(partitioner.partition(&doc1).unwrap(), 0);
        assert_eq!(partitioner.partition(&doc2).unwrap(), 1);
        assert_eq!(partitioner.partition(&doc3).unwrap(), 2);
        assert_eq!(partitioner.partition(&doc4).unwrap(), 3);

        assert_eq!(partitioner.partition_count(), 4); // boundaries.len() + 1
    }

    #[test]
    fn test_value_partitioner() {
        let mut mapping = HashMap::new();
        mapping.insert("US".to_string(), 0);
        mapping.insert("EU".to_string(), 1);
        mapping.insert("Asia".to_string(), 2);

        let partitioner = ValuePartitioner::from_mapping(
            "region".to_string(),
            mapping,
            Some(3), // Default partition
        )
        .unwrap();

        let doc1 = create_test_document("region", FieldValue::Text("US".to_string()));
        let doc2 = create_test_document("region", FieldValue::Text("EU".to_string()));
        let doc3 = create_test_document("region", FieldValue::Text("Unknown".to_string()));

        assert_eq!(partitioner.partition(&doc1).unwrap(), 0);
        assert_eq!(partitioner.partition(&doc2).unwrap(), 1);
        assert_eq!(partitioner.partition(&doc3).unwrap(), 3); // Default partition

        assert_eq!(partitioner.partition_count(), 4);
    }

    #[test]
    fn test_round_robin_partitioner() {
        let partitioner = RoundRobinPartitioner::new(3);

        let doc = create_test_document("field", FieldValue::Text("value".to_string()));

        // Should cycle through partitions 0, 1, 2, 0, 1, 2, ...
        let partitions: Vec<_> = (0..6)
            .map(|_| partitioner.partition(&doc).unwrap())
            .collect();

        assert_eq!(partitions, vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(partitioner.partition_count(), 3);
    }

    #[test]
    fn test_batch_partitioning() {
        let partitioner = HashPartitioner::new("id".to_string(), 2);

        let docs = vec![
            create_test_document("id", FieldValue::Text("doc1".to_string())),
            create_test_document("id", FieldValue::Text("doc2".to_string())),
            create_test_document("id", FieldValue::Text("doc3".to_string())),
        ];

        let partitioned = partitioner.partition_batch(docs).unwrap();

        assert_eq!(partitioned.len(), 3);
        for (partition_idx, _doc) in partitioned {
            assert!(partition_idx < 2);
        }
    }

    #[test]
    fn test_partitioner_validation() {
        let partitioner = HashPartitioner::new("field".to_string(), 0);
        assert!(partitioner.validate().is_err());

        let partitioner = HashPartitioner::new("field".to_string(), 1);
        assert!(partitioner.validate().is_ok());
    }
}
