//! Batch processing functionality for efficient document indexing.

use crate::error::{Result, SarissaError};
use crate::parallel_index::{
    config::IndexingOptions,
    partitioner::DocumentPartitioner,
    writer_manager::IndexWriterHandle,
};
use crate::schema::Document;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Represents a batch of documents for a specific partition.
#[derive(Debug)]
pub struct DocumentBatch {
    /// Partition index.
    pub partition_index: usize,
    
    /// Documents in this batch.
    pub documents: Vec<Document>,
    
    /// Batch creation timestamp.
    pub created_at: Instant,
    
    /// Estimated memory usage in bytes.
    pub estimated_size: usize,
}

impl DocumentBatch {
    /// Create a new document batch.
    pub fn new(partition_index: usize) -> Self {
        Self {
            partition_index,
            documents: Vec::new(),
            created_at: Instant::now(),
            estimated_size: 0,
        }
    }
    
    /// Add a document to the batch.
    pub fn add_document(&mut self, doc: Document) {
        // Rough estimate of document size
        self.estimated_size += std::mem::size_of::<Document>() + 
            doc.fields().iter()
                .map(|(k, v)| k.len() + format!("{v:?}").len())
                .sum::<usize>();
        
        self.documents.push(doc);
    }
    
    /// Check if the batch is full based on size or count limits.
    pub fn is_full(&self, max_size: usize, max_count: usize) -> bool {
        self.documents.len() >= max_count || self.estimated_size >= max_size
    }
    
    /// Check if the batch has expired based on age.
    pub fn is_expired(&self, max_age: Duration) -> bool {
        self.created_at.elapsed() > max_age
    }
    
    /// Get the number of documents in the batch.
    pub fn len(&self) -> usize {
        self.documents.len()
    }
    
    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

/// Configuration for batch processing.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of documents per batch.
    pub max_batch_size: usize,
    
    /// Maximum memory usage per batch in bytes.
    pub max_batch_memory: usize,
    
    /// Maximum age before forcing batch processing.
    pub max_batch_age: Duration,
    
    /// Whether to process batches as soon as they're full.
    pub immediate_processing: bool,
    
    /// Buffer size for incoming documents.
    pub input_buffer_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 1000,
            max_batch_memory: 16 * 1024 * 1024, // 16MB
            max_batch_age: Duration::from_secs(10),
            immediate_processing: true,
            input_buffer_size: 10000,
        }
    }
}

/// Result of processing a batch of documents.
#[derive(Debug)]
pub struct BatchProcessingResult {
    /// Partition index that was processed.
    pub partition_index: usize,
    
    /// Number of documents successfully indexed.
    pub documents_indexed: usize,
    
    /// Number of documents that failed to index.
    pub documents_failed: usize,
    
    /// Time taken to process the batch.
    pub processing_time: Duration,
    
    /// Any errors encountered during processing.
    pub errors: Vec<SarissaError>,
}

/// Batch processor for managing document batches and processing them efficiently.
pub struct BatchProcessor {
    /// Configuration for batch processing.
    config: BatchConfig,
    
    /// Current batches being accumulated (partition_index -> batch).
    active_batches: HashMap<usize, DocumentBatch>,
    
    /// Total number of partitions.
    partition_count: usize,
    
    /// Statistics.
    total_documents_processed: u64,
    total_batches_processed: u64,
    total_processing_time: Duration,
}

impl BatchProcessor {
    /// Create a new batch processor.
    pub fn new(config: BatchConfig, partition_count: usize) -> Self {
        Self {
            config,
            active_batches: HashMap::new(),
            partition_count,
            total_documents_processed: 0,
            total_batches_processed: 0,
            total_processing_time: Duration::ZERO,
        }
    }
    
    /// Add a document to the appropriate batch.
    /// Returns true if a batch is ready for processing.
    pub fn add_document(&mut self, partition_index: usize, document: Document) -> Result<bool> {
        if partition_index >= self.partition_count {
            return Err(SarissaError::invalid_argument(format!(
                "Partition index {} is out of range (max: {})",
                partition_index, self.partition_count - 1
            )));
        }
        
        let batch = self.active_batches
            .entry(partition_index)
            .or_insert_with(|| DocumentBatch::new(partition_index));
        
        batch.add_document(document);
        
        // Check if batch is ready for processing
        Ok(batch.is_full(self.config.max_batch_memory, self.config.max_batch_size) ||
           batch.is_expired(self.config.max_batch_age))
    }
    
    /// Get ready batches (full or expired).
    pub fn get_ready_batches(&mut self) -> Vec<DocumentBatch> {
        let mut ready_batches = Vec::new();
        let mut partitions_to_remove = Vec::new();
        
        for (&partition_index, batch) in &self.active_batches {
            if batch.is_full(self.config.max_batch_memory, self.config.max_batch_size) ||
               batch.is_expired(self.config.max_batch_age) {
                partitions_to_remove.push(partition_index);
            }
        }
        
        for partition_index in partitions_to_remove {
            if let Some(batch) = self.active_batches.remove(&partition_index) {
                if !batch.is_empty() {
                    ready_batches.push(batch);
                }
            }
        }
        
        ready_batches
    }
    
    /// Force flush all active batches.
    pub fn flush_all(&mut self) -> Vec<DocumentBatch> {
        let mut batches = Vec::new();
        
        for (_, batch) in self.active_batches.drain() {
            if !batch.is_empty() {
                batches.push(batch);
            }
        }
        
        batches
    }
    
    /// Process a batch using the given writer handle.
    pub fn process_batch(
        &mut self,
        batch: DocumentBatch,
        writer_handle: &IndexWriterHandle,
        options: &IndexingOptions,
    ) -> BatchProcessingResult {
        let start_time = Instant::now();
        let mut documents_indexed = 0;
        let mut documents_failed = 0;
        let mut errors = Vec::new();
        
        // Validate documents if requested
        let documents = if options.validate_documents {
            self.validate_documents(batch.documents, &mut errors)
        } else {
            batch.documents
        };
        
        // Process documents in chunks if the batch is very large
        let chunk_size = self.config.max_batch_size.min(1000);
        for chunk in documents.chunks(chunk_size) {
            match writer_handle.index_documents(chunk.to_vec()) {
                Ok(()) => documents_indexed += chunk.len(),
                Err(e) => {
                    documents_failed += chunk.len();
                    errors.push(e);
                    
                    // Check if we should stop on too many errors
                    if let Some(max_errors) = options.max_errors {
                        if errors.len() >= max_errors {
                            break;
                        }
                    }
                }
            }
        }
        
        // Force commit if requested
        if options.force_commit {
            if let Err(e) = writer_handle.commit() {
                errors.push(e);
            }
        }
        
        let processing_time = start_time.elapsed();
        
        // Update statistics
        self.total_documents_processed += documents_indexed as u64;
        self.total_batches_processed += 1;
        self.total_processing_time += processing_time;
        
        BatchProcessingResult {
            partition_index: batch.partition_index,
            documents_indexed,
            documents_failed,
            processing_time,
            errors,
        }
    }
    
    /// Validate a list of documents, removing invalid ones.
    fn validate_documents(&self, documents: Vec<Document>, errors: &mut Vec<SarissaError>) -> Vec<Document> {
        let mut valid_documents = Vec::new();
        
        for doc in documents {
            // Basic validation - check if document has any fields
            if doc.fields().is_empty() {
                errors.push(SarissaError::field("Document has no fields".to_string()));
                continue;
            }
            
            // Add more validation rules here as needed
            valid_documents.push(doc);
        }
        
        valid_documents
    }
    
    /// Get processing statistics.
    pub fn get_statistics(&self) -> BatchProcessingStatistics {
        BatchProcessingStatistics {
            total_documents_processed: self.total_documents_processed,
            total_batches_processed: self.total_batches_processed,
            total_processing_time: self.total_processing_time,
            active_batches_count: self.active_batches.len(),
            average_batch_processing_time: if self.total_batches_processed > 0 {
                self.total_processing_time / self.total_batches_processed as u32
            } else {
                Duration::ZERO
            },
            average_documents_per_second: if self.total_processing_time.as_secs() > 0 {
                self.total_documents_processed as f64 / self.total_processing_time.as_secs_f64()
            } else {
                0.0
            },
        }
    }
    
    /// Reset statistics.
    pub fn reset_statistics(&mut self) {
        self.total_documents_processed = 0;
        self.total_batches_processed = 0;
        self.total_processing_time = Duration::ZERO;
    }
}

/// Statistics for batch processing.
#[derive(Debug, Clone)]
pub struct BatchProcessingStatistics {
    /// Total number of documents processed.
    pub total_documents_processed: u64,
    
    /// Total number of batches processed.
    pub total_batches_processed: u64,
    
    /// Total time spent processing batches.
    pub total_processing_time: Duration,
    
    /// Number of currently active batches.
    pub active_batches_count: usize,
    
    /// Average time per batch.
    pub average_batch_processing_time: Duration,
    
    /// Average documents processed per second.
    pub average_documents_per_second: f64,
}

/// Utility function to partition documents into batches efficiently.
pub fn partition_documents_into_batches(
    documents: Vec<Document>,
    partitioner: &dyn DocumentPartitioner,
    batch_config: &BatchConfig,
) -> Result<HashMap<usize, Vec<DocumentBatch>>> {
    let mut processor = BatchProcessor::new(batch_config.clone(), partitioner.partition_count());
    let mut partition_batches: HashMap<usize, Vec<DocumentBatch>> = HashMap::new();
    
    for document in documents {
        let partition_index = partitioner.partition(&document)?;
        
        if processor.add_document(partition_index, document)? {
            // Get ready batches and distribute them
            let ready_batches = processor.get_ready_batches();
            for batch in ready_batches {
                partition_batches
                    .entry(batch.partition_index)
                    .or_default()
                    .push(batch);
            }
        }
    }
    
    // Flush any remaining batches
    let remaining_batches = processor.flush_all();
    for batch in remaining_batches {
        partition_batches
            .entry(batch.partition_index)
            .or_default()
            .push(batch);
    }
    
    Ok(partition_batches)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel_index::partitioner::HashPartitioner;
    use crate::schema::FieldValue;
    
    fn create_test_document(id: &str) -> Document {
        let mut doc = Document::new();
        doc.add_field("id".to_string(), FieldValue::Text(id.to_string()));
        doc.add_field("content".to_string(), FieldValue::Text("test content".to_string()));
        doc
    }
    
    #[test]
    fn test_document_batch() {
        let mut batch = DocumentBatch::new(0);
        assert!(batch.is_empty());
        
        batch.add_document(create_test_document("doc1"));
        batch.add_document(create_test_document("doc2"));
        
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert!(batch.estimated_size > 0);
    }
    
    #[test]
    fn test_batch_processor() {
        let config = BatchConfig {
            max_batch_size: 2,
            max_batch_memory: 1024 * 1024,
            max_batch_age: Duration::from_secs(1),
            immediate_processing: true,
            input_buffer_size: 100,
        };
        
        let mut processor = BatchProcessor::new(config, 2);
        
        // Add documents to partition 0
        assert!(!processor.add_document(0, create_test_document("doc1")).unwrap());
        assert!(processor.add_document(0, create_test_document("doc2")).unwrap()); // Should be ready
        
        let ready_batches = processor.get_ready_batches();
        assert_eq!(ready_batches.len(), 1);
        assert_eq!(ready_batches[0].partition_index, 0);
        assert_eq!(ready_batches[0].len(), 2);
    }
    
    #[test]
    fn test_partition_documents_into_batches() {
        let documents = vec![
            create_test_document("doc1"),
            create_test_document("doc2"),
            create_test_document("doc3"),
            create_test_document("doc4"),
        ];
        
        let partitioner = HashPartitioner::new("id".to_string(), 2);
        let config = BatchConfig {
            max_batch_size: 2,
            ..Default::default()
        };
        
        let partition_batches = partition_documents_into_batches(documents, &partitioner, &config).unwrap();
        
        // Should have batches distributed across partitions
        assert!(!partition_batches.is_empty());
        
        let total_docs: usize = partition_batches.values()
            .flat_map(|batches| batches.iter())
            .map(|batch| batch.len())
            .sum();
        
        assert_eq!(total_docs, 4);
    }
    
    #[test]
    fn test_batch_age_expiration() {
        let config = BatchConfig {
            max_batch_size: 10,
            max_batch_age: Duration::from_millis(1),
            ..Default::default()
        };
        
        let mut batch = DocumentBatch::new(0);
        batch.add_document(create_test_document("doc1"));
        
        // Wait a bit for the batch to expire
        std::thread::sleep(Duration::from_millis(2));
        
        assert!(batch.is_expired(config.max_batch_age));
    }
}