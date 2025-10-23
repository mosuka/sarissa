//! Vector index writer for persisting indexes to storage.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use crate::error::{Result, SageError};
use crate::vector::index::VectorIndexBuilder;

/// Configuration for vector index writer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorWriterConfig {
    /// Buffer size for writing.
    pub buffer_size: usize,
    /// Whether to compress the index data.
    pub compress: bool,
    /// Whether to sync data to disk immediately.
    pub sync: bool,
    /// File permissions (Unix only).
    pub file_permissions: Option<u32>,
}

impl Default for VectorWriterConfig {
    fn default() -> Self {
        Self {
            buffer_size: 64 * 1024, // 64KB
            compress: true,
            sync: true,
            file_permissions: None,
        }
    }
}

/// Writer for persisting vector indexes to storage.
pub struct VectorIndexWriter {
    output_path: PathBuf,
    config: VectorWriterConfig,
    writer: Option<BufWriter<File>>,
}

impl VectorIndexWriter {
    /// Create a new vector index writer.
    pub fn new<P: AsRef<Path>>(output_path: P) -> Result<Self> {
        Ok(Self {
            output_path: output_path.as_ref().to_path_buf(),
            config: VectorWriterConfig::default(),
            writer: None,
        })
    }

    /// Create a new vector index writer with custom configuration.
    pub fn with_config<P: AsRef<Path>>(output_path: P, config: VectorWriterConfig) -> Result<Self> {
        Ok(Self {
            output_path: output_path.as_ref().to_path_buf(),
            config,
            writer: None,
        })
    }

    /// Write a vector index to storage.
    pub fn write_index(&mut self, builder: &dyn VectorIndexBuilder) -> Result<()> {
        // Create output directory if it doesn't exist
        if let Some(parent) = self.output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| SageError::other(format!("Failed to create output directory: {e}")))?;
        }

        // Open file for writing
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.output_path)
            .map_err(|e| SageError::other(format!("Failed to open output file: {e}")))?;

        #[cfg(unix)]
        if let Some(permissions) = self.config.file_permissions {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = file
                .metadata()
                .map_err(|e| SageError::other(format!("Failed to get file metadata: {e}")))?
                .permissions();
            perms.set_mode(permissions);
            file.set_permissions(perms)
                .map_err(|e| SageError::other(format!("Failed to set file permissions: {e}")))?;
        }

        let mut writer = BufWriter::with_capacity(self.config.buffer_size, file);

        // Write index header
        self.write_header(&mut writer, builder)?;

        // Write index data
        self.write_index_data(&mut writer, builder)?;

        // Write footer/metadata
        self.write_footer(&mut writer, builder)?;

        self.writer = Some(writer);
        Ok(())
    }

    /// Flush all pending writes to disk.
    pub fn flush(&mut self) -> Result<()> {
        if let Some(ref mut writer) = self.writer {
            writer
                .flush()
                .map_err(|e| SageError::other(format!("Failed to flush writer: {e}")))?;

            if self.config.sync {
                writer
                    .get_ref()
                    .sync_all()
                    .map_err(|e| SageError::other(format!("Failed to sync to disk: {e}")))?;
            }
        }
        Ok(())
    }

    /// Write index header with metadata.
    fn write_header(
        &self,
        writer: &mut BufWriter<File>,
        builder: &dyn VectorIndexBuilder,
    ) -> Result<()> {
        let header = IndexHeader {
            magic: b"VSRX".to_vec(), // Vector Sage indeX
            version: 1,
            dimension: self.get_dimension(builder),
            vector_count: self.get_vector_count(builder),
            index_type: self.get_index_type(builder),
            compressed: self.config.compress,
            timestamp: chrono::Utc::now().timestamp(),
        };

        let header_bytes =
            bincode::encode_to_vec(&header, bincode::config::standard()).map_err(|e| {
                SageError::SerializationError(format!("Failed to serialize header: {e}"))
            })?;

        writer
            .write_all(&(header_bytes.len() as u32).to_le_bytes())
            .map_err(|e| SageError::other(format!("Failed to write header length: {e}")))?;

        writer
            .write_all(&header_bytes)
            .map_err(|e| SageError::other(format!("Failed to write header: {e}")))?;

        Ok(())
    }

    /// Write the actual index data.
    fn write_index_data(
        &self,
        writer: &mut BufWriter<File>,
        _builder: &dyn VectorIndexBuilder,
    ) -> Result<()> {
        // This would serialize the actual index structure
        // For now, we'll write a placeholder
        let placeholder_data = b"INDEX_DATA_PLACEHOLDER";

        writer
            .write_all(&(placeholder_data.len() as u64).to_le_bytes())
            .map_err(|e| SageError::other(format!("Failed to write data length: {e}")))?;

        if self.config.compress {
            // Here we would compress the data before writing
            // For now, just write as-is
            writer
                .write_all(placeholder_data)
                .map_err(|e| SageError::other(format!("Failed to write index data: {e}")))?;
        } else {
            writer
                .write_all(placeholder_data)
                .map_err(|e| SageError::other(format!("Failed to write index data: {e}")))?;
        }

        Ok(())
    }

    /// Write index footer with checksums and metadata.
    fn write_footer(
        &self,
        writer: &mut BufWriter<File>,
        builder: &dyn VectorIndexBuilder,
    ) -> Result<()> {
        let footer = IndexFooter {
            checksum: 0x12345678, // Placeholder checksum
            build_stats: BuildStats {
                build_time_ms: 0, // Would get from builder
                memory_usage: builder.estimated_memory_usage(),
                optimization_applied: true,
            },
        };

        let footer_bytes =
            bincode::encode_to_vec(&footer, bincode::config::standard()).map_err(|e| {
                SageError::SerializationError(format!("Failed to serialize footer: {e}"))
            })?;

        writer
            .write_all(&(footer_bytes.len() as u32).to_le_bytes())
            .map_err(|e| SageError::other(format!("Failed to write footer length: {e}")))?;

        writer
            .write_all(&footer_bytes)
            .map_err(|e| SageError::other(format!("Failed to write footer: {e}")))?;

        Ok(())
    }

    /// Get dimension from builder (placeholder).
    fn get_dimension(&self, _builder: &dyn VectorIndexBuilder) -> usize {
        128 // Placeholder
    }

    /// Get vector count from builder (placeholder).
    fn get_vector_count(&self, _builder: &dyn VectorIndexBuilder) -> usize {
        0 // Placeholder
    }

    /// Get index type from builder (placeholder).
    fn get_index_type(&self, _builder: &dyn VectorIndexBuilder) -> String {
        "HNSW".to_string() // Placeholder
    }

    /// Get the output path.
    pub fn output_path(&self) -> &Path {
        &self.output_path
    }

    /// Get the writer configuration.
    pub fn config(&self) -> &VectorWriterConfig {
        &self.config
    }
}

/// Index file header.
#[derive(Debug, Serialize, Deserialize, Encode, Decode)]
struct IndexHeader {
    magic: Vec<u8>,
    version: u32,
    dimension: usize,
    vector_count: usize,
    index_type: String,
    compressed: bool,
    timestamp: i64, // Unix timestamp in seconds
}

/// Index file footer.
#[derive(Debug, Serialize, Deserialize, Encode, Decode)]
struct IndexFooter {
    checksum: u64,
    build_stats: BuildStats,
}

/// Statistics from index building.
#[derive(Debug, Serialize, Deserialize, Encode, Decode)]
struct BuildStats {
    build_time_ms: u64,
    memory_usage: usize,
    optimization_applied: bool,
}
