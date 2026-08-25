//! Prefixed storage namespace wrapper.
//!
//! This module provides a storage wrapper that prefixes all file names with a
//! configurable namespace, enabling multiple logical stores to share a single
//! physical storage backend. Each [`PrefixedStorage`] instance transparently
//! maps file names by prepending `<prefix>/` to every operation, so that
//! different subsystems (e.g. separate index segments) can coexist in one
//! underlying [`Storage`] without name collisions.

use std::sync::Arc;

use crate::error::Result;
use crate::storage::{FileMetadata, Storage, StorageInput, StorageOutput};

/// Storage facade that transparently prefixes all file names.
///
/// `PrefixedStorage` wraps any [`Storage`] implementation and maps every
/// file-name argument through a configurable prefix, effectively creating
/// an isolated namespace within the underlying store. This is useful when
/// multiple index segments or subsystems need to share a single storage
/// directory without risk of name collisions.
///
/// Leading and trailing `/` characters are stripped from the prefix during
/// construction, and an empty prefix results in a pass-through (no
/// transformation).
#[derive(Debug)]
pub struct PrefixedStorage {
    /// The namespace prefix prepended to all file names.
    prefix: String,
    /// The underlying storage backend that all operations are delegated to.
    inner: Arc<dyn Storage>,
}

impl PrefixedStorage {
    /// Create a new prefixed storage namespace.
    ///
    /// The prefix is trimmed of leading and trailing `/` characters.
    /// An empty prefix makes this wrapper a transparent pass-through.
    ///
    /// # Arguments
    ///
    /// * `prefix` - The namespace prefix to prepend to all file names.
    /// * `inner` - The underlying [`Storage`] backend to delegate to.
    ///
    /// # Returns
    ///
    /// A new `PrefixedStorage` wrapping the given backend.
    pub fn new(prefix: impl Into<String>, inner: Arc<dyn Storage>) -> Self {
        let prefix = prefix.into();
        let prefix = prefix.trim_matches('/').to_string();
        Self { prefix, inner }
    }

    /// Map a logical file name to its prefixed form in the underlying storage.
    ///
    /// # Arguments
    ///
    /// * `name` - The logical file name within this namespace.
    ///
    /// # Returns
    ///
    /// The prefixed file name as `<prefix>/<name>`, or `name` unchanged when
    /// the prefix is empty.
    fn map_name(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else if name.is_empty() {
            self.prefix.clone()
        } else {
            format!("{}/{}", self.prefix, name)
        }
    }

    /// Strip the namespace prefix from a fully-qualified file name.
    ///
    /// # Arguments
    ///
    /// * `name` - The fully-qualified file name from the underlying storage.
    ///
    /// # Returns
    ///
    /// The logical file name with the prefix removed, or `None` if the name
    /// does not belong to this namespace.
    fn strip_prefix(&self, name: &str) -> Option<String> {
        if self.prefix.is_empty() {
            return Some(name.to_string());
        }
        if name == self.prefix {
            return Some(String::new());
        }
        let prefix = format!("{}/", self.prefix);
        if name.starts_with(&prefix) {
            Some(name[prefix.len()..].to_string())
        } else {
            None
        }
    }
}

impl Storage for PrefixedStorage {
    fn loading_mode(&self) -> crate::storage::LoadingMode {
        // Delegate (#554): the trait default is `Eager`, which misclassified
        // an mmap-backed FileStorage behind the engine's prefix wrapper —
        // and consumers use this to pick between paging-friendly lazy reads
        // and one-shot buffering.
        self.inner.loading_mode()
    }

    fn open_input(&self, name: &str) -> Result<Box<dyn StorageInput>> {
        self.inner.open_input(&self.map_name(name))
    }

    fn create_output(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.inner.create_output(&self.map_name(name))
    }

    fn create_output_append(&self, name: &str) -> Result<Box<dyn StorageOutput>> {
        self.inner.create_output_append(&self.map_name(name))
    }

    fn file_exists(&self, name: &str) -> bool {
        self.inner.file_exists(&self.map_name(name))
    }

    fn delete_file(&self, name: &str) -> Result<()> {
        self.inner.delete_file(&self.map_name(name))
    }

    fn list_files(&self) -> Result<Vec<String>> {
        let prefix = if self.prefix.is_empty() {
            String::new()
        } else {
            format!("{}/", self.prefix)
        };
        let files = self.inner.list_files()?;
        Ok(files
            .into_iter()
            .filter_map(|entry| {
                if prefix.is_empty() {
                    Some(entry)
                } else if entry == self.prefix {
                    Some(String::new())
                } else if entry.starts_with(&prefix) {
                    Some(entry[prefix.len()..].to_string())
                } else {
                    None
                }
            })
            .collect())
    }

    fn file_size(&self, name: &str) -> Result<u64> {
        self.inner.file_size(&self.map_name(name))
    }

    fn metadata(&self, name: &str) -> Result<FileMetadata> {
        self.inner.metadata(&self.map_name(name))
    }

    fn rename_file(&self, old_name: &str, new_name: &str) -> Result<()> {
        self.inner
            .rename_file(&self.map_name(old_name), &self.map_name(new_name))
    }

    fn create_temp_output(&self, prefix: &str) -> Result<(String, Box<dyn StorageOutput>)> {
        let mapped_prefix = self.map_name(prefix);
        let (full_name, handle) = self.inner.create_temp_output(&mapped_prefix)?;
        let relative = if self.prefix.is_empty() {
            full_name.clone()
        } else {
            self.strip_prefix(&full_name).unwrap_or(full_name.clone())
        };
        Ok((relative, handle))
    }

    fn sync(&self) -> Result<()> {
        self.inner.sync()
    }

    fn close(&mut self) -> Result<()> {
        // Namespaced views do not own the underlying storage, so no-op.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::memory::{MemoryStorage, MemoryStorageConfig};

    #[test]
    fn isolates_file_names() {
        let base: Arc<dyn Storage> = Arc::new(MemoryStorage::new(MemoryStorageConfig::default()));
        let prefixed = PrefixedStorage::new("ns", base.clone());

        {
            let mut output = prefixed.create_output("foo.bin").unwrap();
            use std::io::Write;
            output.write_all(b"data").unwrap();
            output.close().unwrap();
        }

        assert!(base.file_exists("ns/foo.bin"));
        assert!(!base.file_exists("foo.bin"));

        let files = prefixed.list_files().unwrap();
        assert_eq!(files, vec!["foo.bin".to_string()]);
    }
}
