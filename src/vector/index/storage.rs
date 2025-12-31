use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex};

use crate::error::{Result, SarissaError};
use crate::storage::StorageInput;
use crate::vector::core::vector::Vector;
use crate::vector::index::io::read_metadata;

/// Storage for vectors (in-memory or on-demand).
#[derive(Debug, Clone)]
pub enum VectorStorage {
    Owned(Arc<HashMap<(u64, String), Vector>>),
    OnDemand {
        input: Arc<Mutex<Box<dyn StorageInput>>>,
        offsets: Arc<HashMap<(u64, String), u64>>,
    },
}

impl VectorStorage {
    pub fn keys(&self) -> Vec<(u64, String)> {
        match self {
            VectorStorage::Owned(map) => map.keys().cloned().collect(),
            VectorStorage::OnDemand { offsets, .. } => offsets.keys().cloned().collect(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            VectorStorage::Owned(map) => map.len(),
            VectorStorage::OnDemand { offsets, .. } => offsets.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains_key(&self, key: &(u64, String)) -> bool {
        match self {
            VectorStorage::Owned(map) => map.contains_key(key),
            VectorStorage::OnDemand { offsets, .. } => offsets.contains_key(key),
        }
    }

    pub fn get(&self, key: &(u64, String), dimension: usize) -> Result<Option<Vector>> {
        match self {
            VectorStorage::Owned(map) => Ok(map.get(key).cloned()),
            VectorStorage::OnDemand { input, offsets } => {
                if let Some(&offset) = offsets.get(key) {
                    let mut input = input
                        .lock()
                        .map_err(|_| SarissaError::internal("Mutex poisoned".to_string()))?;

                    input
                        .seek(SeekFrom::Start(offset))
                        .map_err(|e| SarissaError::Io(e))?;

                    let metadata = read_metadata(&mut *input)?;
                    let mut values = vec![0.0f32; dimension];
                    for value in &mut values {
                        let mut value_buf = [0u8; 4];
                        input.read_exact(&mut value_buf)?;
                        *value = f32::from_le_bytes(value_buf);
                    }
                    Ok(Some(Vector::with_metadata(values, metadata)))
                } else {
                    Ok(None)
                }
            }
        }
    }
}
