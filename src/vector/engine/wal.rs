//! VectorEngine WAL（Write-Ahead Log）関連の型定義
//!
//! このモジュールは WAL 管理、WAL レコード、WAL ペイロードを提供する。

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::vector::core::document::DocumentVector;

#[cfg(test)]
pub const WAL_COMPACTION_THRESHOLD: usize = 4;
#[cfg(not(test))]
pub const WAL_COMPACTION_THRESHOLD: usize = 64;

pub type SeqNumber = u64;

#[derive(Debug, Default)]
pub struct VectorWal {
    records: Mutex<Vec<WalRecord>>,
    next_seq: AtomicU64,
}

impl VectorWal {
    pub fn from_records(records: Vec<WalRecord>) -> Self {
        let max_seq = records.iter().map(|record| record.seq).max().unwrap_or(0);
        Self {
            records: Mutex::new(records),
            next_seq: AtomicU64::new(max_seq),
        }
    }

    pub fn records(&self) -> Vec<WalRecord> {
        self.records.lock().clone()
    }

    pub fn len(&self) -> usize {
        self.records.lock().len()
    }

    pub fn last_seq(&self) -> SeqNumber {
        self.next_seq.load(Ordering::SeqCst)
    }

    pub fn append(&self, payload: WalPayload) -> Result<SeqNumber> {
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst) + 1;
        let record = WalRecord { seq, payload };
        self.records.lock().push(record);
        Ok(seq)
    }

    pub fn replay<F: FnMut(&WalRecord)>(&self, from: SeqNumber, mut handler: F) {
        for record in self.records.lock().iter() {
            if record.seq >= from {
                handler(record);
            }
        }
    }

    pub fn replace_records(&self, records: Vec<WalRecord>) {
        let max_seq = records.iter().map(|record| record.seq).max().unwrap_or(0);
        let mut guard = self.records.lock();
        *guard = records;
        self.next_seq.store(max_seq, Ordering::SeqCst);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalRecord {
    pub seq: SeqNumber,
    pub payload: WalPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalPayload {
    Upsert {
        doc_id: u64,
        document: DocumentVector,
    },
    Delete {
        doc_id: u64,
    },
}
