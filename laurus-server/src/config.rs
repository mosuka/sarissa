//! Server configuration types deserialized from a TOML file.
//!
//! The top-level [`Config`] struct contains sections for the gRPC/HTTP server,
//! index storage, and logging. All sections have sensible defaults so that
//! a minimal (or even empty) TOML file produces a working configuration.

use laurus::{DEFAULT_GROUP_MAX_BYTES, DEFAULT_GROUP_MAX_RECORDS, WalSyncPolicy};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Top-level configuration loaded from a TOML file.
#[derive(Debug, Deserialize, Default)]
pub struct Config {
    /// Network settings for the gRPC server and the optional HTTP gateway.
    #[serde(default)]
    pub server: ServerConfig,
    /// Index storage settings (e.g. data directory path).
    #[serde(default)]
    pub index: IndexConfig,
}

/// Server network configuration.
#[derive(Debug, Deserialize)]
pub struct ServerConfig {
    /// Listen address for the gRPC server.
    #[serde(default = "default_host")]
    pub host: String,
    /// Listen port for the gRPC server.
    #[serde(default = "default_port")]
    pub port: u16,
    /// Listen port for the HTTP Gateway. The Gateway is started only when this is set.
    #[serde(default)]
    pub http_port: Option<u16>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            http_port: None,
        }
    }
}

/// Index storage settings.
#[derive(Debug, Deserialize)]
pub struct IndexConfig {
    /// Filesystem path where the index data (schema and store) is persisted.
    /// Defaults to `"./laurus_data"`.
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
    /// Write-ahead log durability settings. Defaults to per-record fsync.
    #[serde(default)]
    pub wal: WalConfig,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            wal: WalConfig::default(),
        }
    }
}

/// The WAL sync policy selector deserialized from the config file.
///
/// Mirrors the variants of [`laurus::WalSyncPolicy`] without their parameters;
/// the [`WalConfig`] group thresholds are applied when the policy is
/// [`SyncPolicyKind::Group`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SyncPolicyKind {
    /// Fsync after every record (default); maps to [`WalSyncPolicy::PerRecord`].
    #[default]
    PerRecord,
    /// Batch fsyncs; maps to [`WalSyncPolicy::Group`] with the [`WalConfig`]
    /// thresholds.
    Group,
}

/// Write-ahead log durability configuration.
///
/// Selects the [`laurus::WalSyncPolicy`] used for the index. The default is
/// [`SyncPolicyKind::PerRecord`], preserving per-record durability. When
/// `sync_policy = "group"`, the optional `group_*` thresholds tune the batch;
/// unset thresholds fall back to [`laurus::DEFAULT_GROUP_MAX_RECORDS`] /
/// [`laurus::DEFAULT_GROUP_MAX_BYTES`].
#[derive(Debug, Clone, Copy, Deserialize, Default)]
pub struct WalConfig {
    /// Which durability policy to use. Defaults to `per_record`.
    #[serde(default)]
    pub sync_policy: SyncPolicyKind,
    /// Flush once this many records accumulate (group policy only). Falls back
    /// to [`laurus::DEFAULT_GROUP_MAX_RECORDS`] when unset.
    #[serde(default)]
    pub group_max_records: Option<usize>,
    /// Flush once this many bytes accumulate (group policy only). Falls back to
    /// [`laurus::DEFAULT_GROUP_MAX_BYTES`] when unset.
    #[serde(default)]
    pub group_max_bytes: Option<usize>,
    /// Periodic flush interval in milliseconds (group policy only). When unset
    /// no background timer runs. Honored on native targets only.
    #[serde(default)]
    pub group_max_interval_ms: Option<u64>,
}

impl WalConfig {
    /// Convert this configuration into a [`laurus::WalSyncPolicy`].
    ///
    /// # Returns
    ///
    /// [`WalSyncPolicy::PerRecord`] when `sync_policy` is `per_record`, or
    /// [`WalSyncPolicy::Group`] with the configured thresholds (defaults applied
    /// for unset values) when `sync_policy` is `group`.
    pub fn to_policy(&self) -> WalSyncPolicy {
        match self.sync_policy {
            SyncPolicyKind::PerRecord => WalSyncPolicy::PerRecord,
            SyncPolicyKind::Group => WalSyncPolicy::Group {
                max_records: self.group_max_records.unwrap_or(DEFAULT_GROUP_MAX_RECORDS),
                max_bytes: self.group_max_bytes.unwrap_or(DEFAULT_GROUP_MAX_BYTES),
                max_interval: self.group_max_interval_ms.map(Duration::from_millis),
            },
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    50051
}

fn default_data_dir() -> PathBuf {
    PathBuf::from("./laurus_data")
}

impl Config {
    /// Load configuration from a TOML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Filesystem path to the TOML configuration file.
    ///
    /// # Returns
    ///
    /// A fully populated [`Config`] instance with defaults applied for any
    /// missing sections or fields.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if the TOML content
    /// cannot be deserialized into a [`Config`].
    pub fn from_file(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_wal_policy_is_per_record() {
        // An empty config (no [index.wal] section) must keep per-record fsync.
        let config: Config = toml::from_str("").unwrap();
        assert_eq!(config.index.wal.sync_policy, SyncPolicyKind::PerRecord);
        assert_eq!(config.index.wal.to_policy(), WalSyncPolicy::PerRecord);
    }

    #[test]
    fn group_policy_uses_defaults_when_thresholds_omitted() {
        let toml = r#"
            [index.wal]
            sync_policy = "group"
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.index.wal.sync_policy, SyncPolicyKind::Group);
        assert_eq!(
            config.index.wal.to_policy(),
            WalSyncPolicy::Group {
                max_records: DEFAULT_GROUP_MAX_RECORDS,
                max_bytes: DEFAULT_GROUP_MAX_BYTES,
                max_interval: None,
            }
        );
    }

    #[test]
    fn group_policy_honors_explicit_thresholds_and_interval() {
        let toml = r#"
            [index.wal]
            sync_policy = "group"
            group_max_records = 256
            group_max_bytes = 4096
            group_max_interval_ms = 500
        "#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(
            config.index.wal.to_policy(),
            WalSyncPolicy::Group {
                max_records: 256,
                max_bytes: 4096,
                max_interval: Some(Duration::from_millis(500)),
            }
        );
    }
}
