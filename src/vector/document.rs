//! Vector ドキュメント関連型
//!
//! lexical::document と対称なモジュール構造を提供する。
//!
//! # 構成
//!
//! - [`document`] - ドキュメントレベルの型（DocumentVector, DocumentPayload）
//! - [`field`] - フィールドレベルの型（FieldVectors, FieldPayload, StoredVector）
//! - [`payload`] - ペイロードとベクトルタイプ（VectorType, PayloadSource, SegmentPayload）
//!
//! # 対称性
//!
//! | lexical::document | vector::document | 説明 |
//! |-------------------|------------------|------|
//! | document::Document | document::DocumentVector | ドキュメント本体 |
//! | document::field::Field | document::field::FieldVectors | フィールド定義 |
//! | document::field::FieldValue | document::field::StoredVector | フィールド値 |

pub mod document;
pub mod field;
pub mod payload;
