//! Document CRUD endpoints.

use axum::Json;
use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use super::GatewayState;
use super::convert;
use super::error::{BadRequest, GatewayError};
use crate::proto::laurus::v1;

/// `PUT /v1/documents/:id` — Inserts or replaces a document.
pub async fn put_document(
    State(mut state): State<GatewayState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, Response> {
    let document = convert::json_to_proto_document(
        body.get("document")
            .ok_or_else(|| BadRequest("missing \"document\" key".to_string()).into_response())?,
    )
    .map_err(|e| BadRequest(e).into_response())?;

    state
        .document_client
        .put_document(v1::PutDocumentRequest {
            id,
            document: Some(document),
        })
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    Ok(Json(json!({})))
}

/// `POST /v1/documents/:id` — Adds a document as a chunk.
pub async fn add_document(
    State(mut state): State<GatewayState>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, Response> {
    let document = convert::json_to_proto_document(
        body.get("document")
            .ok_or_else(|| BadRequest("missing \"document\" key".to_string()).into_response())?,
    )
    .map_err(|e| BadRequest(e).into_response())?;

    state
        .document_client
        .add_document(v1::AddDocumentRequest {
            id,
            document: Some(document),
        })
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    Ok(Json(json!({})))
}

/// `GET /v1/documents/:id` — Retrieves documents with the specified ID.
pub async fn get_documents(
    State(mut state): State<GatewayState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, Response> {
    let response = state
        .document_client
        .get_documents(v1::GetDocumentsRequest { id })
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    let inner = response.into_inner();
    let documents: Vec<Value> = inner
        .documents
        .iter()
        .map(convert::proto_document_to_json)
        .collect();

    Ok(Json(json!({ "documents": documents })))
}

/// `DELETE /v1/documents/:id` — Deletes documents with the specified ID.
pub async fn delete_documents(
    State(mut state): State<GatewayState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, Response> {
    state
        .document_client
        .delete_documents(v1::DeleteDocumentsRequest { id })
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    Ok(Json(json!({})))
}

/// `POST /v1/commit` — Persists pending changes.
pub async fn commit(State(mut state): State<GatewayState>) -> Result<Json<Value>, Response> {
    state
        .document_client
        .commit(v1::CommitRequest {})
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    Ok(Json(json!({})))
}

/// `POST /v1/flush_wal` — Forces buffered WAL records durable without a full
/// commit (group-commit on-demand barrier; near no-op under per-record sync).
pub async fn flush_wal(State(mut state): State<GatewayState>) -> Result<Json<Value>, Response> {
    state
        .document_client
        .flush_wal(v1::FlushWalRequest {})
        .await
        .map_err(|s| GatewayError(s).into_response())?;

    Ok(Json(json!({})))
}
