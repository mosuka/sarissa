//! HTTP Gateway module.
//!
//! Provides HTTP/JSON endpoints that act as a proxy to the gRPC server.
//! `User Request (HTTP/JSON) → gRPC Gateway (axum) → gRPC Server (tonic) → Engine`

mod convert;
mod document;
mod error;
mod health;
mod index;
mod search;

use axum::Router;
use axum::routing::{delete, get, post, put};
use tonic::transport::Channel;

use crate::proto::laurus::v1::document_service_client::DocumentServiceClient;
use crate::proto::laurus::v1::health_service_client::HealthServiceClient;
use crate::proto::laurus::v1::index_service_client::IndexServiceClient;
use crate::proto::laurus::v1::search_service_client::SearchServiceClient;

/// Shared state for the Gateway. Holds each gRPC client.
#[derive(Clone)]
pub struct GatewayState {
    health_client: HealthServiceClient<Channel>,
    index_client: IndexServiceClient<Channel>,
    document_client: DocumentServiceClient<Channel>,
    search_client: SearchServiceClient<Channel>,
}

impl GatewayState {
    /// Creates a `GatewayState` from a gRPC channel.
    pub fn new(channel: Channel) -> Self {
        Self {
            health_client: HealthServiceClient::new(channel.clone()),
            index_client: IndexServiceClient::new(channel.clone()),
            document_client: DocumentServiceClient::new(channel.clone()),
            search_client: SearchServiceClient::new(channel),
        }
    }
}

/// Creates the axum router for the Gateway.
pub fn create_router(state: GatewayState) -> Router {
    Router::new()
        .route("/v1/health", get(health::check))
        .route("/v1/index", post(index::create).get(index::get_index))
        .route("/v1/schema", get(index::get_schema))
        .route("/v1/schema/fields", post(index::add_field))
        .route("/v1/schema/fields/{name}", delete(index::delete_field))
        .route(
            "/v1/documents/{id}",
            put(document::put_document)
                .post(document::add_document)
                .get(document::get_documents)
                .delete(document::delete_documents),
        )
        .route("/v1/documents:bulk", post(document::bulk_documents))
        .route("/v1/commit", post(document::commit))
        .route("/v1/flush_wal", post(document::flush_wal))
        .route("/v1/search", post(search::search))
        .route("/v1/search/stream", post(search::search_stream))
        .route("/v1/search/batch", post(search::search_batch))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The router must build without panicking — in particular the literal
    /// `:bulk` suffix in `/v1/documents:bulk` must be accepted by axum 0.8's
    /// `{param}` syntax (a bare colon is a plain path character there).
    #[tokio::test]
    async fn router_builds_with_bulk_route() {
        let channel = tonic::transport::Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        let _router = create_router(GatewayState::new(channel));
    }
}
