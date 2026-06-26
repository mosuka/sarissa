//! Document CRUD and commit gRPC service.
//!
//! Provides RPCs for inserting, updating, retrieving, and deleting documents,
//! as well as explicitly committing pending changes to durable storage.

use std::sync::Arc;

use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

use laurus::Engine;

use crate::convert::{document as doc_convert, error};
use crate::proto::laurus::v1::{
    AddDocumentRequest, AddDocumentResponse, CommitRequest, CommitResponse, DeleteDocumentsRequest,
    DeleteDocumentsResponse, FlushWalRequest, FlushWalResponse, GetDocumentsRequest,
    GetDocumentsResponse, PutDocumentRequest, PutDocumentResponse,
    document_service_server::DocumentService as DocumentServiceTrait,
};

/// gRPC DocumentService implementation.
#[derive(Clone)]
pub struct DocumentService {
    /// Shared, mutable reference to the current search engine instance.
    /// `None` when no index has been created yet.
    pub engine: Arc<RwLock<Option<Engine>>>,
}

impl DocumentService {
    #[allow(clippy::result_large_err)]
    fn get_engine_ref(guard: &Option<Engine>) -> Result<&Engine, Status> {
        guard
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("No index is open. Create an index first."))
    }
}

#[tonic::async_trait]
impl DocumentServiceTrait for DocumentService {
    /// Inserts or replaces a document with the given ID.
    async fn put_document(
        &self,
        request: Request<PutDocumentRequest>,
    ) -> Result<Response<PutDocumentResponse>, Status> {
        let req = request.into_inner();
        let doc = req
            .document
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("document is required"))?;
        let doc = doc_convert::from_proto(doc);

        let guard = self.engine.read().await;
        let engine = Self::get_engine_ref(&guard)?;
        engine
            .put_document(&req.id, doc)
            .await
            .map_err(error::to_status)?;

        Ok(Response::new(PutDocumentResponse {}))
    }

    /// Adds a new document. Fails if a document with the same ID already exists.
    async fn add_document(
        &self,
        request: Request<AddDocumentRequest>,
    ) -> Result<Response<AddDocumentResponse>, Status> {
        let req = request.into_inner();
        let doc = req
            .document
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("document is required"))?;
        let doc = doc_convert::from_proto(doc);

        let guard = self.engine.read().await;
        let engine = Self::get_engine_ref(&guard)?;
        engine
            .add_document(&req.id, doc)
            .await
            .map_err(error::to_status)?;

        Ok(Response::new(AddDocumentResponse {}))
    }

    /// Retrieves documents matching the given ID.
    async fn get_documents(
        &self,
        request: Request<GetDocumentsRequest>,
    ) -> Result<Response<GetDocumentsResponse>, Status> {
        let req = request.into_inner();

        let guard = self.engine.read().await;
        let engine = Self::get_engine_ref(&guard)?;
        let docs = engine
            .get_documents(&req.id)
            .await
            .map_err(error::to_status)?;

        let documents = docs.iter().map(doc_convert::to_proto).collect();
        Ok(Response::new(GetDocumentsResponse { documents }))
    }

    /// Deletes documents matching the given ID.
    async fn delete_documents(
        &self,
        request: Request<DeleteDocumentsRequest>,
    ) -> Result<Response<DeleteDocumentsResponse>, Status> {
        let req = request.into_inner();

        let guard = self.engine.read().await;
        let engine = Self::get_engine_ref(&guard)?;
        engine
            .delete_documents(&req.id)
            .await
            .map_err(error::to_status)?;

        Ok(Response::new(DeleteDocumentsResponse {}))
    }

    /// Flushes all pending changes to durable storage.
    async fn commit(
        &self,
        _request: Request<CommitRequest>,
    ) -> Result<Response<CommitResponse>, Status> {
        let guard = self.engine.read().await;
        let engine = Self::get_engine_ref(&guard)?;
        engine.commit().await.map_err(error::to_status)?;

        Ok(Response::new(CommitResponse {}))
    }

    /// Forces any buffered WAL records durable without a full commit.
    ///
    /// A near no-op under the default per-record sync policy; under the
    /// group-commit policy it flushes a partial batch on demand.
    async fn flush_wal(
        &self,
        _request: Request<FlushWalRequest>,
    ) -> Result<Response<FlushWalResponse>, Status> {
        let guard = self.engine.read().await;
        let engine = Self::get_engine_ref(&guard)?;
        engine.flush_wal().map_err(error::to_status)?;

        Ok(Response::new(FlushWalResponse {}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use laurus::storage::memory::MemoryStorage;
    use laurus::{Schema, Storage, WalSyncPolicy};

    async fn service_with_engine(policy: WalSyncPolicy) -> DocumentService {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new(Default::default()));
        let engine = Engine::builder(storage, Schema::default())
            .wal_sync_policy(policy)
            .build()
            .await
            .unwrap();
        DocumentService {
            engine: Arc::new(RwLock::new(Some(engine))),
        }
    }

    #[tokio::test]
    async fn flush_wal_fails_without_an_index() {
        let service = DocumentService {
            engine: Arc::new(RwLock::new(None)),
        };
        let status = service
            .flush_wal(Request::new(FlushWalRequest {}))
            .await
            .unwrap_err();
        assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    }

    #[tokio::test]
    async fn flush_wal_succeeds_under_per_record_policy() {
        let service = service_with_engine(WalSyncPolicy::PerRecord).await;
        service
            .flush_wal(Request::new(FlushWalRequest {}))
            .await
            .expect("flush_wal must succeed (near no-op) under per-record policy");
    }

    #[tokio::test]
    async fn flush_wal_succeeds_under_group_policy() {
        let service = service_with_engine(WalSyncPolicy::group_with_defaults()).await;
        service
            .flush_wal(Request::new(FlushWalRequest {}))
            .await
            .expect("flush_wal must act as a durability barrier under group policy");
    }
}
