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
    AddDocumentRequest, AddDocumentResponse, AddDocumentsRequest, AddDocumentsResponse,
    CommitRequest, CommitResponse, DeleteDocumentsRequest, DeleteDocumentsResponse, DocumentEntry,
    FlushWalRequest, FlushWalResponse, GetDocumentsRequest, GetDocumentsResponse,
    PutDocumentRequest, PutDocumentResponse, PutDocumentsRequest, PutDocumentsResponse,
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

    /// Convert a batched request's entries into the engine's
    /// `(external_id, Document)` pairs, rejecting entries without a document.
    ///
    /// # Errors
    ///
    /// Returns `Status::invalid_argument` naming the offending position when
    /// an entry carries no document.
    #[allow(clippy::result_large_err)]
    fn entries_to_docs(
        entries: Vec<DocumentEntry>,
    ) -> Result<Vec<(String, laurus::Document)>, Status> {
        entries
            .into_iter()
            .enumerate()
            .map(|(index, entry)| {
                let doc = entry.document.as_ref().ok_or_else(|| {
                    Status::invalid_argument(format!(
                        "documents[{index}] (id '{}'): document is required",
                        entry.id
                    ))
                })?;
                Ok((entry.id, doc_convert::from_proto(doc)))
            })
            .collect()
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

    /// Batched upsert: applies the entries sequentially, in input order, with
    /// one WAL fsync for the whole batch (see `Engine::put_documents`).
    ///
    /// Fails fast at the first entry that cannot be applied — already-applied
    /// entries are not rolled back, and the returned status message carries
    /// the failing position, its id, and the applied count, so clients can
    /// retry the batch (or its suffix) idempotently.
    async fn put_documents(
        &self,
        request: Request<PutDocumentsRequest>,
    ) -> Result<Response<PutDocumentsResponse>, Status> {
        let req = request.into_inner();
        let docs = Self::entries_to_docs(req.documents)?;
        let applied = docs.len() as u32;

        let guard = self.engine.read().await;
        let engine = Self::get_engine_ref(&guard)?;
        engine.put_documents(docs).await.map_err(error::to_status)?;

        Ok(Response::new(PutDocumentsResponse { applied }))
    }

    /// Batched chunk append: like `put_documents` but never deletes existing
    /// documents, so a batch may repeat an ID to add multiple chunks.
    async fn add_documents(
        &self,
        request: Request<AddDocumentsRequest>,
    ) -> Result<Response<AddDocumentsResponse>, Status> {
        let req = request.into_inner();
        let docs = Self::entries_to_docs(req.documents)?;
        let applied = docs.len() as u32;

        let guard = self.engine.read().await;
        let engine = Self::get_engine_ref(&guard)?;
        engine.add_documents(docs).await.map_err(error::to_status)?;

        Ok(Response::new(AddDocumentsResponse { applied }))
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

    fn entry(id: &str, title: &str) -> DocumentEntry {
        let doc = laurus::Document::builder()
            .add_field("title", laurus::DataValue::Text(title.into()))
            .build();
        DocumentEntry {
            id: id.to_string(),
            document: Some(doc_convert::to_proto(&doc)),
        }
    }

    /// #865: PutDocuments applies every entry (dedup included) and reports
    /// the applied count; the docs are retrievable after a commit.
    #[tokio::test]
    async fn put_documents_applies_batch_and_reports_count() {
        let service = service_with_engine(WalSyncPolicy::PerRecord).await;

        let documents = vec![entry("a", "one"), entry("b", "two"), entry("a", "one-v2")];
        let resp = service
            .put_documents(Request::new(PutDocumentsRequest { documents }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(resp.applied, 3);

        service
            .commit(Request::new(CommitRequest {}))
            .await
            .unwrap();
        let docs = service
            .get_documents(Request::new(GetDocumentsRequest { id: "a".into() }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(
            docs.documents.len(),
            1,
            "the duplicate id within the batch must dedup (last wins)"
        );
    }

    /// #865: AddDocuments never dedups — repeated ids accumulate as chunks.
    #[tokio::test]
    async fn add_documents_accumulates_chunks() {
        let service = service_with_engine(WalSyncPolicy::PerRecord).await;

        let documents = vec![entry("doc", "chunk-0"), entry("doc", "chunk-1")];
        let resp = service
            .add_documents(Request::new(AddDocumentsRequest { documents }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(resp.applied, 2);

        service
            .commit(Request::new(CommitRequest {}))
            .await
            .unwrap();
        let docs = service
            .get_documents(Request::new(GetDocumentsRequest { id: "doc".into() }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(docs.documents.len(), 2);
    }

    /// #865: an empty batch succeeds with applied == 0; an entry without a
    /// document is rejected up front naming its position.
    #[tokio::test]
    async fn put_documents_validates_entries() {
        let service = service_with_engine(WalSyncPolicy::PerRecord).await;

        let resp = service
            .put_documents(Request::new(PutDocumentsRequest { documents: vec![] }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(resp.applied, 0);

        let documents = vec![
            entry("ok", "fine"),
            DocumentEntry {
                id: "broken".into(),
                document: None,
            },
        ];
        let status = service
            .put_documents(Request::new(PutDocumentsRequest { documents }))
            .await
            .unwrap_err();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(
            status.message().contains("documents[1]") && status.message().contains("'broken'"),
            "the error must name the offending entry: {}",
            status.message()
        );
    }
}
