//! `laurus-server` provides a gRPC server and an optional HTTP gateway (gRPC-Gateway)
//! for the laurus search engine.
//!
//! The crate exposes four gRPC services:
//!
//! * **HealthService** – health-check endpoint.
//! * **IndexService** – index creation and schema management.
//! * **DocumentService** – document CRUD and commit operations.
//! * **SearchService** – lexical, vector, and hybrid search.
//!
//! When `http_port` is configured, an HTTP/JSON gateway is started alongside the gRPC server
//! so that clients can use either protocol.

pub mod config;
mod context;
pub mod convert;
pub mod gateway;
pub mod server;
pub mod service;

/// Generated protobuf/gRPC code.
pub mod proto {
    pub mod laurus {
        pub mod v1 {
            tonic::include_proto!("laurus.v1");
        }
    }
}
