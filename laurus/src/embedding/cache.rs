//! Query-time embedding cache (Issue [#678](https://github.com/mosuka/laurus/issues/678)).
//!
//! Vector / hybrid search embeds the query payload before searching. The
//! same payload (same field, same embedder) produces the same vector, so
//! re-embedding identical queries wastes work — model inference for local
//! embedders, or a network round trip for remote ones.
//!
//! [`EmbeddingCache`] is a small LRU keyed by [`EmbeddingCacheKey`]. It is
//! shared (via `Arc`) between the two query-time embedding sites:
//!
//! - [`crate::engine::Engine::search`]'s direct `Payloads` resolution, and
//! - [`crate::vector::query::parser::VectorQueryParser`]'s DSL resolution.
//!
//! [`embed_with_cache`] is the single helper both call sites use, so the
//! cache-lookup / embed / store logic lives in one place.

use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;

use crate::embedding::embedder::{EmbedInput, Embedder};
use crate::embedding::per_field::PerFieldEmbedder;
use crate::error::Result;
use crate::vector::core::vector::Vector;

/// Cache key for a query-time embedding.
///
/// The payload is stored as a 64-bit hash (see [`hash_embed_input`]) rather
/// than its full bytes to keep the cache compact. `field` distinguishes
/// per-field embedders and `embedder` distinguishes different embedder
/// configurations, so an entry is only reused when the same embedder embeds
/// the same payload for the same field.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct EmbeddingCacheKey {
    field: String,
    embedder: String,
    payload_hash: u64,
}

/// Hash an [`EmbedInput`] for use in an [`EmbeddingCacheKey`].
///
/// A leading discriminant byte keeps `Text("x")` and `Bytes(b"x", None)`
/// from colliding.
fn hash_embed_input(input: &EmbedInput<'_>) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    match input {
        EmbedInput::Text(t) => {
            0u8.hash(&mut hasher);
            t.hash(&mut hasher);
        }
        EmbedInput::Bytes(b, mime) => {
            1u8.hash(&mut hasher);
            b.hash(&mut hasher);
            mime.hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// A bounded LRU cache of query embeddings.
///
/// A `Mutex` (not `RwLock`) is required because [`LruCache::get`] takes
/// `&mut self` to update recency. Critical sections are tiny (a hash-map
/// probe); the embedder call always runs outside the lock.
#[derive(Debug)]
pub struct EmbeddingCache {
    inner: Mutex<LruCache<EmbeddingCacheKey, Vector>>,
}

impl EmbeddingCache {
    /// Create a cache holding up to `capacity` entries.
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            inner: Mutex::new(LruCache::new(capacity)),
        }
    }

    /// Look up a cached embedding, bumping its recency on a hit.
    fn get(&self, key: &EmbeddingCacheKey) -> Option<Vector> {
        self.inner.lock().get(key).cloned()
    }

    /// Insert an embedding, evicting the least-recently-used entry when full.
    fn put(&self, key: EmbeddingCacheKey, value: Vector) {
        self.inner.lock().put(key, value);
    }
}

/// Embed `input` for `field`, consulting `cache` first when it is `Some`.
///
/// On a cache hit the stored [`Vector`] is cloned (an `Arc` bump) and
/// returned without invoking the embedder. On a miss the embedder runs —
/// the per-field embedder when `embedder` is a [`PerFieldEmbedder`],
/// otherwise the default — and the result is stored before being returned.
/// When `cache` is `None` the embedder always runs.
///
/// This is the single embedding entry point shared by the engine's direct
/// payload path and the vector query parser's DSL path (Issue #678).
///
/// # Arguments
///
/// * `cache` - The shared embedding cache, or `None` to bypass caching.
/// * `embedder` - The configured embedder.
/// * `field` - The vector field the payload targets.
/// * `input` - The prepared embedding input.
pub async fn embed_with_cache(
    cache: Option<&Arc<EmbeddingCache>>,
    embedder: &Arc<dyn Embedder>,
    field: &str,
    input: &EmbedInput<'_>,
) -> Result<Vector> {
    let key = cache.map(|_| EmbeddingCacheKey {
        field: field.to_string(),
        embedder: embedder.name().to_string(),
        payload_hash: hash_embed_input(input),
    });

    if let (Some(cache), Some(key)) = (cache, &key)
        && let Some(vector) = cache.get(key)
    {
        return Ok(vector);
    }

    let vector = if let Some(pf) = embedder.as_any().downcast_ref::<PerFieldEmbedder>() {
        pf.embed_field(field, input).await?
    } else {
        embedder.embed(input).await?
    };

    if let (Some(cache), Some(key)) = (cache, key) {
        cache.put(key, vector.clone());
    }

    Ok(vector)
}
