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
//! [`embed_with_cache`] is the single-input helper and
//! [`embed_batch_with_cache`] its batch counterpart (Issue #671); both call
//! sites embed all of a query's payloads through the batch helper so the
//! cache-lookup / embed / store logic lives in one place and a batch-capable
//! embedder pays one round trip instead of one per payload.

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

/// Batch variant of [`embed_with_cache`] (Issue #671).
///
/// Embeds many `(field, input)` pairs together so an embedder with a real
/// batch API (e.g. a remote model that serves many inputs per request, like
/// [`OpenAIEmbedder`](crate::embedding::openai_embedder::OpenAIEmbedder)) pays
/// one round trip instead of one per payload. Replaces the per-payload
/// `embed_with_cache` loops in the engine's `Payloads` fallback and the
/// vector query parser.
///
/// Cache semantics match [`embed_with_cache`] exactly: each item is looked up
/// by `(field, embedder.name(), hash(input))`, only misses are embedded, and
/// fresh embeddings are stored. Per-field routing is preserved: when
/// `embedder` is a [`PerFieldEmbedder`] the misses are grouped by field and
/// each group is dispatched to that field's embedder before calling
/// [`Embedder::embed_batch`]; otherwise all misses are embedded in one call.
///
/// # Arguments
///
/// * `cache` - The shared embedding cache, or `None` to bypass caching.
/// * `embedder` - The configured embedder.
/// * `items` - The `(field, input)` pairs to embed.
///
/// # Returns
///
/// One [`Vector`] per item, in the same order as `items`.
///
/// # Errors
///
/// Returns an error if any underlying [`Embedder::embed_batch`] call fails, or
/// (defensively) if an embedder returns fewer vectors than inputs.
pub async fn embed_batch_with_cache(
    cache: Option<&Arc<EmbeddingCache>>,
    embedder: &Arc<dyn Embedder>,
    items: &[(String, EmbedInput<'_>)],
) -> Result<Vec<Vector>> {
    // Per-item cache keys (`None` when caching is disabled). Computed once and
    // reused for both the lookup and the store so the keying matches
    // `embed_with_cache`.
    let keys: Vec<Option<EmbeddingCacheKey>> = items
        .iter()
        .map(|(field, input)| {
            cache.map(|_| EmbeddingCacheKey {
                field: field.clone(),
                embedder: embedder.name().to_string(),
                payload_hash: hash_embed_input(input),
            })
        })
        .collect();

    // Fill cache hits immediately; collect the indices that still need work.
    let mut out: Vec<Option<Vector>> = vec![None; items.len()];
    let mut misses: Vec<usize> = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        if let (Some(cache), Some(key)) = (cache, key)
            && let Some(vector) = cache.get(key)
        {
            out[i] = Some(vector);
        } else {
            misses.push(i);
        }
    }

    // Group misses so each batch call goes to the right embedder. Field only
    // affects routing for a `PerFieldEmbedder`; a field-agnostic embedder
    // takes every miss in a single batch. Grouping is order-stable so the
    // first occurrence of each field keeps its position.
    let per_field = embedder.as_any().downcast_ref::<PerFieldEmbedder>();
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    if per_field.is_some() {
        for &i in &misses {
            let field = &items[i].0;
            if let Some((_, idxs)) = groups.iter_mut().find(|(f, _)| f == field) {
                idxs.push(i);
            } else {
                groups.push((field.clone(), vec![i]));
            }
        }
    } else if !misses.is_empty() {
        groups.push((String::new(), misses));
    }

    for (field, idxs) in groups {
        let inputs: Vec<EmbedInput<'_>> = idxs.iter().map(|&i| items[i].1.clone()).collect();
        let group_embedder = match per_field {
            Some(pf) => pf.get_embedder(&field),
            None => embedder.clone(),
        };
        let vectors = group_embedder.embed_batch(&inputs).await?;
        for (&i, vector) in idxs.iter().zip(vectors) {
            if let (Some(cache), Some(key)) = (cache, &keys[i]) {
                cache.put(key.clone(), vector.clone());
            }
            out[i] = Some(vector);
        }
    }

    // Every slot is filled by construction; the error guards against an
    // embedder returning fewer vectors than inputs rather than panicking.
    out.into_iter()
        .map(|v| {
            v.ok_or_else(|| {
                crate::error::LaurusError::internal(
                    "embed_batch_with_cache: embedder returned fewer vectors than inputs",
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::embedder::EmbedInputType;
    use async_trait::async_trait;
    use std::any::Any;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Encode an input as a single deterministic scalar so a returned vector
    /// can be traced back to the input that produced it (order assertions).
    fn seed(input: &EmbedInput<'_>) -> f32 {
        match input {
            EmbedInput::Text(t) => t.bytes().map(|b| b as u32).sum::<u32>() as f32,
            EmbedInput::Bytes(b, _) => b.iter().map(|x| *x as u32).sum::<u32>() as f32,
        }
    }

    /// Embedder that counts `embed` / `embed_batch` calls and how many inputs
    /// the batch path saw, so tests can assert batching and routing.
    #[derive(Debug)]
    struct CountingEmbedder {
        name: String,
        embed_calls: AtomicUsize,
        embed_batch_calls: AtomicUsize,
        inputs_seen: AtomicUsize,
    }

    impl CountingEmbedder {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                embed_calls: AtomicUsize::new(0),
                embed_batch_calls: AtomicUsize::new(0),
                inputs_seen: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl Embedder for CountingEmbedder {
        async fn embed(&self, input: &EmbedInput<'_>) -> Result<Vector> {
            self.embed_calls.fetch_add(1, Ordering::SeqCst);
            Ok(Vector::new(vec![seed(input)]))
        }
        async fn embed_batch(&self, inputs: &[EmbedInput<'_>]) -> Result<Vec<Vector>> {
            self.embed_batch_calls.fetch_add(1, Ordering::SeqCst);
            self.inputs_seen.fetch_add(inputs.len(), Ordering::SeqCst);
            Ok(inputs.iter().map(|i| Vector::new(vec![seed(i)])).collect())
        }
        fn supported_input_types(&self) -> Vec<EmbedInputType> {
            vec![EmbedInputType::Text]
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn cache(cap: usize) -> Arc<EmbeddingCache> {
        Arc::new(EmbeddingCache::new(NonZeroUsize::new(cap).unwrap()))
    }

    fn item(field: &str, text: &'static str) -> (String, EmbedInput<'static>) {
        (field.to_string(), EmbedInput::Text(text))
    }

    /// One `embed_batch` call replaces the per-item `embed` loop, and results
    /// come back in input order.
    #[tokio::test]
    async fn batches_in_one_call_and_preserves_order() {
        let counter = Arc::new(CountingEmbedder::new("m"));
        let embedder: Arc<dyn Embedder> = counter.clone();
        let items = [item("f", "a"), item("f", "bb"), item("f", "ccc")];

        let got = embed_batch_with_cache(None, &embedder, &items)
            .await
            .unwrap();

        assert_eq!(counter.embed_batch_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counter.embed_calls.load(Ordering::SeqCst), 0);
        let expected: Vec<f32> = items.iter().map(|(_, i)| seed(i)).collect();
        let actual: Vec<f32> = got.iter().map(|v| v.data[0]).collect();
        assert_eq!(actual, expected, "results must be in input order");
    }

    /// Cached items are not re-embedded; only fresh items reach `embed_batch`.
    #[tokio::test]
    async fn reuses_cached_and_only_embeds_misses() {
        let counter = Arc::new(CountingEmbedder::new("m"));
        let embedder: Arc<dyn Embedder> = counter.clone();
        let cache = cache(16);

        let first = [item("f", "a"), item("f", "b")];
        embed_batch_with_cache(Some(&cache), &embedder, &first)
            .await
            .unwrap();
        assert_eq!(counter.embed_batch_calls.load(Ordering::SeqCst), 1);
        assert_eq!(counter.inputs_seen.load(Ordering::SeqCst), 2);

        // Identical second call: all hits, no new embedding.
        embed_batch_with_cache(Some(&cache), &embedder, &first)
            .await
            .unwrap();
        assert_eq!(counter.embed_batch_calls.load(Ordering::SeqCst), 1);

        // Mixed: "a" hits, "x" misses -> one more batch call over just "x".
        let mixed = [item("f", "a"), item("f", "x")];
        let got = embed_batch_with_cache(Some(&cache), &embedder, &mixed)
            .await
            .unwrap();
        assert_eq!(counter.embed_batch_calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            counter.inputs_seen.load(Ordering::SeqCst),
            3,
            "only the miss is embedded"
        );
        assert_eq!(got[0].data[0], seed(&EmbedInput::Text("a")));
        assert_eq!(got[1].data[0], seed(&EmbedInput::Text("x")));
    }

    /// A `PerFieldEmbedder` routes each field's misses to that field's embedder
    /// (grouped into one batch call per field), never the default.
    #[tokio::test]
    async fn routes_misses_per_field() {
        let title = Arc::new(CountingEmbedder::new("title"));
        let body = Arc::new(CountingEmbedder::new("body"));
        let default = Arc::new(CountingEmbedder::new("default"));

        let pf = PerFieldEmbedder::new(default.clone() as Arc<dyn Embedder>);
        pf.add_embedder("title", title.clone() as Arc<dyn Embedder>);
        pf.add_embedder("body", body.clone() as Arc<dyn Embedder>);
        let embedder: Arc<dyn Embedder> = Arc::new(pf);

        // Interleaved fields exercise the order-stable grouping.
        let items = [item("title", "a"), item("body", "b"), item("title", "c")];
        let got = embed_batch_with_cache(None, &embedder, &items)
            .await
            .unwrap();

        assert_eq!(title.embed_batch_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            title.inputs_seen.load(Ordering::SeqCst),
            2,
            "title got a + c"
        );
        assert_eq!(body.embed_batch_calls.load(Ordering::SeqCst), 1);
        assert_eq!(body.inputs_seen.load(Ordering::SeqCst), 1, "body got b");
        assert_eq!(default.embed_batch_calls.load(Ordering::SeqCst), 0);

        let expected: Vec<f32> = items.iter().map(|(_, i)| seed(i)).collect();
        let actual: Vec<f32> = got.iter().map(|v| v.data[0]).collect();
        assert_eq!(
            actual, expected,
            "results stay in original interleaved order"
        );
    }

    /// No items -> no embedding, empty result.
    #[tokio::test]
    async fn empty_items_is_a_noop() {
        let counter = Arc::new(CountingEmbedder::new("m"));
        let embedder: Arc<dyn Embedder> = counter.clone();
        let got = embed_batch_with_cache(None, &embedder, &[]).await.unwrap();
        assert!(got.is_empty());
        assert_eq!(counter.embed_batch_calls.load(Ordering::SeqCst), 0);
    }
}
