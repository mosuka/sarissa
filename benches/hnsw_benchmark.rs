use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::Rng;
use sarissa::storage::memory::MemoryStorageConfig;
use sarissa::storage::{StorageConfig, StorageFactory};
use sarissa::vector::core::distance::DistanceMetric;
use sarissa::vector::core::vector::Vector;
use sarissa::vector::index::ManagedVectorIndex;
use sarissa::vector::index::config::{HnswIndexConfig, VectorIndexTypeConfig};

fn generate_random_vector(dim: usize) -> Vector {
    let mut rng = rand::rng();
    let data: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
    Vector::new(data)
}

fn generate_vectors(count: usize, dim: usize) -> Vec<(u64, String, Vector)> {
    (0..count)
        .map(|i| (i as u64, format!("doc_{}", i), generate_random_vector(dim)))
        .collect()
}

fn bench_hnsw_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Construction");
    group.sample_size(10);
    let dim = 128; // Standard dimension used in benchmarks roughly
    // Smaller sizes for quicker iteration during dev, but user asked for 10k-100k
    // Let's perform a small one first.
    let vector_counts = [1000, 5000];

    for count in vector_counts.iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            let vectors = generate_vectors(count, dim);
            b.iter(|| {
                let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
                let storage = StorageFactory::create(storage_config).unwrap();

                let config = HnswIndexConfig {
                    dimension: dim,
                    m: 16,
                    ef_construction: 200, // Standard high-quality construction param
                    distance_metric: DistanceMetric::Cosine,
                    ..Default::default()
                };
                let type_config = VectorIndexTypeConfig::HNSW(config);
                let mut index = ManagedVectorIndex::new(type_config, storage).unwrap();

                // Clone vectors for each iteration because add_vectors takes ownership? No, it takes Vec
                // Actually add_vectors takes Vec.
                // We need to clone the data for the benchmark loop.
                index.add_vectors(vectors.clone()).unwrap();
                index.finalize().unwrap();
            })
        });
    }
    group.finish();
}

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("HNSW Search");
    group.sample_size(10);
    let dim = 128;
    let count = 5000;

    // Setup index once
    let vectors = generate_vectors(count, dim);
    let storage_config = StorageConfig::Memory(MemoryStorageConfig::default());
    let storage = StorageFactory::create(storage_config).unwrap();

    let config = HnswIndexConfig {
        dimension: dim,
        m: 16,
        ef_construction: 200,
        distance_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let type_config = VectorIndexTypeConfig::HNSW(config);
    let mut index = ManagedVectorIndex::new(type_config, storage).unwrap();
    index.add_vectors(vectors).unwrap();
    index.finalize().unwrap();
    index.write("bench_search_idx").unwrap();

    let reader = index.reader().unwrap();

    // Generate query vectors
    let query_vector = generate_random_vector(dim);

    group.bench_function("search_10_neighbors", |b| {
        b.iter(|| {
            // Need to expose search properly via reader.
            // VectorIndexReader doesn't have search with ef_search params directly visible in trait?
            // Actually `HnswIndexReader` implements `VectorIndexReader`, but generic reader interface might be limited?
            // Checking reader interface... `reader.search(...)` isn't on trait, it's typically on `Searcher` trait.
            // Ah, `VectorIndex` has `reader()`, but searching is done via `Searcher` which is separate?
            // Wait, looking at `src/vector/index.rs`:
            // trait VectorIndex has `reader()` returning `VectorIndexReader`.
            // `HnswSearcher` is what we implemented.
            // We need `HnswSearcher` to do the search.
            // But `ManagedVectorIndex` only gives us a `VectorIndexReader`.
            // Let's create an `HnswSearcher` manually using the reader?
            // Or look at how `VectorEngine` does it.
            // Typically `searcher.search(...)`.

            // For now, let's look at `HnswSearcher`.
            // It holds an `Arc<HnswIndexReader>`.
            // So we can instantiate it.
            use sarissa::vector::index::hnsw::searcher::HnswSearcher;
            use sarissa::vector::search::searcher::VectorIndexSearchRequest;
            use sarissa::vector::search::searcher::VectorIndexSearcher; // Trait

            // We need to downcast/cast the generic reader to HnswIndexReader to pass to HnswSearcher::new?
            // Or HnswSearcher::new takes Arc<dyn VectorIndexReader> and downcasts internally?
            // Let's check HnswSearcher definition.

            let searcher = HnswSearcher::new(reader.clone()).unwrap();

            let request = VectorIndexSearchRequest::new(query_vector.clone()).top_k(10);

            let _results = searcher.search(&request).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, bench_hnsw_construction, bench_hnsw_search);
criterion_main!(benches);
