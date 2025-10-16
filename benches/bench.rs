//! Criterion benchmarks for Sage search engine.
//!
//! This module contains comprehensive benchmarks for all major components
//! of the Sage search engine, including:
//! - Text analysis and tokenization
//! - Vector similarity search (HNSW)
//! - Spell correction
//! - Parallel operations

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use sage::analysis::analyzer::analyzer::Analyzer;
use sage::analysis::analyzer::standard::StandardAnalyzer;
use sage::spelling::corrector::SpellingCorrector;
use sage::vector::{DistanceMetric, Vector};
use sage::vector_index::hnsw_builder::HnswIndexBuilder;
use sage::vector_index::{VectorIndexBuildConfig, VectorIndexBuilder};
use std::hint::black_box;

/// Generate test documents for benchmarking.
fn generate_test_documents(count: usize) -> Vec<String> {
    let words = vec![
        "search",
        "engine",
        "full",
        "text",
        "index",
        "query",
        "document",
        "field",
        "term",
        "phrase",
        "boolean",
        "vector",
        "similarity",
        "relevance",
        "score",
        "analysis",
        "tokenization",
        "stemming",
        "normalization",
        "clustering",
        "machine",
        "learning",
        "algorithm",
        "data",
        "structure",
        "performance",
        "optimization",
        "memory",
        "storage",
        "retrieval",
        "ranking",
        "filtering",
    ];

    let mut documents = Vec::with_capacity(count);
    for i in 0..count {
        let doc_length = 50 + (i % 100); // Variable length documents
        let mut doc_words = Vec::with_capacity(doc_length);

        for j in 0..doc_length {
            let word_idx = (i * 7 + j * 13) % words.len(); // Pseudo-random distribution
            doc_words.push(words[word_idx]);
        }

        documents.push(doc_words.join(" "));
    }

    documents
}

/// Generate test vectors for benchmarking.
fn generate_test_vectors(count: usize, dimension: usize) -> Vec<Vector> {
    let mut vectors = Vec::with_capacity(count);

    for i in 0..count {
        let mut data = Vec::with_capacity(dimension);
        for j in 0..dimension {
            // Create somewhat realistic vector data with patterns
            let value = ((i as f32 * 0.1 + j as f32 * 0.01).sin() * 0.5 + 0.5) * 2.0 - 1.0;
            data.push(value);
        }
        vectors.push(Vector::new(data));
    }

    vectors
}

/// Benchmark text analysis and tokenization.
fn bench_text_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("text_analysis");

    let analyzer = StandardAnalyzer::new().unwrap();
    let texts = generate_test_documents(1000);

    // Single document analysis
    group.bench_function("analyze_single_document", |b| {
        b.iter(|| {
            let result = analyzer.analyze(black_box(&texts[0]));
            black_box(result)
        })
    });

    // Batch document analysis
    group.throughput(Throughput::Elements(100));
    group.bench_function("analyze_batch_documents", |b| {
        b.iter(|| {
            for text in texts.iter().take(100) {
                let result = analyzer.analyze(black_box(text));
                let _ = black_box(result);
            }
        })
    });

    group.finish();
}

/// Benchmark vector operations.
fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");
    group.sample_size(20); // Reduce sample size for vector operations

    let dimension = 128;
    let vectors = generate_test_vectors(1000, dimension);

    // HNSW index construction
    group.throughput(Throughput::Elements(100));
    group.bench_function("hnsw_index_construction", |b| {
        b.iter_with_setup(
            || {
                HnswIndexBuilder::new(VectorIndexBuildConfig {
                    dimension,
                    ..Default::default()
                })
                .unwrap()
            },
            |mut builder| {
                let indexed_vectors: Vec<(u64, Vector)> = vectors
                    .iter()
                    .take(100)
                    .enumerate()
                    .map(|(i, v)| (i as u64, v.clone()))
                    .collect();
                let _ = builder.build(indexed_vectors);
                black_box(builder);
            },
        )
    });

    // Vector operations simplified (search requires full implementation)
    group.bench_function("vector_operations_basic", |b| {
        let query_vector = vectors[0].clone();

        b.iter(|| {
            // Basic vector operations for benchmarking
            let mut results = Vec::new();
            for (i, vector) in vectors.iter().take(50).enumerate() {
                let distance = DistanceMetric::Cosine
                    .distance(black_box(&query_vector.data), black_box(&vector.data))
                    .unwrap();
                results.push((i as u64, distance));
            }
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            black_box(results)
        })
    });

    // Vector distance calculations
    group.bench_function("cosine_distance_batch", |b| {
        let query = &vectors[0];
        let targets = &vectors[1..101]; // 100 vectors

        b.iter(|| {
            for target in targets {
                let distance = DistanceMetric::Cosine
                    .distance(black_box(&query.data), black_box(&target.data))
                    .unwrap();
                black_box(distance);
            }
        })
    });

    // Vector normalization
    group.throughput(Throughput::Elements(100));
    group.bench_function("vector_normalization", |b| {
        b.iter_with_setup(
            || vectors[0..100].to_vec(),
            |mut test_vectors| {
                for vector in &mut test_vectors {
                    vector.normalize();
                }
                black_box(test_vectors);
            },
        )
    });

    group.finish();
}

/// Benchmark spell correction operations.
fn bench_spell_correction(c: &mut Criterion) {
    let mut group = c.benchmark_group("spell_correction");
    group.sample_size(20); // Reduce sample size for faster execution

    let mut corrector = SpellingCorrector::new();

    // Common misspellings (reduced set for faster execution)
    let misspellings = vec!["searc", "engin", "documnet", "qurey", "algortihm"];

    // Single word correction
    group.bench_function("correct_single_word", |b| {
        b.iter(|| {
            let result = corrector.correct(black_box("searc"));
            black_box(result)
        })
    });

    // Batch correction
    group.throughput(Throughput::Elements(misspellings.len() as u64));
    group.bench_function("correct_batch_words", |b| {
        b.iter(|| {
            for word in &misspellings {
                let result = corrector.correct(black_box(word));
                black_box(result);
            }
        })
    });

    group.finish();
}

/// Benchmark parallel operations.
fn bench_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_operations");

    let vectors = generate_test_vectors(1000, 128);
    let query_vector = &vectors[0];

    // Parallel distance calculation
    group.throughput(Throughput::Elements(500));
    group.bench_function("parallel_distance_calculation", |b| {
        b.iter(|| {
            use rayon::prelude::*;
            let distances: Vec<_> = vectors[1..501]
                .par_iter()
                .map(|v| {
                    DistanceMetric::Cosine
                        .distance(&query_vector.data, &v.data)
                        .unwrap()
                })
                .collect();
            black_box(distances);
        })
    });

    // Sequential distance calculation for comparison
    group.bench_function("sequential_distance_calculation", |b| {
        b.iter(|| {
            let distances: Vec<_> = vectors[1..501]
                .iter()
                .map(|v| {
                    DistanceMetric::Cosine
                        .distance(&query_vector.data, &v.data)
                        .unwrap()
                })
                .collect();
            black_box(distances);
        })
    });

    // Parallel vector normalization
    group.throughput(Throughput::Elements(500));
    group.bench_function("parallel_vector_normalization", |b| {
        b.iter_with_setup(
            || vectors[0..500].to_vec(),
            |mut test_vectors| {
                use rayon::prelude::*;
                test_vectors.par_iter_mut().for_each(|v| v.normalize());
                black_box(test_vectors);
            },
        )
    });

    group.finish();
}

/// Memory usage and allocation benchmarks.
fn bench_memory_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_operations");

    // Vector allocation
    group.throughput(Throughput::Elements(1000));
    group.bench_function("vector_allocation", |b| {
        b.iter(|| {
            let mut vectors = Vec::new();
            for i in 0..1000 {
                let data: Vec<f32> = (0..128).map(|j| (i + j) as f32).collect();
                vectors.push(Vector::new(data));
            }
            black_box(vectors);
        })
    });

    group.finish();
}

/// Comprehensive benchmark suite covering different data sizes.
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    group.sample_size(10);

    for size in [100, 200].iter() {
        // Reduced sizes for faster execution
        // Vector indexing scalability
        group.bench_with_input(
            format!("vector_index_{size}_vectors"),
            size,
            |b, &vector_count| {
                let vectors = generate_test_vectors(vector_count, 128);

                b.iter_with_setup(
                    || {
                        HnswIndexBuilder::new(VectorIndexBuildConfig {
                            dimension: 128,
                            ..Default::default()
                        })
                        .unwrap()
                    },
                    |mut builder| {
                        let indexed_vectors: Vec<(u64, Vector)> = vectors
                            .iter()
                            .enumerate()
                            .map(|(i, v)| (i as u64, v.clone()))
                            .collect();
                        let _ = builder.build(indexed_vectors);
                        black_box(builder);
                    },
                )
            },
        );
    }

    group.finish();
}

/// Benchmark synonym dictionary operations.
fn bench_synonym_dictionary(c: &mut Criterion) {
    let mut group = c.benchmark_group("synonym_dictionary");

    // Create dictionary with varying sizes
    let small_dict = create_test_dictionary(100);
    let medium_dict = create_test_dictionary(1000);
    let large_dict = create_test_dictionary(10000);

    // Benchmark single lookup
    group.bench_function("lookup_small_100", |b| {
        b.iter(|| {
            let result = small_dict.get_synonyms(black_box("term_50"));
            black_box(result)
        })
    });

    group.bench_function("lookup_medium_1k", |b| {
        b.iter(|| {
            let result = medium_dict.get_synonyms(black_box("term_500"));
            black_box(result)
        })
    });

    group.bench_function("lookup_large_10k", |b| {
        b.iter(|| {
            let result = large_dict.get_synonyms(black_box("term_5000"));
            black_box(result)
        })
    });

    // Benchmark batch lookups
    group.throughput(Throughput::Elements(100));
    group.bench_function("batch_lookup_100", |b| {
        b.iter(|| {
            for i in 0..100 {
                let term = format!("term_{}", i);
                let result = large_dict.get_synonyms(black_box(&term));
                black_box(result);
            }
        })
    });

    // Benchmark dictionary creation
    group.bench_function("build_dict_1k", |b| {
        b.iter(|| {
            let dict = create_test_dictionary(1000);
            black_box(dict)
        })
    });

    group.finish();
}

/// Create a test dictionary with specified number of synonym groups.
fn create_test_dictionary(num_groups: usize) -> sage::analysis::synonym::SynonymDictionary {
    use sage::analysis::synonym::SynonymDictionary;

    let mut groups = Vec::new();
    for i in 0..num_groups {
        groups.push(vec![
            format!("term_{}", i),
            format!("synonym_a_{}", i),
            format!("synonym_b_{}", i),
        ]);
    }

    let mut dict = SynonymDictionary::new(None).unwrap();
    for group in groups {
        dict.add_synonym_group(group);
    }
    dict
}

// Group all benchmarks - core benchmarks for faster execution
criterion_group!(
    benches,
    bench_text_analysis,
    bench_vector_search,
    bench_parallel_operations,
    bench_memory_operations,
    bench_synonym_dictionary
);

// Separate group for slower benchmarks
criterion_group!(slow_benches, bench_spell_correction, bench_scalability);

criterion_main!(benches);
