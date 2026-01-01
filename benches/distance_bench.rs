use criterion::{Criterion, black_box, criterion_group, criterion_main};
use sarissa::vector::core::distance::DistanceMetric;

fn generate_test_vectors(count: usize, dimension: usize) -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(count);
    for i in 0..count {
        let mut data = Vec::with_capacity(dimension);
        for j in 0..dimension {
            let value = ((i as f32 * 0.1 + j as f32 * 0.01).sin() * 0.5 + 0.5) * 2.0 - 1.0;
            data.push(value);
        }
        vectors.push(data);
    }
    vectors
}

fn bench_distances(c: &mut Criterion) {
    let dimension = 128;
    let vectors = generate_test_vectors(101, dimension);
    let query = &vectors[0];
    let targets = &vectors[1..101];

    let mut group = c.benchmark_group("distance_metrics");

    for metric in [
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::DotProduct,
    ] {
        group.bench_function(metric.name(), |b| {
            b.iter(|| {
                for target in targets {
                    let _ = black_box(
                        metric
                            .distance(black_box(query), black_box(target))
                            .unwrap(),
                    );
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_distances);
criterion_main!(benches);
