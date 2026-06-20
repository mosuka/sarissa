# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"

# Tests for the HNSW quantizer / rerank_storage schema options (Issue #797).
#
# These assert the values configured on +add_hnsw_field+ actually reach the
# Rust core via deterministic observables, not merely that search succeeds:
#
# * +rerank_storage: "f32"+ makes the core write a +*.hnsw.f32+ Stage-2
#   sidecar on disk (mirrors the server-side guard in #793/#800); the
#   default writes no sidecar.
# * +quantizer: "product_quantization"+ forwards +subvector_count+ to the
#   core's PQ training, which rejects a count that does not divide the
#   field dimension.
class TestVectorQuantizerRerank < Minitest::Test
  # Stable two-cluster corpus mirroring the core's
  # +test_hnsw_pq_search_returns_corpus_neighbour+ (issue #730).
  NEAR_OFFSETS = [
    [0.0, 0.0, 0.0, 0.0],
    [0.1, 0.1, 0.1, 0.1],
    [-0.1, -0.1, -0.1, -0.1],
    [0.2, -0.2, 0.2, -0.2],
    [-0.2, 0.2, -0.2, 0.2],
    [0.05, 0.05, -0.05, -0.05],
    [-0.05, -0.05, 0.05, 0.05],
    [0.15, -0.1, 0.1, -0.15],
  ].freeze
  NEAR_BASE = [10.0, 10.0, 20.0, 20.0].freeze
  FAR_BASE = [-100.0, -100.0, -200.0, -200.0].freeze

  def test_rerank_storage_f32_writes_sidecar
    Dir.mktmpdir do |dir|
      schema = Laurus::Schema.new
      schema.add_hnsw_field("embedding", 4, rerank_storage: "f32")
      idx = Laurus::Index.new(path: dir, schema: schema)
      idx.put_document("doc1", { "embedding" => [0.1, 0.2, 0.3, 0.4] })
      idx.put_document("doc2", { "embedding" => [0.9, 0.8, 0.7, 0.6] })
      idx.commit
      refute_empty Dir.glob(File.join(dir, "**", "*.hnsw.f32")),
                   "rerank_storage: 'f32' must write a .hnsw.f32 sidecar"
    end
  end

  def test_no_rerank_storage_writes_no_sidecar
    Dir.mktmpdir do |dir|
      schema = Laurus::Schema.new
      schema.add_hnsw_field("embedding", 4)
      idx = Laurus::Index.new(path: dir, schema: schema)
      idx.put_document("doc1", { "embedding" => [0.1, 0.2, 0.3, 0.4] })
      idx.put_document("doc2", { "embedding" => [0.9, 0.8, 0.7, 0.6] })
      idx.commit
      assert_empty Dir.glob(File.join(dir, "**", "*.hnsw.f32"))
    end
  end

  def test_product_quantization_builds_and_searches
    schema = Laurus::Schema.new
    # PQ is an L2 quantizer, so use Euclidean (matching the core's
    # test_hnsw_pq_search_returns_corpus_neighbour).
    schema.add_hnsw_field(
      "embedding", 4,
      distance: "euclidean", quantizer: "product_quantization", subvector_count: 2
    )
    idx = Laurus::Index.new(schema: schema)
    NEAR_OFFSETS.each_with_index do |off, i|
      idx.put_document("near#{i}", { "embedding" => NEAR_BASE.map.with_index { |b, j| b + off[j] } })
      idx.put_document("far#{i}", { "embedding" => FAR_BASE.map.with_index { |b, j| b + off[j] } })
    end
    idx.commit

    results = idx.search(Laurus::VectorQuery.new("embedding", NEAR_BASE), limit: 3)
    assert_equal 3, results.length
    assert(results.all? { |r| r.id.start_with?("near") })
  end

  def test_pq_subvector_count_must_divide_dimension
    schema = Laurus::Schema.new
    schema.add_hnsw_field(
      "embedding", 4, quantizer: "product_quantization", subvector_count: 3
    )
    idx = Laurus::Index.new(schema: schema)
    idx.put_document("doc1", { "embedding" => [0.1, 0.2, 0.3, 0.4] })
    assert_raises { idx.commit }
  end

  def test_unknown_quantizer_rejected
    schema = Laurus::Schema.new
    assert_raises(ArgumentError) { schema.add_hnsw_field("embedding", 4, quantizer: "bogus") }
  end

  def test_pq_requires_subvector_count
    schema = Laurus::Schema.new
    assert_raises(ArgumentError) do
      schema.add_hnsw_field("embedding", 4, quantizer: "product_quantization")
    end
  end

  def test_subvector_count_rejected_for_scalar
    schema = Laurus::Schema.new
    assert_raises(ArgumentError) { schema.add_hnsw_field("embedding", 4, subvector_count: 2) }
  end

  def test_unknown_rerank_storage_rejected
    schema = Laurus::Schema.new
    assert_raises(ArgumentError) { schema.add_hnsw_field("embedding", 4, rerank_storage: "bogus") }
  end
end
