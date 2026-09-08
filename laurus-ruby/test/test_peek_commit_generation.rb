# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"

# Tests for `Laurus.peek_commit_generation()` (Issue #1101).
#
# Unlike `Index#stats`, this is a module function, not tied to any `Index`
# instance: it reads `commit_generation.json` directly from disk without
# building an `Engine` at all, so it works even when no `Index` for the
# given path has ever been constructed in this process -- the point being
# to let a caller cheaply decide whether opening (or reloading) the index
# is worth doing at all.
class TestPeekCommitGeneration < Minitest::Test
  def test_rejects_a_non_index_directory
    Dir.mktmpdir do |dir|
      err = assert_raises(ArgumentError) do
        Laurus.peek_commit_generation(dir)
      end
      assert_match(/not a laurus index directory/, err.message)
    end
  end

  def test_is_zero_before_any_commit
    Dir.mktmpdir do |dir|
      schema = Laurus::Schema.new
      schema.add_text_field("title")
      Laurus::Index.new(path: dir, schema: schema)

      assert_equal 0, Laurus.peek_commit_generation(dir)
    end
  end

  def test_advances_after_a_commit
    # laurus-ruby's `stats` doesn't expose a `commit_generation` key (that's
    # currently laurus-python-only), so this only checks the raw counter.
    Dir.mktmpdir do |dir|
      schema = Laurus::Schema.new
      schema.add_text_field("title")
      idx = Laurus::Index.new(path: dir, schema: schema)
      idx.put_document("doc1", { "title" => "hello world" })
      idx.commit

      assert_equal 1, Laurus.peek_commit_generation(dir)
    end
  end

  def test_sees_a_commit_made_by_another_handle
    Dir.mktmpdir do |dir|
      a = Laurus::Index.new(path: dir)
      a.put_document("doc1", {})
      a.commit
      a.close

      before = Laurus.peek_commit_generation(dir)

      b = Laurus::Index.new(path: dir)
      b.put_document("doc2", {})
      b.commit
      b.close

      refute_equal before, Laurus.peek_commit_generation(dir)
    end
  end
end
