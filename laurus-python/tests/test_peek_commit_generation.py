"""Tests for `laurus.peek_commit_generation()` (Issue #1101).

Unlike `Index.commit_generation()`, this is a module-level function, not an
`Index` method: it reads `commit_generation.json` directly from disk without
building an `Engine` at all, so it works even when no `Index` for the given
path has ever been constructed in this process -- the point being to let a
caller cheaply decide whether opening (or reloading) the index is worth
doing at all.
"""

import pytest
import laurus


def test_peek_commit_generation_rejects_a_non_index_directory(tmp_path):
    with pytest.raises(ValueError, match="not a laurus index directory"):
        laurus.peek_commit_generation(str(tmp_path))


def test_peek_commit_generation_is_zero_before_any_commit(tmp_path):
    path = str(tmp_path)
    schema = laurus.Schema()
    schema.add_text_field("title")
    laurus.Index(path=path, schema=schema)

    assert laurus.peek_commit_generation(path) == 0


def test_peek_commit_generation_matches_commit_generation_after_a_commit(tmp_path):
    path = str(tmp_path)
    schema = laurus.Schema()
    schema.add_text_field("title")
    index = laurus.Index(path=path, schema=schema)
    index.put_document("doc1", {"title": "hello world"})
    index.commit()

    assert laurus.peek_commit_generation(path) == index.commit_generation()
    assert laurus.peek_commit_generation(path) == 1


def test_peek_commit_generation_sees_a_commit_made_by_another_handle(tmp_path):
    path = str(tmp_path)
    a = laurus.Index(path=path)
    a.put_document("doc1", {})
    a.commit()
    a.close()

    before = laurus.peek_commit_generation(path)

    b = laurus.Index(path=path)
    b.put_document("doc2", {})
    b.commit()
    b.close()

    assert laurus.peek_commit_generation(path) != before
