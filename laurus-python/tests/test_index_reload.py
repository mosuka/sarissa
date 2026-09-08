"""Tests for `Index.reload()` (Issue #1089).

Before this change, the only way to pick up changes committed by another
process was to construct a brand-new `Index`, which always pays the full
`Engine::build()` cost (embedding-model reload in particular). `reload()`
rebuilds the `Engine` for the same directory in place, reusing the
already-loaded embedder(s) when the schema's embedding configuration hasn't
changed, and works whether the `Index` is currently open or already
`close()`d -- reopening the same directory either way. That close()-interop
is what makes "pick up an external commit" testable deterministically: the
directory lock (Issue #1086) is a single, process-wide exclusive lock, so
simulating "another process wrote" only requires a second `Index` handle
that opens after the first one closes.
"""

import laurus


def _index_with_title_field(path=None):
    schema = laurus.Schema()
    schema.add_text_field("title")
    schema.set_default_fields(["title"])
    return laurus.Index(path=path, schema=schema)


def test_reload_noop_when_nothing_changed(tmp_path):
    index = _index_with_title_field(str(tmp_path))
    index.put_document("doc1", {"title": "hello world"})
    index.commit()

    assert index.reload() is False
    results = index.search("title:hello", limit=5)
    assert len(results) == 1


def test_reload_picks_up_external_commit(tmp_path):
    path = str(tmp_path)
    a = _index_with_title_field(path)
    a.put_document("doc1", {"title": "hello world"})
    a.commit()
    a.close()

    # Simulate another process writing to the same directory while `a` is
    # not holding the lock.
    b = laurus.Index(path=path)
    b.put_document("doc2", {"title": "hello again"})
    b.commit()
    b.close()

    assert a.reload() is True
    results = a.search("title:hello", limit=5)
    assert len(results) == 2


def test_reload_requires_file_backed_index():
    index = laurus.Index()
    try:
        index.reload()
        assert False, "expected reload() to raise on an in-memory index"
    except ValueError:
        pass


def test_reload_preserves_commit_policy(tmp_path):
    path = str(tmp_path)
    schema = laurus.Schema()
    schema.add_text_field("title")
    policy = laurus.CommitPolicy.every_docs(1)
    index = laurus.Index(path=path, schema=schema, commit_policy=policy)

    index.put_document("doc1", {"title": "auto committed"})
    assert len(index.get_documents("doc1")) == 1  # visible without an explicit commit()

    index.reload()

    # Auto-commit must still apply after reload(), not silently reset to manual.
    index.put_document("doc2", {"title": "still auto committed"})
    assert len(index.get_documents("doc2")) == 1


def test_commit_generation_matches_stats(tmp_path):
    index = _index_with_title_field(str(tmp_path))
    index.put_document("doc1", {"title": "hello world"})
    index.commit()

    assert index.commit_generation() == index.stats()["commit_generation"]
