"""Integration tests for ``Index.put_documents`` / ``Index.add_documents`` (#866).

These tests cover the Python-facing batched ingestion API:

- Empty batch is a no-op.
- ``put_documents`` applies the batch and deduplicates duplicate ids
  within one batch (last occurrence wins), matching a sequence of
  ``put_document`` calls.
- ``add_documents`` accumulates repeated ids as separate versions.
- A malformed entry (not an ``(id, dict)`` pair) is rejected naming its
  position.
"""

import pytest
import laurus


def test_put_documents_empty_batch_is_noop():
    idx = laurus.Index()
    idx.put_documents([])
    idx.add_documents([])
    idx.commit()
    assert idx.stats()["document_count"] == 0


def test_put_documents_applies_and_dedupes():
    idx = laurus.Index()
    idx.put_documents(
        [
            ("doc1", {"title": "One"}),
            ("doc2", {"title": "Two"}),
            ("doc1", {"title": "One v2"}),  # duplicate id: last wins
        ]
    )
    idx.commit()

    assert idx.stats()["document_count"] == 2
    docs = idx.get_documents("doc1")
    assert len(docs) == 1
    assert docs[0]["title"] == "One v2"


def test_add_documents_accumulates_chunks():
    idx = laurus.Index()
    idx.add_documents(
        [
            ("doc", {"title": "chunk 0"}),
            ("doc", {"title": "chunk 1"}),
        ]
    )
    idx.commit()

    docs = idx.get_documents("doc")
    assert len(docs) == 2


def test_put_documents_rejects_malformed_entry():
    idx = laurus.Index()
    with pytest.raises(Exception) as excinfo:
        idx.put_documents([("ok", {"title": "fine"}), "not-a-pair"])
    assert "documents[1]" in str(excinfo.value)
