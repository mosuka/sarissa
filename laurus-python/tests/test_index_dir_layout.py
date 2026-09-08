"""Tests for the on-disk directory layout of a file-backed `Index` (Issue #1059).

Before this change, `laurus.Index(path=X)` wrote segment files directly
under `X`, incompatible with `laurus-cli`'s `<X>/schema.toml` + `<X>/store/`
convention. These tests verify the new shared layout: schema
auto-persistence, auto-loading on reopen, the reopen-with-schema conflict
error, and legacy-layout detection.
"""

import pytest
import laurus


def test_creating_a_file_backed_index_writes_schema_toml_and_store(tmp_path):
    schema = laurus.Schema()
    schema.add_text_field("title")

    laurus.Index(path=str(tmp_path), schema=schema)

    assert (tmp_path / "schema.toml").is_file()
    assert (tmp_path / "store").is_dir()
    # No stray top-level segment directories from the old flat layout.
    assert not (tmp_path / "lexical").exists()


def test_reopen_without_schema_loads_persisted_schema_and_data(tmp_path):
    schema = laurus.Schema()
    schema.add_text_field("title")
    schema.set_default_fields(["title"])

    idx = laurus.Index(path=str(tmp_path), schema=schema)
    idx.put_document("doc1", {"title": "hello world"})
    idx.commit()
    idx.close()

    reopened = laurus.Index(path=str(tmp_path))
    results = reopened.search("title:hello", limit=5)
    assert len(results) == 1


def test_reopen_with_explicit_schema_raises_value_error(tmp_path):
    schema = laurus.Schema()
    schema.add_text_field("title")
    laurus.Index(path=str(tmp_path), schema=schema)

    with pytest.raises(ValueError, match="schema.toml"):
        laurus.Index(path=str(tmp_path), schema=schema)


def test_reopen_with_no_schema_at_all_succeeds_on_empty_default(tmp_path):
    # First call with no schema creates an empty-schema index (unchanged
    # default behavior); reopening it (also with no schema) must not raise.
    idx = laurus.Index(path=str(tmp_path))
    idx.close()
    laurus.Index(path=str(tmp_path))


def test_legacy_flat_layout_is_rejected(tmp_path):
    # Simulate a directory written by a laurus-python version predating
    # Issue #1059: segment files directly under the path, no schema.toml.
    (tmp_path / "engine.wal").write_bytes(b"")

    with pytest.raises(ValueError, match="pre-Issue-1059"):
        laurus.Index(path=str(tmp_path))


def test_new_empty_directory_is_not_treated_as_legacy(tmp_path):
    # A directory that merely exists but has no laurus files at all is a
    # normal fresh-create case, not a legacy-layout error.
    laurus.Index(path=str(tmp_path))
    assert (tmp_path / "schema.toml").is_file()
