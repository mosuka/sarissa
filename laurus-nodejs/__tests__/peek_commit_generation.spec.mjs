// Tests for `peekCommitGeneration()` (Issue #1101).
//
// Unlike `Index.stats()`, this is a module-level export, not tied to any
// `Index` instance: it reads `commit_generation.json` directly from disk
// without building an `Engine` at all, so it works even when no `Index` for
// the given path has ever been constructed in this process -- the point
// being to let a caller cheaply decide whether opening (or reloading) the
// index is worth doing at all.

import { describe, it, expect } from "vitest";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Index, Schema, peekCommitGeneration } from "../index.js";

function freshDir() {
  return mkdtempSync(join(tmpdir(), "laurus-peek-"));
}

describe("peekCommitGeneration (#1101)", () => {
  it("rejects a directory with no schema.toml", () => {
    const dir = freshDir();
    expect(() => peekCommitGeneration(dir)).toThrow(/not a laurus index directory/);
  });

  it("is zero before any commit", async () => {
    const dir = freshDir();
    const schema = new Schema();
    schema.addTextField("title");
    await Index.create(dir, schema);

    expect(peekCommitGeneration(dir)).toBe(0);
  });

  it("advances after a commit", async () => {
    // stats() doesn't expose a commitGeneration key yet in this binding
    // (that's currently laurus-python-only), so this only checks the raw
    // counter.
    const dir = freshDir();
    const index = await Index.create(dir);
    await index.putDocument("doc1", { title: "hello world" });
    await index.commit();

    expect(peekCommitGeneration(dir)).toBe(1);
  });

  it("sees a commit made by another handle", async () => {
    const dir = freshDir();
    const a = await Index.create(dir);
    await a.putDocument("doc1", {});
    await a.commit();
    a.close();

    const before = peekCommitGeneration(dir);

    const b = await Index.create(dir);
    await b.putDocument("doc2", {});
    await b.commit();
    b.close();

    expect(peekCommitGeneration(dir)).not.toBe(before);
  });
});
