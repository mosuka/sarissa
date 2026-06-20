// Tests for the HNSW quantizer / rerankStorage schema options (Issue #797).
//
// These assert the values configured on `addHnswField` actually reach the
// Rust core via deterministic observables, not merely that search succeeds:
//
// * `rerankStorage: "f32"` makes the core write a `*.hnsw.f32` Stage-2
//   sidecar on disk (mirrors the server-side guard in #793/#800); the
//   default writes no sidecar.
// * `quantizer: "product_quantization"` forwards `subvectorCount` to the
//   core's PQ training, which rejects a count that does not divide the
//   field dimension.

import { describe, it, expect } from "vitest";
import { mkdtempSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Index, Schema } from "../index.js";

// `addHnswField` is positional:
//   (name, dimension, distance, m, efConstruction, defaultEfSearch,
//    embedder, quantizer, subvectorCount, rerankStorage)
function findSidecars(dir) {
  let found = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) found = found.concat(findSidecars(p));
    else if (entry.name.endsWith(".hnsw.f32")) found.push(p);
  }
  return found;
}

describe("HNSW quantizer / rerankStorage options (#797)", () => {
  it("rerankStorage 'f32' writes a .hnsw.f32 sidecar on disk", async () => {
    const dir = mkdtempSync(join(tmpdir(), "laurus-rerank-"));
    const schema = new Schema();
    schema.addHnswField(
      "embedding", 4, undefined, undefined, undefined, undefined, undefined, undefined, undefined, "f32",
    );
    const index = await Index.create(dir, schema);
    await index.putDocument("doc1", { embedding: [0.1, 0.2, 0.3, 0.4] });
    await index.putDocument("doc2", { embedding: [0.9, 0.8, 0.7, 0.6] });
    await index.commit();
    expect(findSidecars(dir).length).toBeGreaterThan(0);
  });

  it("no rerankStorage writes no sidecar", async () => {
    const dir = mkdtempSync(join(tmpdir(), "laurus-norerank-"));
    const schema = new Schema();
    schema.addHnswField("embedding", 4);
    const index = await Index.create(dir, schema);
    await index.putDocument("doc1", { embedding: [0.1, 0.2, 0.3, 0.4] });
    await index.putDocument("doc2", { embedding: [0.9, 0.8, 0.7, 0.6] });
    await index.commit();
    expect(findSidecars(dir).length).toBe(0);
  });

  it("product_quantization builds and searches the near cluster", async () => {
    const schema = new Schema();
    // PQ is an L2 quantizer, so use Euclidean (matching the core's
    // `test_hnsw_pq_search_returns_corpus_neighbour`).
    schema.addHnswField(
      "embedding", 4, "euclidean", undefined, undefined, undefined, undefined, "product_quantization", 2, undefined,
    );
    const index = await Index.create(null, schema);
    // Stable two-cluster corpus mirroring the core's
    // `test_hnsw_pq_search_returns_corpus_neighbour` (issue #730).
    const nearOffsets = [
      [0.0, 0.0, 0.0, 0.0],
      [0.1, 0.1, 0.1, 0.1],
      [-0.1, -0.1, -0.1, -0.1],
      [0.2, -0.2, 0.2, -0.2],
      [-0.2, 0.2, -0.2, 0.2],
      [0.05, 0.05, -0.05, -0.05],
      [-0.05, -0.05, 0.05, 0.05],
      [0.15, -0.1, 0.1, -0.15],
    ];
    const nearBase = [10.0, 10.0, 20.0, 20.0];
    const farBase = [-100.0, -100.0, -200.0, -200.0];
    for (let i = 0; i < nearOffsets.length; i++) {
      await index.putDocument(`near${i}`, {
        embedding: nearBase.map((b, j) => b + nearOffsets[i][j]),
      });
      await index.putDocument(`far${i}`, {
        embedding: farBase.map((b, j) => b + nearOffsets[i][j]),
      });
    }
    await index.commit();

    const results = await index.searchVector("embedding", nearBase, 3);
    expect(results.length).toBe(3);
    expect(results.every((r) => r.id.startsWith("near"))).toBe(true);
  });

  it("rejects product_quantization whose subvectorCount does not divide the dimension", async () => {
    const schema = new Schema();
    schema.addHnswField(
      "embedding", 4, undefined, undefined, undefined, undefined, undefined, "product_quantization", 3, undefined,
    );
    const index = await Index.create(null, schema);
    await index.putDocument("doc1", { embedding: [0.1, 0.2, 0.3, 0.4] });
    await expect(index.commit()).rejects.toThrow();
  });

  it("rejects an unknown quantizer", () => {
    const schema = new Schema();
    expect(() =>
      schema.addHnswField(
        "embedding", 4, undefined, undefined, undefined, undefined, undefined, "bogus",
      ),
    ).toThrow();
  });

  it("rejects product_quantization without subvectorCount", () => {
    const schema = new Schema();
    expect(() =>
      schema.addHnswField(
        "embedding", 4, undefined, undefined, undefined, undefined, undefined, "product_quantization",
      ),
    ).toThrow();
  });

  it("rejects subvectorCount for a scalar quantizer", () => {
    const schema = new Schema();
    expect(() =>
      schema.addHnswField(
        "embedding", 4, undefined, undefined, undefined, undefined, undefined, undefined, 2,
      ),
    ).toThrow();
  });

  it("rejects an unknown rerankStorage", () => {
    const schema = new Schema();
    expect(() =>
      schema.addHnswField(
        "embedding", 4, undefined, undefined, undefined, undefined, undefined, undefined, undefined, "bogus",
      ),
    ).toThrow();
  });
});
