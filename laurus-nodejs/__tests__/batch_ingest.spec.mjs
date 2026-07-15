/**
 * Integration tests for `Index.putDocuments` / `Index.addDocuments` (#866).
 *
 * Covers:
 * - Empty batch is a no-op.
 * - `putDocuments` applies the batch and deduplicates duplicate ids within
 *   one batch (last occurrence wins).
 * - `addDocuments` accumulates repeated ids as separate versions.
 */

import { describe, it, expect } from "vitest";
import { Index, Schema } from "../index.js";

async function createTextIndex() {
  const schema = new Schema();
  schema.addTextField("title");
  schema.setDefaultFields(["title"]);
  return await Index.create(null, schema);
}

describe("putDocuments / addDocuments", () => {
  it("empty batch is a no-op", async () => {
    const index = await createTextIndex();
    await index.putDocuments([]);
    await index.addDocuments([]);
    await index.commit();
    expect(index.stats().documentCount).toBe(0);
  });

  it("putDocuments applies the batch and dedups duplicate ids", async () => {
    const index = await createTextIndex();
    await index.putDocuments([
      ["doc1", { title: "One" }],
      ["doc2", { title: "Two" }],
      ["doc1", { title: "One v2" }],
    ]);
    await index.commit();

    expect(index.stats().documentCount).toBe(2);
    const docs = await index.getDocuments("doc1");
    expect(docs.length).toBe(1);
    expect(docs[0].title).toBe("One v2");
  });

  it("addDocuments accumulates repeated ids as chunks", async () => {
    const index = await createTextIndex();
    await index.addDocuments([
      ["doc", { title: "chunk 0" }],
      ["doc", { title: "chunk 1" }],
    ]);
    await index.commit();

    const docs = await index.getDocuments("doc");
    expect(docs.length).toBe(2);
  });
});
