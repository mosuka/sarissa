/*
 * Multilingual MiniLM embedder helper for the demo samples.
 *
 * Loads `Xenova/paraphrase-multilingual-MiniLM-L12-v2` via Hugging
 * Face Transformers.js (CDN) and returns a callable that emits
 * 384-dim L2-normalised embeddings. The model itself (~120 MB) is
 * cached by the browser after first load.
 */

import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

/** Embedder registration name used by the samples. */
export const EMBEDDER_NAME = 'multilingual-minilm';

/** Hugging Face model id. */
export const EMBEDDER_MODEL = 'Xenova/paraphrase-multilingual-MiniLM-L12-v2';

/** Output dimensionality of the model. Matches HNSW field config. */
export const EMBED_DIM = 384;

/**
 * Quantization dtype passed to Transformers.js. `q8` matches the
 * WASM backend default while silencing the v3 warning that fires
 * when the dtype is left unspecified.
 */
const EMBEDDER_DTYPE = 'q8';

/**
 * Load the embedder pipeline. Returns an async function suitable
 * for use as a laurus-wasm callback embedder.
 *
 * @param {object} [options]
 * @param {{log: Function, ok: Function}} [options.logger]
 * @returns {Promise<(text: string) => Promise<number[]>>}
 */
export async function loadEmbedder({ logger } = {}) {
  logger?.log?.(`Loading embedding model (${EMBEDDER_MODEL}, dtype=${EMBEDDER_DTYPE})...`);
  const t0 = performance.now();
  const pipe = await pipeline('feature-extraction', EMBEDDER_MODEL, {
    dtype: EMBEDDER_DTYPE,
  });
  const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
  logger?.ok?.(`Embedding model ready in ${elapsed}s.`);

  return async (text) => {
    const out = await pipe(text, { pooling: 'mean', normalize: true });
    return Array.from(out.data);
  };
}
