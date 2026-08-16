/*
 * OPFS index version gate for the demo samples.
 *
 * A persisted index written by an older laurus-wasm build may use
 * on-disk formats the current build can no longer read. The failure
 * surfaces lazily — `Index.open` succeeds and only the first affected
 * search fails (e.g. "Unsupported legacy term dictionary format.
 * Rebuild required.") — so the samples gate proactively: each index
 * name is stamped with the building laurus-wasm version in
 * localStorage, and a mismatching (or missing) stamp wipes the index
 * directory so the sample rebuilds from scratch instead of serving
 * broken searches. See GitHub issue #981.
 *
 * The dictionary cache is untouched: it has its own Lindera-version
 * gate (issue #975).
 */

import { version } from '../../pkg/laurus_wasm.js';

/**
 * Ensure the persisted OPFS index (if any) was written by the current
 * laurus-wasm build; remove it otherwise so the caller's `Index.open`
 * starts from a clean directory. Call after `init()` and before
 * `Index.open`.
 *
 * @param {string} indexName - OPFS directory name of the index.
 * @param {{log: Function, ok: Function}} [logger] - Optional demo logger.
 */
export async function ensureIndexVersion(indexName, logger) {
  const key = `laurus-index-version:${indexName}`;
  const current = version();
  let stored = null;
  try {
    stored = localStorage.getItem(key);
  } catch {
    // Storage access denied (e.g. some private modes) — fall through
    // and wipe, which is the safe default.
  }
  if (stored !== current) {
    if (stored !== null) {
      logger?.log?.(
        `Index "${indexName}" was built by laurus-wasm ${stored}; `
        + `current is ${current} — rebuilding.`,
      );
    }
    const root = await navigator.storage.getDirectory();
    try {
      await root.removeEntry(indexName, { recursive: true });
    } catch (e) {
      if (e?.name !== 'NotFoundError') throw e;
    }
  }
  try {
    localStorage.setItem(key, current);
  } catch {
    // Non-fatal: without a stamp the next visit wipes again.
  }
}
