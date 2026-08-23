/*
 * UniDic dictionary loader for the demo samples.
 *
 * Wraps the OPFS helpers exported by the laurus-wasm `pkg/opfs.js`
 * postbuild bundle and the `JapaneseAnalyzer.fromBytes` constructor.
 *
 * The dictionary cache key (`DICT_NAME`) is intentionally shared
 * across samples so that a returning visitor never re-downloads the
 * ~52 MB UniDic zip just because they switched samples.
 */

import {
  downloadDictionary,
  getDictionaryVersion,
  hasDictionary,
  loadDictionaryFiles,
  removeDictionary,
} from '../../pkg/opfs.js';
import { JapaneseAnalyzer } from '../../pkg/laurus_wasm.js';

/** Shared OPFS key for the UniDic dictionary. Do not rename. */
export const DICT_NAME = 'unidic';

/** Default analyzer registration name used by the samples. */
export const ANALYZER_NAME = 'ja-unidic';

/**
 * Default location of the UniDic zip relative to each sample HTML.
 * The CI deploy and the local development layout both place
 * `dict/lindera-unidic.zip` at `examples/dict/`, so each sample
 * references it via `../dict/lindera-unidic.zip`.
 */
export const DEFAULT_DICT_URL = '../dict/lindera-unidic.zip';

/**
 * Default location of the dictionary manifest written by the deploy
 * workflow next to the zip. Its `lindera_version` field identifies
 * which Lindera version the zip was built for, letting the loader
 * invalidate an OPFS cache left over from an older deployment (the
 * binary dictionary format is not stable across Lindera versions).
 */
export const DEFAULT_MANIFEST_URL = '../dict/manifest.json';

/**
 * Fetch the expected dictionary version from the deploy manifest.
 * Returns null when the manifest is unavailable or malformed (e.g.,
 * a local development server without a generated manifest), in which
 * case the caller falls back to trusting any cached dictionary.
 *
 * @param {string} url - URL of the manifest JSON.
 * @returns {Promise<string | null>} The expected version, or null.
 */
async function fetchExpectedVersion(url) {
  try {
    // The manifest is tiny; skip stale HTTP caches so a fresh deploy
    // is picked up promptly.
    const response = await fetch(url, { cache: 'no-cache' });
    if (!response.ok) {
      return null;
    }
    const manifest = await response.json();
    const version = manifest?.lindera_version;
    return typeof version === 'string' && version.length > 0 ? version : null;
  } catch {
    return null;
  }
}

/**
 * Make sure the UniDic dictionary is present in OPFS, downloading
 * the bundled zip on first visit. Reports progress through the
 * supplied logger and an optional status DOM hook.
 *
 * A cached dictionary is reused only if its stored version stamp
 * matches the `lindera_version` in the deploy manifest; on mismatch
 * (including caches stored before version stamping existed) the
 * cache is discarded and the zip is re-downloaded. When no manifest
 * is available the cache is trusted as-is.
 *
 * @param {object} options
 * @param {string} [options.url] - URL of the dictionary zip.
 * @param {string} [options.manifestUrl] - URL of the manifest JSON.
 * @param {{log: Function, ok: Function, err: Function}} options.logger
 * @param {(text: string) => void} [options.setStatus] - Optional status hook.
 */
export async function ensureDictionary({
  url = DEFAULT_DICT_URL,
  manifestUrl = DEFAULT_MANIFEST_URL,
  logger,
  setStatus,
} = {}) {
  const expected = await fetchExpectedVersion(manifestUrl);

  if (await hasDictionary(DICT_NAME)) {
    if (expected === null) {
      logger?.ok?.(
        `Dictionary "${DICT_NAME}" found in OPFS cache `
        + '(no manifest to verify against).',
      );
      return;
    }
    const cached = await getDictionaryVersion(DICT_NAME);
    if (cached === expected) {
      logger?.ok?.(
        `Dictionary "${DICT_NAME}" (Lindera ${cached}) found in OPFS cache.`,
      );
      return;
    }
    logger?.log?.(
      `Cached dictionary "${DICT_NAME}" is for Lindera `
      + `${cached ?? 'unknown'} but this build needs ${expected}; `
      + 're-downloading...',
    );
    await removeDictionary(DICT_NAME);
  }

  logger?.log?.(`Downloading dictionary "${DICT_NAME}" from ${url}...`);
  const t0 = performance.now();
  await downloadDictionary(url, DICT_NAME, {
    version: expected ?? undefined,
    onProgress: ({ phase, loaded, total }) => {
      if (phase === 'downloading' && total) {
        const pct = ((loaded / total) * 100).toFixed(0);
        const mb = (loaded / 1024 / 1024).toFixed(1);
        const totalMb = (total / 1024 / 1024).toFixed(1);
        setStatus?.(`downloading ${pct}% (${mb}/${totalMb} MB)`);
      } else if (phase === 'extracting') {
        setStatus?.('extracting...');
      } else if (phase === 'storing') {
        setStatus?.('storing to OPFS...');
      }
    },
  });
  const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
  logger?.ok?.(`Dictionary downloaded and cached in OPFS in ${elapsed}s.`);
}

/**
 * Build a `JapaneseAnalyzer` from the UniDic bytes stored in OPFS.
 * Call `ensureDictionary` first.
 *
 * @param {object} [options]
 * @param {'normal' | 'search' | 'extended'} [options.mode] - Tokenizer mode.
 * @returns {Promise<import('../../pkg/laurus_wasm.js').JapaneseAnalyzer>}
 */
export async function buildJapaneseAnalyzer({ mode = 'normal' } = {}) {
  const files = await loadDictionaryFiles(DICT_NAME);
  return JapaneseAnalyzer.fromBytes(
    files.metadata,
    files.dictTrie,
    files.dictValsIdx,
    files.dictVals,
    files.dictWordsIdx,
    files.dictWords,
    files.matrixMtx,
    files.charDef,
    files.unk,
    mode,
  );
}

/**
 * Remove the cached UniDic dictionary from OPFS. Swallows
 * `NotFoundError` so callers can use this in idempotent cleanup
 * helpers.
 *
 * @returns {Promise<boolean>} true if a directory was removed.
 */
export async function clearDictionary() {
  try {
    await removeDictionary(DICT_NAME);
    return true;
  } catch (e) {
    if (e?.name === 'NotFoundError') {
      return false;
    }
    throw e;
  }
}
