/*
 * Lightweight log panel helper for the demo samples.
 *
 * Each sample has a `<div class="log" id="log"></div>` element where
 * timestamped status lines are appended.
 */

/**
 * Create a logger bound to the element with the given id.
 *
 * @param {string} elementId - The DOM id of the log container.
 * @returns {{log: (msg: string, cls?: string) => void, ok: (msg: string) => void, err: (msg: string) => void}}
 */
export function createLogger(elementId = 'log') {
  const el = document.getElementById(elementId);
  if (!el) {
    return {
      log: () => {},
      ok: () => {},
      err: () => {},
    };
  }

  function append(msg, cls = '') {
    const line = document.createElement('span');
    line.className = cls;
    line.textContent = `${msg}\n`;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
  }

  return {
    log: (msg, cls = '') => append(msg, cls),
    ok: (msg) => append(msg, 'ok'),
    err: (msg) => append(msg, 'err'),
  };
}
