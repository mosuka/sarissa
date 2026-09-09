"""Prove that Index methods release the GIL while doing Rust/tokio work.

Before Issue #1103, every `Index` method held the GIL for the full duration
of its `rt.block_on(...)` call. A background Python thread calling another
`Index` method could not even *start* running until the foreground call
returned, because acquiring the GIL is a prerequisite for making any Python
call at all -- including entering another `Index` method.

These tests use that fact directly: kick off a slow call on one thread, and
from a second thread, call a quick `Index` method shortly after the slow
one starts. If the GIL is genuinely released while the slow call's
Rust/tokio work runs, the quick call can begin (and finish) *before* the
slow one completes. If the GIL were held for the whole slow call, the quick
call could not even begin until the slow call returns the GIL, so it would
necessarily start after the slow call finishes.

This measures actual interleaving of two genuine `Index` calls, rather than
inferring GIL state from a Python-level busy-loop counter's progress: an
earlier version of this test used a background thread incrementing a
counter as its signal, but that turned out to be an unreliable proxy in
this environment (it advanced during a plain blocking Rust call regardless
of whether the GIL was actually released around it).
"""

import threading
import time

import laurus


def _run_concurrently(slow, quick, quick_delay=0.02):
    """Run `slow` and `quick` on separate threads, starting `quick` after a
    short delay so `slow` is already well into its call. Returns a dict with
    perf_counter() timestamps: slow_start, slow_end, quick_start, quick_end.
    """
    start_barrier = threading.Event()
    times = {}

    def run_slow():
        start_barrier.wait()
        times["slow_start"] = time.perf_counter()
        slow()
        times["slow_end"] = time.perf_counter()

    def run_quick():
        start_barrier.wait()
        time.sleep(quick_delay)
        times["quick_start"] = time.perf_counter()
        quick()
        times["quick_end"] = time.perf_counter()

    t_slow = threading.Thread(target=run_slow)
    t_quick = threading.Thread(target=run_quick)
    t_slow.start()
    t_quick.start()
    start_barrier.set()
    t_slow.join()
    t_quick.join()
    return times


def _assert_quick_call_was_not_blocked_by(times, slow_label, quick_label, min_slow_duration=0.05):
    slow_duration = times["slow_end"] - times["slow_start"]
    assert slow_duration > min_slow_duration, (
        f"{slow_label} finished too fast for this test to be meaningful "
        f"({slow_duration:.4f}s) -- make the workload heavier"
    )

    assert times["quick_start"] < times["slow_end"], (
        f"{quick_label} did not start until after the concurrent {slow_label} "
        f"finished -- the GIL may not be released during {slow_label}"
    )


def _big_docs(n=50_000):
    return [
        (f"doc{i}", {"title": f"document number {i} rust python search engine benchmark"})
        for i in range(n)
    ]


def test_search_starts_before_a_concurrent_put_documents_finishes():
    idx = laurus.Index()
    idx.put_document("seed", {"title": "hello world"})
    idx.commit()
    docs = _big_docs()

    times = _run_concurrently(
        slow=lambda: idx.put_documents(docs),
        quick=lambda: idx.search("title:hello", limit=5),
    )
    _assert_quick_call_was_not_blocked_by(times, "put_documents()", "search()")


def test_search_starts_before_a_concurrent_commit_finishes():
    idx = laurus.Index()
    idx.put_document("seed", {"title": "hello world"})
    idx.commit()
    idx.put_documents(_big_docs())  # left uncommitted; commit() below applies it

    times = _run_concurrently(
        slow=idx.commit,
        quick=lambda: idx.search("title:hello", limit=5),
    )
    _assert_quick_call_was_not_blocked_by(times, "commit()", "search()", min_slow_duration=0.01)


def test_search_starts_before_a_concurrent_search_batch_finishes():
    idx = laurus.Index()
    idx.put_document("seed", {"title": "hello world"})
    idx.put_documents(_big_docs())
    idx.commit()

    times = _run_concurrently(
        slow=lambda: idx.search_batch(["title:document"] * 200, limit=10),
        quick=lambda: idx.search("title:hello", limit=5),
    )
    _assert_quick_call_was_not_blocked_by(times, "search_batch()", "search()")
