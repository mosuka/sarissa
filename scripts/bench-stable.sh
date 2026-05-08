#!/usr/bin/env bash
# Run a Criterion bench with the noise floor knocked down.
#
# What it does (Linux only):
#   * Pins the cargo bench process to one logical CPU via `taskset`. This
#     stops the kernel from migrating samples across cores mid-measurement.
#   * Raises scheduling priority a notch via `nice` so background work is
#     less likely to preempt the bench.
#
# What it does NOT do:
#   * Disable turbo boost or pin the CPU governor — both need root and we
#     do not want to silently mutate system state. Document them as host-
#     side preparation if you need extra precision (see common.rs doc
#     comment).
#
# Usage:
#   ./scripts/bench-stable.sh <cargo-bench args>
#
# Examples:
#   ./scripts/bench-stable.sh --bench lexical_search_bench -- topk_or_skewed_tf
#   ./scripts/bench-stable.sh --bench lexical_search_bench -- \
#       'lexical/topk_or_skewed_tf/should_or_topk10/100000' --save-baseline pre-466
#
# Override which core to pin to with PIN_CORE (default: highest CPU index, on
# the assumption it is least contended on a desktop):
#   PIN_CORE=2 ./scripts/bench-stable.sh --bench lexical_search_bench
#
# On macOS / non-Linux hosts the wrapper degrades to a plain `cargo bench`
# call with a one-line warning, since `taskset` is Linux-only.

set -euo pipefail

OS="$(uname -s)"

if [[ "$OS" != "Linux" ]]; then
    echo "warning: bench-stable.sh: CPU pinning requires Linux; running plain cargo bench on $OS" >&2
    exec cargo bench "$@"
fi

if ! command -v taskset >/dev/null 2>&1; then
    echo "warning: bench-stable.sh: taskset not on PATH; running plain cargo bench" >&2
    exec cargo bench "$@"
fi

# Pick a default core. We avoid core 0 (it fields the bulk of interrupts
# on a Linux desktop) and we avoid the high end of the index space — on
# Intel hybrid CPUs (12th gen+) the higher-numbered logical CPUs are
# efficiency cores, and pinning to one halves throughput. Core 2 is on a
# P-core on every consumer Intel hybrid layout and is just an interior
# core on AMD / older Intel where all cores are equal.
NPROC="$(nproc)"
DEFAULT_CORE=2
if (( NPROC <= DEFAULT_CORE )); then
    DEFAULT_CORE=$((NPROC - 1))
fi
PIN_CORE="${PIN_CORE:-$DEFAULT_CORE}"

if (( PIN_CORE < 0 || PIN_CORE >= NPROC )); then
    echo "error: PIN_CORE=$PIN_CORE is out of range for nproc=$NPROC" >&2
    exit 1
fi

echo "bench-stable.sh: pinning to core $PIN_CORE / $NPROC, nice +5" >&2
echo "bench-stable.sh: override with PIN_CORE=<n>; on Intel hybrid CPUs avoid efficiency cores (high indices)" >&2

exec nice -n 5 taskset -c "$PIN_CORE" cargo bench "$@"
