#!/usr/bin/env bash
# Per-protocol adaptive-scheduling experiment driver.
#
# Usage: scripts/run_protocol.sh <protocol>
#   <protocol> ∈ { mysticeti, blue-bottle, nemo-nemo, mahi-mahi-4, mahi-mahi-5, mahi-mahi-5-hl }
#
# Runs the adaptive, fixed-n=1 and fixed-n=5 variants of the 3-phase
# oscillating-network experiment RUNS times each, sequentially, then plots the
# pooled per-protocol comparison. The experiment tests are `#[ignore]`d
# (long-running), so they are invoked here with `--ignored`.
#
# Nothing is cleared: results accumulate under results/<protocol>/<variant>/,
# so re-running the script simply adds more runs to the pool. Runs are
# sequential so each one sees the previous run's CSV — the in-test `run_index`
# then auto-increments, giving every run a distinct slow-set rotation and seed.
set -u

PROTOCOL="${1:-}"
if [ -z "$PROTOCOL" ]; then
    echo "usage: $0 <protocol>"
    echo "  <protocol> ∈ { mysticeti, blue-bottle, nemo-nemo, mahi-mahi-4, mahi-mahi-5, mahi-mahi-5-hl }"
    exit 1
fi

RUNS=9
# Test functions use underscores; the protocol folder uses hyphens.
IDENT="${PROTOCOL//-/_}"
VARIANTS=("${IDENT}_adaptive" "${IDENT}_fixed_1" "${IDENT}_fixed_5")

echo "=== Building experiment test binary (no run) ==="
cargo test -p simulator --test adaptive_scheduling --no-run 2>&1 | tail -5
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "Build failed — aborting."
    exit 1
fi

echo ""
echo "=== Running $RUNS sequential runs x ${#VARIANTS[@]} variants for '$PROTOCOL' ==="
for i in $(seq 1 "$RUNS"); do
    for v in "${VARIANTS[@]}"; do
        echo "--- Run $i/$RUNS: $v ---"
        cargo test -p simulator --test adaptive_scheduling "$v" \
            -- --ignored --exact --nocapture 2>&1 \
            | grep -E "outcome|wrote|Leader scheduler|UNDECIDED-CHECK|FAILED|error" \
            || true
    done
done

echo ""
echo "=== All done. Generating plot for '$PROTOCOL'... ==="
python3 scripts/plot_adaptive_scheduling_normalized.py --protocol "$PROTOCOL"
