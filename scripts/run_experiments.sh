#!/usr/bin/env bash
# Adaptive leader-scheduling experiment driver.
#
# Runs the `adaptive`, `fixed_1`, and `fixed_5` variants of the 3-phase
# oscillating-network experiment several times each, then plots the pooled
# results. The experiment tests are `#[ignore]`d (long-running), so they are
# invoked here with `--ignored`.
set -e

RUNS=3
DELAY=3

echo "=== Building experiment test binary (no run) ==="
cargo test -p simulator --test adaptive_scheduling --no-run 2>&1 | tail -5

echo ""
echo "=== Launching $RUNS runs x 3 variants (staggered ${DELAY}s apart) ==="

run_variant() {
    local name="$1"
    cargo test -p simulator --test adaptive_scheduling "$name" -- --ignored --nocapture \
        2>&1 | grep -E "outcome|wrote|FAILED|error" &
}

for i in $(seq 1 "$RUNS"); do
    echo "--- Run $i: adaptive ---"
    run_variant adaptive_leader_scheduling
    sleep "$DELAY"

    echo "--- Run $i: fixed n=1 ---"
    run_variant fixed_leaders_1
    sleep "$DELAY"

    echo "--- Run $i: fixed n=5 ---"
    run_variant fixed_leaders_5
    sleep "$DELAY"
done

echo ""
echo "=== Waiting for all experiments to finish... ==="
wait

echo ""
echo "=== All done. Generating plot... ==="
python3 scripts/plot_adaptive_scheduling_normalized.py
