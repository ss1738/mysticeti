#!/usr/bin/env python3
"""
Per-protocol plots, emitted as THREE separate figures:

  * <protocol>_latency.png  — time-series latency line graph
  * <protocol>_phases.png   — per-phase average bar chart
  * <protocol>_leaders.png  — leaders-per-round trace over time (the adaptive
    scheduler's reaction); needs the `leaders` CSV column, so it is skipped
    for older result files
  * <protocol>_timeouts.png — share of rounds in which a validator waited out
    the leader timeout (mean per validator): the cost of slow or dead leader
    slots, which is what grows with the leader count; needs the `timeouts`,
    `direct_commits`, `direct_skips`, `indirect` and `leaders` columns, so it
    is skipped for older result files

The latency and phase figures share a single y-axis scale that is *normalised
across every protocol* under the results root, so graphs from different
protocols are directly comparable. Each figure still covers one protocol (the
adaptive scheduler vs fixed leader counts of 1 and 5).

Reads the CSVs produced by the `adaptive_scheduling` integration test
(`crates/simulator/tests/adaptive_scheduling.rs`).

Usage:
  # every protocol found under the results root (shared y-axis across all):
  python3 scripts/plot_adaptive_scheduling_normalized.py
  # a specific subset:
  python3 scripts/plot_adaptive_scheduling_normalized.py --protocol mysticeti blue-bottle
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


VARIANTS = [
    ("adaptive", "adaptive scheduler"),
    ("fixed_1",  "fixed leaders (n=1)"),
    ("fixed_5",  "fixed leaders (n=5)"),
]

# Human-readable scenario name (for figure titles) and the label drawn over the
# stressed middle phase, keyed by the results-root subfolder.
SCENARIO_DISPLAY   = {"crash": "crash faults", "degraded": "slow network"}
SCENARIO_MID_LABEL = {"crash": "crash",        "degraded": "slow network"}


def pretty_protocol(name):
    """'blue-bottle' -> 'Blue Bottle', 'mahi-mahi-4' -> 'Mahi Mahi 4'."""
    return name.replace("-", " ").title()

# Headroom above the shared data max so lines / bar labels are not clipped.
YMAX_HEADROOM = 1.08


def load_csv(path):
    data = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            avg = int(row["avg_ms"])
            if avg == 0:
                continue
            t     = int(row["time_s"])
            phase = int(row["phase"])
            data[t].append((avg, phase))

    t1 = t2 = None
    for t in sorted(data):
        phase = data[t][0][1]
        if t1 is None and phase == 2:
            t1 = t
        if t2 is None and phase == 3:
            t2 = t

    return data, t1, t2


def load_column_csv(path, column):
    """Per-time mean of an integer `column` over validators, or None for CSVs
    written before the column existed."""
    per_time = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get(column) in (None, ""):
                return None
            per_time[int(row["time_s"])].append(int(row[column]))
    return {t: sum(v) / len(v) for t, v in per_time.items()}


def load_column_dir(dirpath, column):
    """One `{time: mean value}` trace per run in the directory (runs without
    the column are skipped)."""
    runs = []
    if not os.path.isdir(dirpath):
        return runs
    for fname in sorted(f for f in os.listdir(dirpath) if f.endswith(".csv")):
        trace = load_column_csv(os.path.join(dirpath, fname), column)
        if trace:
            runs.append(trace)
    return runs


def load_leaders_dir(dirpath):
    return load_column_dir(dirpath, "leaders")


TIMEOUT_COLUMNS = ("timeouts", "direct_commits", "direct_skips", "indirect", "leaders")


def load_timeout_share_csv(path):
    """Per-time mean over validators of the share of rounds that waited out the
    leader timeout in the window: `timeouts / rounds`, with the rounds a window
    covered derived from its decided leader slots (`decisions / leaders`).
    None for CSVs written before the mechanism columns existed."""
    per_time = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if any(row.get(c) in (None, "") for c in TIMEOUT_COLUMNS):
                return None
            decisions = int(row["direct_commits"]) + int(row["direct_skips"]) + int(row["indirect"])
            rounds = decisions / max(int(row["leaders"]), 1)
            share = min(int(row["timeouts"]) / rounds, 1.0) if rounds > 0 else 0.0
            per_time[int(row["time_s"])].append(share)
    return {t: sum(v) / len(v) for t, v in per_time.items()}


def load_timeout_share_dir(dirpath):
    runs = []
    if not os.path.isdir(dirpath):
        return runs
    for fname in sorted(f for f in os.listdir(dirpath) if f.endswith(".csv")):
        trace = load_timeout_share_csv(os.path.join(dirpath, fname))
        if trace:
            runs.append(trace)
    return runs


def load_csv_dir(dirpath):
    """Load and merge all CSVs in a directory. Entries across runs are pooled
    per time point so the existing averaging functions naturally average across runs."""
    combined = defaultdict(list)
    t1 = t2 = None

    csv_files = sorted(
        f for f in os.listdir(dirpath) if f.endswith(".csv")
    ) if os.path.isdir(dirpath) else []

    for fname in csv_files:
        data, file_t1, file_t2 = load_csv(os.path.join(dirpath, fname))
        for t, entries in data.items():
            combined[t].extend(entries)
        if t1 is None and file_t1 is not None:
            t1 = file_t1
        if t2 is None and file_t2 is not None:
            t2 = file_t2

    return combined, t1, t2


def aggregate_timeseries(data):
    result = []
    for t in sorted(data):
        vals = data[t]
        mean_avg = sum(v[0] for v in vals) / len(vals)
        result.append((t, mean_avg))
    return result


PHASE2_SKIP_POINTS = 3  # drop first N timestamps of phase 2 while network stabilises
PHASE3_SKIP_POINTS = 5  # drop first N timestamps of phase 3 while network stabilises

def phase_means(data):
    by_phase = defaultdict(list)
    phase2_times = sorted(t for t in data if data[t][0][1] == 2)
    phase3_times = sorted(t for t in data if data[t][0][1] == 3)
    skip_before_p2 = phase2_times[PHASE2_SKIP_POINTS] if len(phase2_times) > PHASE2_SKIP_POINTS else None
    skip_before_p3 = phase3_times[PHASE3_SKIP_POINTS] if len(phase3_times) > PHASE3_SKIP_POINTS else None

    for t in data:
        for avg, phase in data[t]:
            if phase == 2 and skip_before_p2 is not None and t < skip_before_p2:
                continue
            if phase == 3 and skip_before_p3 is not None and t < skip_before_p3:
                continue
            by_phase[phase].append(avg)

    return tuple(
        sum(by_phase[p]) / len(by_phase[p]) if by_phase[p] else None
        for p in (1, 2, 3)
    )


def add_phase_shading(ax, t1, t2, t_max, labels=True, mid_label="high latency"):
    if t1 is None or t2 is None:
        return
    ax.axvspan(0,  t1,    alpha=0.06, color="green")
    ax.axvspan(t1, t2,    alpha=0.06, color="red")
    ax.axvspan(t2, t_max, alpha=0.06, color="green")
    for x_val, lbl in [(t1, "t₁"), (t2, "t₂")]:
        ax.axvline(x_val, color="dimgrey", linestyle="--", linewidth=1)
    if labels:
        xform = ax.get_xaxis_transform()
        for x_val, lbl in [(t1, "t₁"), (t2, "t₂")]:
            ax.text(x_val, 1.02, lbl, transform=xform,
                    ha="center", fontsize=10, color="dimgrey")
        ax.text((0 + t1) / 2,    1.07, "healthy",      transform=xform,
                ha="center", fontsize=9, color="darkgreen")
        ax.text((t1 + t2) / 2,   1.07, mid_label, transform=xform,
                ha="center", fontsize=9, color="firebrick")
        ax.text((t2 + t_max) / 2, 1.07, "healthy",     transform=xform,
                ha="center", fontsize=9, color="darkgreen")


COLORS  = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
MARKERS = ["o", "s", "^", "D"]


def leader_lag(pooled, t1, t2):
    """Measure the scheduler's lag from a pooled `{time: mean leaders}` trace:
    (first back-off after t1, time the trace settles at its phase-2 minimum,
    first time after t2 it is back at the healthy level). Each entry is
    None when the event does not occur."""
    if not pooled or t1 is None or t2 is None:
        return None, None, None
    times = sorted(pooled)
    healthy = [pooled[t] for t in times if t < t1]
    if not healthy:
        return None, None, None
    level = sum(healthy) / len(healthy)
    mid = [t for t in times if t1 <= t < t2]
    first_backoff = next((t for t in mid if pooled[t] < level - 0.5), None)
    settled = None
    if mid:
        floor = min(pooled[t] for t in mid)
        settled = next((t for t in mid if pooled[t] <= floor + 0.25), None)
    recovered = next((t for t in times if t >= t2 and pooled[t] >= level - 0.25), None)
    return first_backoff, settled, recovered


def plot_leaders(series, output, title=None, mid_label="high latency"):
    """Leaders-per-round trace. series: list of (label, runs, t1, t2) where
    `runs` is a list of per-run `{time: mean leaders}` traces."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if title:
        ax.set_title(title, fontweight="bold", fontsize=13, pad=40)

    t_max = max((t for _, runs, _, _ in series for r in runs for t in r), default=1)
    t1 = next((t1 for _, _, t1, _ in series if t1 is not None), None)
    t2 = next((t2 for _, _, _, t2 in series if t2 is not None), None)
    add_phase_shading(ax, t1, t2, t_max, labels=True, mid_label=mid_label)

    for idx, (label, runs, _, _) in enumerate(series):
        color = COLORS[idx % len(COLORS)]
        pooled = defaultdict(list)
        for run in runs:
            for t, v in run.items():
                pooled[t].append(v)
        pooled = {t: sum(v) / len(v) for t, v in pooled.items()}
        times = sorted(pooled)
        values = [pooled[t] for t in times]
        if max(values) == min(values):
            # A fixed leader count is a constant: draw it as a thin dashed
            # reference line underneath so it never hides the adaptive trace.
            ax.axhline(values[0], color=color, linestyle="--", linewidth=1.2,
                       alpha=0.8, zorder=1, label=label)
            continue
        for run in runs:
            run_times = sorted(run)
            ax.step(run_times, [run[t] for t in run_times], where="post",
                    color=color, linewidth=0.8, alpha=0.25, zorder=2)
        ax.step(times, values, where="post", color=color, linewidth=2.8,
                zorder=3, label=label)
        first_backoff, settled, recovered = leader_lag(pooled, t1, t2)
        if first_backoff is not None:
            parts = [f"first back-off t1+{first_backoff - t1:.0f}s"]
            if settled is not None:
                parts.append(f"settled t1+{settled - t1:.0f}s")
            parts.append(f"recovered t2+{recovered - t2:.0f}s" if recovered is not None
                         else "not recovered by end of run")
            print(f"  {label} lag: " + ", ".join(parts))

    ax.set_ylabel("Leaders per round", fontweight="bold")
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_xlim(0, t_max)
    ax.set_ylim(0.5, max(5.5, ax.get_ylim()[1]))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Legend below the axes so it never covers the traces.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(series),
              fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output}")


def plot_timeseries(series, output, ymax, title=None, mid_label="high latency"):
    """Time-series latency line graph. series: list of (label, ts_records, means, t1, t2)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    if title:
        ax.set_title(title, fontweight="bold", fontsize=13, pad=40)

    all_times = [r[0] for _, recs, _, _, _ in series for r in recs]
    t_max = max(all_times) if all_times else 1
    t1 = next((t1 for _, _, _, t1, _ in series if t1 is not None), None)
    t2 = next((t2 for _, _, _, _, t2 in series if t2 is not None), None)

    add_phase_shading(ax, t1, t2, t_max, labels=True, mid_label=mid_label)

    for idx, (label, ts_recs, _, _, _) in enumerate(series):
        times = [r[0] for r in ts_recs]
        avgs  = [r[1] for r in ts_recs]
        ax.plot(times, avgs, label=label, color=COLORS[idx % len(COLORS)],
                linewidth=1.2, marker=MARKERS[idx % len(MARKERS)], markersize=2.5)

    ax.set_ylabel("Latency (ms)", fontweight="bold")
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output}")


def plot_phases(series, output, ymax, title=None, mid_label="high latency"):
    """Per-phase average latency bar chart. series: list of (label, ts_records, means, t1, t2)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    if title:
        ax.set_title(title, fontweight="bold")

    phase_labels = ["Phase 1\n(healthy)", f"Phase 2\n({mid_label})", "Phase 3\n(healthy)"]
    n_series = len(series)
    x = np.arange(len(phase_labels))
    width = 0.22
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

    for idx, (label, _, means, _, _) in enumerate(series):
        heights = [m if m is not None else 0 for m in means]
        bars = ax.bar(x + offsets[idx], heights, width, label=label,
                      color=COLORS[idx % len(COLORS)], alpha=0.85)
        for bar, val in zip(bars, means):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{val:.0f}ms", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(phase_labels)
    ax.set_ylabel("Avg latency (ms)", fontweight="bold")
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output}")


def load_protocol_series(proto_dir, announce=False):
    """Return the per-variant series list for one protocol, or [] if no data."""
    series = []
    for variant, label in VARIANTS:
        dirpath = os.path.join(proto_dir, variant)
        if os.path.isdir(dirpath) and any(f.endswith(".csv") for f in os.listdir(dirpath)):
            data, t1, t2 = load_csv_dir(dirpath)
            if announce:
                n_runs = sum(1 for f in os.listdir(dirpath) if f.endswith(".csv"))
                print(f"Loaded {n_runs} run(s) for '{label}'")
            series.append((label, aggregate_timeseries(data), phase_means(data), t1, t2))
        elif announce:
            print(f"Warning: no CSVs found in {dirpath}, skipping '{label}'")
    return series


def load_protocol_column(proto_dir, column):
    """Per-variant traces of one CSV column: list of (label, runs, t1, t2);
    only variants whose CSVs carry the column are included."""
    series = []
    for variant, label in VARIANTS:
        dirpath = os.path.join(proto_dir, variant)
        runs = load_column_dir(dirpath, column)
        if not runs:
            continue
        _, t1, t2 = load_csv_dir(dirpath)
        series.append((label, runs, t1, t2))
    return series


def load_protocol_leaders(proto_dir):
    return load_protocol_column(proto_dir, "leaders")


def load_protocol_timeout_share(proto_dir):
    series = []
    for variant, label in VARIANTS:
        dirpath = os.path.join(proto_dir, variant)
        runs = load_timeout_share_dir(dirpath)
        if not runs:
            continue
        _, t1, t2 = load_csv_dir(dirpath)
        series.append((label, runs, t1, t2))
    return series


def plot_timeouts(series, output, title=None, mid_label="high latency"):
    """Share of rounds that waited out the leader timeout, per window, mean
    per validator, pooled over runs. series: list of (label, runs, t1, t2) as
    from `load_protocol_timeout_share`."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    if title:
        ax.set_title(title, fontweight="bold", fontsize=13, pad=40)
    t_max = max((t for _, runs, _, _ in series for r in runs for t in r), default=1)
    t1 = next((t1 for _, _, t1, _ in series if t1 is not None), None)
    t2 = next((t2 for _, _, _, t2 in series if t2 is not None), None)
    add_phase_shading(ax, t1, t2, t_max, labels=True, mid_label=mid_label)
    for idx, (label, runs, _, _) in enumerate(series):
        pooled = defaultdict(list)
        for run in runs:
            for t, v in run.items():
                pooled[t].append(v)
        times = sorted(pooled)
        ax.plot(times, [sum(pooled[t]) / len(pooled[t]) for t in times],
                color=COLORS[idx % len(COLORS)], marker=MARKERS[idx % len(MARKERS)],
                markersize=3, linewidth=1.5, label=label)
    ax.set_ylabel("Rounds that waited out\nthe leader timeout", fontweight="bold")
    ax.set_xlabel("Time (s)", fontweight="bold")
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(series),
              fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output}")


def discover_protocols(results_root):
    """Protocol folders under results_root that hold at least one variant directory."""
    if not os.path.isdir(results_root):
        return []
    protos = []
    for name in sorted(os.listdir(results_root)):
        pdir = os.path.join(results_root, name)
        if not os.path.isdir(pdir):
            continue
        if any(os.path.isdir(os.path.join(pdir, v)) for v, _ in VARIANTS):
            protos.append(name)
    return protos


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", nargs="+", default=None, metavar="PROTOCOL",
                        help="one or more protocol folders under the results root "
                             "(e.g. mysticeti blue-bottle nemo-nemo mahi-mahi-4 "
                             "mahi-mahi-5). Default: every protocol found.")
    parser.add_argument("--results-root", default="crates/simulator/results",
                        help="base directory holding the per-protocol result folders")
    parser.add_argument("--ymax", type=float, default=None,
                        help="override the shared y-axis max (ms); default is "
                             "auto-computed across all selected protocols")
    parser.add_argument("--scenario", default=None,
                        help="scenario name used in titles and the middle-phase "
                             "label; default inferred from the results-root "
                             "basename ('crash' or 'degraded')")
    args = parser.parse_args()

    scenario = args.scenario or os.path.basename(os.path.normpath(args.results_root))
    scenario_display = SCENARIO_DISPLAY.get(scenario, scenario)
    mid_label = SCENARIO_MID_LABEL.get(scenario, "high latency")

    protocols = args.protocol or discover_protocols(args.results_root)
    if not protocols:
        print(f"No protocol folders with data found under '{args.results_root}'. "
              f"Run the experiments first (scripts/run_protocol.sh <protocol>).")
        return
    print(f"Protocols: {', '.join(protocols)}")

    # Load every selected protocol's series first so the shared, normalised
    # y-axis can be computed across all of them and applied to each figure.
    loaded = []
    for proto in protocols:
        proto_dir = os.path.join(args.results_root, proto)
        print(f"\n[{proto}]")
        series = load_protocol_series(proto_dir, announce=True)
        if series:
            loaded.append((proto, proto_dir, series))
        else:
            print(f"  No data for '{proto}', skipping. "
                  f"(run scripts/run_protocol.sh {proto})")

    if not loaded:
        print("\nNo data found for any selected protocol.")
        return

    data_max = args.ymax if args.ymax is not None else max(
        avg
        for _, _, series in loaded
        for _, ts_recs, _, _, _ in series
        for _, avg in ts_recs
    )
    ymax = data_max * YMAX_HEADROOM if data_max and data_max > 0 else None
    if ymax:
        print(f"\nShared y-axis max: {ymax:.0f} ms (data max {data_max:.0f} ms)")

    for proto, proto_dir, series in loaded:
        prefix = os.path.join(proto_dir, proto)
        os.makedirs(proto_dir, exist_ok=True)
        title = f"{pretty_protocol(proto)} — {scenario_display}"
        plot_timeseries(series, f"{prefix}_latency.png", ymax, title, mid_label)
        plot_phases(series,     f"{prefix}_phases.png",  ymax, title, mid_label)
        leaders = load_protocol_leaders(proto_dir)
        if leaders:
            plot_leaders(leaders, f"{prefix}_leaders.png", title, mid_label)
        else:
            print(f"  No `leaders` column in {proto}'s CSVs, skipping the leaders trace")
        timeouts = load_protocol_timeout_share(proto_dir)
        if timeouts:
            plot_timeouts(timeouts, f"{prefix}_timeouts.png", title, mid_label)
        else:
            print(f"  No `timeouts` column in {proto}'s CSVs, skipping the timeouts figure")


if __name__ == "__main__":
    main()
