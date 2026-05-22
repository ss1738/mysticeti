#!/usr/bin/env python3
"""
Combined plot: time-series latency (top) + per-phase average bar chart (bottom).

Reads the CSVs produced by the `adaptive_scheduling` integration test
(`crates/simulator/tests/adaptive_scheduling.rs`) and compares the adaptive
leader scheduler against fixed leader counts of 1 and 5.

Usage:
  python3 scripts/plot_adaptive_scheduling_normalized.py --protocol mysticeti
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


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


def add_phase_shading(ax, t1, t2, t_max, labels=True):
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
        ax.text((t1 + t2) / 2,   1.07, "high latency", transform=xform,
                ha="center", fontsize=9, color="firebrick")
        ax.text((t2 + t_max) / 2, 1.07, "healthy",     transform=xform,
                ha="center", fontsize=9, color="darkgreen")


def plot(series, output):
    """series: list of (label, timeseries_records, phase_means, t1, t2)"""
    fig, (ax_ts, ax_bar) = plt.subplots(
        2, 1, figsize=(11, 9),
        gridspec_kw={"height_ratios": [2, 1.2]},
    )

    colors  = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    markers = ["o", "s", "^", "D"]

    # Top: time-series
    all_times = [r[0] for _, recs, _, _, _ in series for r in recs]
    t_max = max(all_times) if all_times else 1
    t1 = next((t1 for _, _, _, t1, _ in series if t1 is not None), None)
    t2 = next((t2 for _, _, _, _, t2 in series if t2 is not None), None)

    add_phase_shading(ax_ts, t1, t2, t_max, labels=True)

    for idx, (label, ts_recs, _, _, _) in enumerate(series):
        times = [r[0] for r in ts_recs]
        avgs  = [r[1] for r in ts_recs]
        ax_ts.plot(times, avgs, label=label, color=colors[idx % len(colors)],
                   linewidth=1.2, marker=markers[idx % len(markers)], markersize=2.5)

    ax_ts.set_ylabel("Latency (ms)", fontweight="bold")
    ax_ts.set_xlim(0, t_max)
    ax_ts.set_ylim(bottom=0)
    ax_ts.legend(loc="upper left", fontsize=8)
    ax_ts.grid(True, alpha=0.3)

    # Bottom: per-phase bar chart
    phase_labels = ["Phase 1\n(healthy)", "Phase 2\n(high latency)", "Phase 3\n(healthy)"]
    n_series = len(series)
    x = np.arange(len(phase_labels))
    width = 0.22
    offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * width

    for idx, (label, _, means, _, _) in enumerate(series):
        heights = [m if m is not None else 0 for m in means]
        bars = ax_bar.bar(x + offsets[idx], heights, width, label=label,
                          color=colors[idx % len(colors)], alpha=0.85)
        for bar, val in zip(bars, means):
            if val is not None:
                ax_bar.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.02,
                            f"{val:.0f}ms", ha="center", va="bottom", fontsize=7)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(phase_labels)
    ax_bar.set_ylabel("Avg latency (ms)", fontweight="bold")
    ax_bar.set_ylim(bottom=0)
    ax_bar.legend(fontsize=8)
    ax_bar.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True,
                        help="protocol folder under the results root "
                             "(e.g. mysticeti, blue-bottle, nemo-nemo, "
                             "mahi-mahi-4, mahi-mahi-5)")
    parser.add_argument("--results-root", default="crates/simulator/results",
                        help="base directory holding the per-protocol result folders")
    parser.add_argument("--output", default=None,
                        help="output PNG path "
                             "(default: <results-root>/<protocol>/<protocol>.png)")
    args = parser.parse_args()

    proto_dir = os.path.join(args.results_root, args.protocol)
    output = args.output or os.path.join(proto_dir, f"{args.protocol}.png")

    series = []
    for variant, label in [
        ("adaptive", "adaptive scheduler"),
        ("fixed_1",  "fixed leaders (n=1)"),
        ("fixed_5",  "fixed leaders (n=5)"),
    ]:
        dirpath = os.path.join(proto_dir, variant)
        if os.path.isdir(dirpath) and any(f.endswith(".csv") for f in os.listdir(dirpath)):
            data, t1, t2 = load_csv_dir(dirpath)
            n_runs = sum(1 for f in os.listdir(dirpath) if f.endswith(".csv"))
            print(f"Loaded {n_runs} run(s) for '{label}'")
            series.append((label, aggregate_timeseries(data), phase_means(data), t1, t2))
        else:
            print(f"Warning: no CSVs found in {dirpath}, skipping '{label}'")

    if not series:
        print(f"No data found for protocol '{args.protocol}'. Run the experiments "
              f"first (scripts/run_protocol.sh {args.protocol}).")
        return

    os.makedirs(os.path.dirname(output), exist_ok=True)
    plot(series, output)


if __name__ == "__main__":
    main()
