"""run_evaluation.py -- main comparative evaluation script.

Implements the full evaluation pipeline:

  1. Load telemetry data from one server (CSV).
  2. Split into a clean baseline window and an evaluation window.
  3. Calibrate EWMA and MD to the same false alarm rate on baseline.
     OOL is NOT calibrated -- its limits are static. The calibration
     target is the number of alarms OOL produces on the baseline window,
     so all three methods end up with the same baseline false alarm count.
  4. For each of `repetitions` randomized runs:
       - randomly pick injection positions inside the evaluation window
         (kept away from the edges by EDGE_MARGIN minutes),
       - inject spike, drift and correlation-break,
       - run EWMA, MD and OOL on baseline + injected eval,
       - score detections / false alarms against the ground-truth intervals.
  5. Aggregate the per-run results into mean +/- std across repetitions.
  6. Print a comparison report and save it to CSV.

Usage
-----
    python evaluation/run_evaluation.py [--data PATH] [--baseline-ratio 0.5]
            [--repetitions 30] [--seed 42] [--target-fa ool]
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from EWMA.cleaning_utils import extract_all_features
from injection import inject_spike, inject_drift, inject_correlation_break, \
    dict_to_vector_layout
from window_split import split_windows, merge_windows
from calibration import calibrate_ewma, calibrate_md
from method_loader import run_method, list_methods
from metrics import evaluate_raw


######
# Scenario name -> injection function.
######

SCENARIO_INJECTORS = {
    "spike": inject_spike,
    "drift": inject_drift,
    "correlation_break": inject_correlation_break,
}

######
# Feature(s) each scenario injects into. Used by metrics._matches_feature
# to decide whether an alarm on a given feature counts as a detection.
######

SCENARIO_FEATURES = {
    "spike": ["cpu_user_pct"],
    "drift": ["mem_util_pct"],
    "correlation_break": ["multi"],
}

# Keep injection start points at least this many minutes from the
# edges of the evaluation window so the anomaly has room to develop.
EDGE_MARGIN = 30


######
# Run one repetition: inject all three scenarios at randomized positions,
# run every method, return {method: {scenario: report}}.
######

def _run(split, thresholds, positions, cfg):
    eval_start = split['split_index']
    eval_features = split['eval_features']
    eval_timestamps = split['eval_timestamps']
    n_eval = len(eval_timestamps)

    # Apply all three injections to the same eval window copy.
    injected = {f: list(v) for f, v in eval_features.items()}
    intervals = []
    for scenario_name, inject_fn in SCENARIO_INJECTORS.items():
        params = dict(cfg[scenario_name])
        params["start_idx"] = positions[scenario_name]
        res = inject_fn(injected, eval_timestamps, **params)
        injected = res['features']
        iv = res['intervals'][0]
        iv = dict(iv)
        iv['features'] = SCENARIO_FEATURES[scenario_name]
        iv['start_idx'] += eval_start
        iv['end_idx'] += eval_start
        intervals.append(iv)

    full_features = merge_windows(split['baseline_features'], injected, layout="dict")
    full_timestamps = list(split['baseline_timestamps']) + list(eval_timestamps)

    reports = {}
    for method in list_methods():
        thr = thresholds.get(method)
        if method == "MD":
            raw = run_method(
                method, full_features, full_timestamps, threshold=thr,
                baseline_features=split['baseline_features'],
                baseline_timestamps=split['baseline_timestamps'],
            )
        else:
            raw = run_method(method, full_features, full_timestamps, threshold=thr)

        # Score each scenario interval separately so we can aggregate
        # per scenario across repetitions.
        per_scenario = {}
        for iv in intervals:
            r = evaluate_raw(
                method_name=method,
                raw_anomalies=raw,
                timestamps=full_timestamps,
                intervals=[iv],
                total_minutes=n_eval,
                eval_start=eval_start,
            )
            per_scenario[iv['scenario']] = r
        reports[method] = per_scenario
    return reports


######
# Aggregate per-run reports into mean +/- std per method/scenario/metric.
######

def _aggregate(all_runs):
    methods = list_methods()
    metrics_keys = ["detected_count", "detection_rate",
                    "mean_delay_minutes", "false_alarms",
                    "false_alarm_rate_per_hour"]
    summary = {}
    for method in methods:
        summary[method] = {}
        for scenario in SCENARIO_INJECTORS:
            runs = [run[method][scenario] for run in all_runs if scenario in run[method]]
            if not runs:
                continue
            agg = {}
            for k in metrics_keys:
                vals = [r[k] for r in runs if r[k] is not None]
                if not vals:
                    agg[k] = (None, None)
                else:
                    mean = sum(vals) / len(vals)
                    if len(vals) > 1:
                        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                        std = var ** 0.5
                    else:
                        std = 0.0
                    agg[k] = (mean, std)
            summary[method][scenario] = agg
    return summary


######
# Format one (mean, std) pair.
######

def _fmt(pair, pct=False, decimals=2):
    mean, std = pair
    if mean is None:
        return "N/A"
    if pct:
        return f"{mean * 100:.1f}% +/- {std * 100:.1f}"
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


######
# Run the full comparative evaluation, return {method: {scenario: agg}}.
######

def run_evaluation(data_path, baseline_ratio=0.5, repetitions=30, seed=42,
                   target_fa="ool", spike_mag=5.0, drift_mag=5.0, corr_mag=3.0,
                   output_dir=None):
    print("=" * 70)
    print("  COMPARATIVE ANOMALY DETECTION EVALUATION")
    print("=" * 70)

    rng = random.Random(seed)

    # 1. Load data
    print(f"\n[1] Loading data from {data_path}")
    features, timestamps = extract_all_features(data_path)
    n = len(timestamps)
    print(f"    {n} minutes, {len(features)} features")
    if n == 0:
        print("    ERROR: no data found. Aborting.")
        return {}

    # 2. Split into baseline / evaluation windows
    print(f"\n[2] Splitting data (baseline ratio = {baseline_ratio})")
    split = split_windows(features, timestamps, baseline_ratio)
    n_eval = len(split['eval_timestamps'])
    print(f"    Baseline window : {len(split['baseline_timestamps'])} minutes")
    print(f"    Eval window     : {n_eval} minutes")

    # 3. Calibration target: OOL's baseline alarm count (or an explicit int)
    print(f"\n[3] Calibrating EWMA and MD")
    ool_baseline_alarms = len(run_method(
        "OOL", split['baseline_features'], split['baseline_timestamps']))
    if str(target_fa).lower() == "ool":
        target = ool_baseline_alarms
    else:
        target = int(target_fa)
    print(f"    OOL baseline alarms = {ool_baseline_alarms}  -> calibration target = {target}")

    ewma_thr = calibrate_ewma(split['baseline_features'],
                              split['baseline_timestamps'], target)
    baseline_vectors, _ = dict_to_vector_layout(
        split['baseline_features'], split['baseline_timestamps'])
    md_thr = calibrate_md(baseline_vectors,
                         split['baseline_timestamps'], target)
    thresholds = {"EWMA": ewma_thr, "MD": md_thr, "OOL": None}
    print(f"    Thresholds: EWMA={ewma_thr:.4f}  MD={md_thr:.4f}  OOL=static")

    # 4. Injection config (magnitudes from CLI)
    cfg = {
        "spike": {"feature": "cpu_user_pct", "duration": 5,
                  "magnitude": spike_mag, "unit": "std"},
        "drift": {"feature": "mem_util_pct", "duration": 60,
                  "total_increase": drift_mag, "unit": "std"},
        "correlation_break": {"feature_a": "cpu_user_pct",
                              "feature_b": "sys_load_avg_1",
                              "duration": 30, "magnitude": corr_mag,
                              "unit": "std"},
    }

    # 5. Run `repetitions` randomized repetitions
    print(f"\n[4] Running {repetitions} repetitions (seed={seed})")
    all_runs = []
    for rep in range(repetitions):
        positions = {}
        for scenario_name, inject_fn in SCENARIO_INJECTORS.items():
            duration = cfg[scenario_name].get("duration", 30)
            lo = EDGE_MARGIN
            hi = max(lo + 1, n_eval - EDGE_MARGIN - duration)
            positions[scenario_name] = rng.randint(lo, hi)
        run = _run(split, thresholds, positions, cfg)
        all_runs.append(run)
        if (rep + 1) % 5 == 0 or rep == 0 or rep == repetitions - 1:
            print(f"    rep {rep + 1}/{repetitions} done")

    # 6. Aggregate
    print("\n[5] Aggregating results")
    summary = _aggregate(all_runs)

    # 7. Print report
    print("\n" + "=" * 70)
    print("  RESULTS (mean +/- std over {} repetitions)".format(repetitions))
    print("=" * 70)
    for method in list_methods():
        print(f"\n=== {method} ===")
        for scenario in SCENARIO_INJECTORS:
            agg = summary[method].get(scenario, {})
            if not agg:
                continue
            det = agg.get("detected_count", (None, None))
            rate = agg.get("detection_rate", (None, None))
            delay = agg.get("mean_delay_minutes", (None, None))
            fa = agg.get("false_alarms", (None, None))
            far = agg.get("false_alarm_rate_per_hour", (None, None))
            print(f"  [{scenario}]")
            print(f"      detected      : {_fmt(det)} / 1")
            print(f"      detection rate: {_fmt(rate, pct=True)}")
            print(f"      delay (min)   : {_fmt(delay)}")
            print(f"      false alarms  : {_fmt(fa)}")
            print(f"      FA rate (/h)  : {_fmt(far)}")

    # 8. Save to CSV
    if output_dir is None:
        output_dir = os.path.join(_PROJECT_ROOT, "evaluation", "RESULTS")
    os.makedirs(output_dir, exist_ok=True)
    _save_csv(summary, all_runs, thresholds, cfg, repetitions, seed, target, output_dir)
    print(f"\nResults saved to {output_dir}")

    return summary


######
# Save aggregated results, per-run raw results, thresholds and config to CSV.
######

def _save_csv(summary, all_runs, thresholds, cfg, repetitions, seed, target, output_dir):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-method/scenario summary (mean +/- std)
    summary_path = os.path.join(output_dir, f"scenario_summary_{ts}.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "scenario", "repetitions",
                    "detected_mean", "detected_std",
                    "detection_rate_mean", "detection_rate_std",
                    "delay_mean", "delay_std",
                    "false_alarms_mean", "false_alarms_std",
                    "fa_rate_mean", "fa_rate_std"])
        for method in list_methods():
            for scenario in SCENARIO_INJECTORS:
                agg = summary[method].get(scenario)
                if not agg:
                    continue
                row = [method, scenario, repetitions]
                for k in ["detected_count", "detection_rate",
                          "mean_delay_minutes", "false_alarms",
                          "false_alarm_rate_per_hour"]:
                    m, s = agg.get(k, (None, None))
                    row.append("" if m is None else f"{m:.4f}")
                    row.append("" if s is None else f"{s:.4f}")
                w.writerow(row)

    # Per-run raw results (one row per repetition x method x scenario)
    runs_path = os.path.join(output_dir, f"runs_{ts}.csv")
    with open(runs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["repetition", "method", "scenario",
                    "detected", "detection_rate", "delay_minutes",
                    "false_alarms", "fa_rate_per_hour"])
        for rep_idx, run in enumerate(all_runs):
            for method in list_methods():
                for scenario in SCENARIO_INJECTORS:
                    r = run[method].get(scenario)
                    if r is None:
                        continue
                    sr = r['scenario_results'][0] if r['scenario_results'] else None
                    w.writerow([
                        rep_idx, method, scenario,
                        sr['detected'] if sr else "",
                        f"{r['detection_rate']:.4f}",
                        sr['delay_minutes'] if sr and sr['delay_minutes'] is not None else "",
                        r['false_alarms'],
                        f"{r['false_alarm_rate_per_hour']:.4f}",
                    ])

    # Config / thresholds
    config_path = os.path.join(output_dir, f"config_{ts}.csv")
    with open(config_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "value"])
        w.writerow(["repetitions", repetitions])
        w.writerow(["seed", seed])
        w.writerow(["target_fa", target])
        for m, t in thresholds.items():
            w.writerow([f"threshold_{m}", "" if t is None else f"{t:.4f}"])
        for scenario, params in cfg.items():
            for k, v in params.items():
                w.writerow([f"{scenario}.{k}", v])


######
# CLI entry point
######

def main():
    parser = argparse.ArgumentParser(
        description="Comparative evaluation of EWMA, MD and OOL anomaly detection")
    parser.add_argument("--data", default=None,
                        help="Path to the CSV data file "
                             "(default: ../Test_Data/raw_data.csv)")
    parser.add_argument("--baseline-ratio", type=float, default=0.5,
                        help="Fraction of data to use as clean baseline (default: 0.5)")
    parser.add_argument("--repetitions", type=int, default=30,
                        help="Number of randomized repetitions (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for injection positions (default: 42)")
    parser.add_argument("--target-fa", default="ool",
                        help="Calibration target: 'ool' (use OOL baseline alarm "
                             "count) or an explicit integer (default: ool)")
    parser.add_argument("--spike-mag", type=float, default=5.0,
                        help="Spike magnitude in std units (default: 5.0)")
    parser.add_argument("--drift-mag", type=float, default=5.0,
                        help="Drift total increase in std units (default: 5.0)")
    parser.add_argument("--corr-mag", type=float, default=3.0,
                        help="Correlation break magnitude in std units (default: 3.0)")
    args = parser.parse_args()

    if args.data is None:
        args.data = os.path.join(_PROJECT_ROOT, "Test_Data", "raw_data.csv")

    run_evaluation(
        data_path=args.data,
        baseline_ratio=args.baseline_ratio,
        repetitions=args.repetitions,
        seed=args.seed,
        target_fa=args.target_fa,
        spike_mag=args.spike_mag,
        drift_mag=args.drift_mag,
        corr_mag=args.corr_mag,
    )


if __name__ == "__main__":
    main()
