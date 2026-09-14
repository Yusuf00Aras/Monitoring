"""run_evaluation.py — main comparative evaluation script.

Implements the full evaluation pipeline:

  1. Load telemetry data from one server (CSV).
  2. Split into a clean baseline window and an evaluation window.
  3. Calibrate EWMA, MD and OOL to the same false alarm rate on baseline.
  4. Inject the three scenarios into the evaluation window (Python mask).
  5. Run all three methods on the injected evaluation window.
  6. Measure detection rate, detection delay and false alarm rate.
  7. Print a comparison report and save it to CSV.

Usage
-----
    python evaluation/run_evaluation.py [--data PATH] [--baseline-ratio 0.5]
            [--target-fa 0] [--no-calibrate]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

from EWMA.cleaning_utils import extract_all_features
from injection import inject_all, dict_to_vector_layout
from window_split import split_windows, merge_windows
from calibration import calibrate_all
from method_loader import run_method, list_methods
from metrics import evaluate_raw, format_report


######
# Default thresholds (used when --no-calibrate is passed)
######

DEFAULT_THRESHOLDS = {
    "EWMA": 3.0,
    "OOL": 3.0,
    "MD": 7.0,
}


######
# Run the full comparative evaluation, return dict {method: report}
######

def run_evaluation(data_path, baseline_ratio=0.5, target_false_alarms=0,
                   calibrate=True, output_dir=None):
    print("=" * 70)
    print("  COMPARATIVE ANOMALY DETECTION EVALUATION")
    print("=" * 70)

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
    print(f"    Baseline window : {len(split['baseline_timestamps'])} minutes")
    print(f"    Eval window     : {len(split['eval_timestamps'])} minutes")

    # 3. Calibrate thresholds
    if calibrate:
        print(f"\n[3] Calibrating thresholds (target false alarms = {target_false_alarms})")
        baseline_vectors, _ = dict_to_vector_layout(
            split['baseline_features'], split['baseline_timestamps'])
        thresholds = calibrate_all(
            split['baseline_features'],
            baseline_vectors,
            split['baseline_timestamps'],
            target_false_alarms=target_false_alarms,
        )
    else:
        print("\n[3] Skipping calibration, using default thresholds")
        thresholds = dict(DEFAULT_THRESHOLDS)
    print(f"    Thresholds: {thresholds}")

    # 4. Inject anomalies into the evaluation window
    print("\n[4] Injecting anomalies (Python mask)")
    injection_result = inject_all(split['eval_features'], split['eval_timestamps'])
    injected_eval = injection_result['features']
    intervals = injection_result['intervals']
    for iv in intervals:
        print(f"    [{iv['scenario']}] idx {iv['start_idx']}–{iv['end_idx']}  {iv['description']}")

    # 5. Merge baseline + injected eval so each method sees the full stream
    full_features = merge_windows(split['baseline_features'], injected_eval, layout="dict")
    full_timestamps = list(split['baseline_timestamps']) + list(split['eval_timestamps'])

    # 6. Run all three methods
    print("\n[5] Running methods on injected data")
    reports = {}
    eval_start = split['split_index']
    # Shift interval indices: they are relative to the eval window, but
    # detected anomalies are indexed in the full stream.
    shifted_intervals = [
        {
            'start_idx': iv['start_idx'] + eval_start,
            'end_idx': iv['end_idx'] + eval_start,
            'scenario': iv['scenario'],
            'feature': iv['feature'],
            'description': iv['description'],
        }
        for iv in intervals
    ]

    for method in list_methods():
        print(f"    Running {method} ...")
        thr = thresholds[method]
        raw_anomalies = run_method(method, full_features, full_timestamps, threshold=thr)
        report = evaluate_raw(
            method_name=method,
            raw_anomalies=raw_anomalies,
            timestamps=full_timestamps,
            intervals=shifted_intervals,
            total_minutes=len(split['eval_timestamps']),
        )
        reports[method] = report

    # 7. Print comparison report
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    for method, report in reports.items():
        print()
        print(format_report(report))

    # 8. Save results to CSV
    if output_dir is None:
        output_dir = os.path.join(_PROJECT_ROOT, "evaluation", "RESULTS")
    os.makedirs(output_dir, exist_ok=True)
    _save_reports_csv(reports, output_dir)
    print(f"\nResults saved to {output_dir}")

    return reports


######
# Save the evaluation reports to summary and per-scenario CSV files
######

def _save_reports_csv(reports, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = os.path.join(output_dir, f"summary_{timestamp}.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "total_injections", "detected_count",
                     "detection_rate", "mean_delay_minutes",
                     "false_alarms", "false_alarm_rate_per_hour"])
        for method, r in reports.items():
            w.writerow([r['method'], r['total_injections'], r['detected_count'],
                        f"{r['detection_rate']:.4f}",
                        r['mean_delay_minutes'] if r['mean_delay_minutes'] is not None else "",
                        r['false_alarms'],
                        f"{r['false_alarm_rate_per_hour']:.4f}"])

    detail_path = os.path.join(output_dir, f"scenarios_{timestamp}.csv")
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "scenario", "feature", "detected",
                     "delay_minutes", "description"])
        for method, r in reports.items():
            for sr in r['scenario_results']:
                iv = sr['interval']
                w.writerow([r['method'], iv['scenario'], iv['feature'],
                            sr['detected'],
                            sr['delay_minutes'] if sr['delay_minutes'] is not None else "",
                            iv['description']])


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
    parser.add_argument("--target-fa", type=int, default=0,
                        help="Target false alarm count on the baseline for calibration (default: 0)")
    parser.add_argument("--no-calibrate", action="store_true",
                        help="Skip calibration, use default thresholds")
    args = parser.parse_args()

    if args.data is None:
        args.data = os.path.join(_PROJECT_ROOT, "Test_Data", "raw_data.csv")

    run_evaluation(
        data_path=args.data,
        baseline_ratio=args.baseline_ratio,
        target_false_alarms=args.target_fa,
        calibrate=not args.no_calibrate,
    )


if __name__ == "__main__":
    main()
