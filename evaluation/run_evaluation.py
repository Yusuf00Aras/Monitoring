"""run_evaluation.py -- main comparative evaluation script.

Implements the full evaluation pipeline:

  1. Load telemetry data from one server (CSV).
  2. Split into a clean baseline window and an evaluation window.
  3. Calibrate EWMA, OOL and MD to the same false alarm rate on baseline.
  4. Run spike, drift and correlation-break as three INDEPENDENT
     evaluations (Option A): each starts from the same clean baseline,
     with freshly initialized method state, and only that scenario's
     injected evaluation data -- no state carries over between scenarios.
  5. Combine the three per-scenario results per method into one report.
  6. Print a comparison report and save it to CSV.

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
from injection import inject_spike, inject_drift, inject_correlation_break, \
    dict_to_vector_layout, _default_injection_config
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
# Scenario name -> injection function. Each is run independently: same
# clean baseline, fresh method state, only this scenario injected.
######

SCENARIO_INJECTORS = {
    "spike": inject_spike,
    "drift": inject_drift,
    "correlation_break": inject_correlation_break,
}


######
# Combine three independent single-scenario reports (one per scenario)
# for one method into a single report with the same shape evaluate()
# would produce for 3 intervals, so downstream printing/CSV export
# doesn't need to change.
######

def _combine_scenario_reports(method_name, scenario_reports):
    scenario_results = []
    all_detected = []
    total_false_alarms = 0
    total_minutes = 0

    for r in scenario_reports:
        scenario_results.extend(r['scenario_results'])
        all_detected.extend(r['all_detected'])
        total_false_alarms += r['false_alarms']
        total_minutes += r['total_minutes']

    total_injections = len(scenario_results)
    detected_count = sum(1 for sr in scenario_results if sr['detected'])
    detection_rate = detected_count / total_injections if total_injections else 0.0

    delays = [sr['delay_minutes'] for sr in scenario_results if sr['detected']]
    mean_delay = sum(delays) / len(delays) if delays else None

    hours = total_minutes / 60.0 if total_minutes > 0 else 1.0
    false_alarm_rate = total_false_alarms / hours

    return {
        'method': method_name,
        'total_injections': total_injections,
        'detected_count': detected_count,
        'detection_rate': detection_rate,
        'mean_delay_minutes': mean_delay,
        'false_alarms': total_false_alarms,
        'false_alarm_rate_per_hour': false_alarm_rate,
        'total_minutes': total_minutes,
        'scenario_results': scenario_results,
        'all_detected': all_detected,
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

    # 4. Run spike, drift and correlation-break as three INDEPENDENT
    #    evaluations. Each uses the same clean baseline, fresh method
    #    state, and only its own injected scenario -- no state or
    #    injected values leak between scenarios.
    print("\n[4] Running each scenario independently (Option A)")
    cfg = _default_injection_config(len(split['eval_timestamps']))
    eval_start = split['split_index']

    per_method_scenario_reports = {m: [] for m in list_methods()}

    for scenario_name, inject_fn in SCENARIO_INJECTORS.items():
        injection_result = inject_fn(split['eval_features'], split['eval_timestamps'],
                                     **cfg[scenario_name])
        injected_eval = injection_result['features']
        intervals = injection_result['intervals']

        full_features = merge_windows(split['baseline_features'], injected_eval, layout="dict")
        full_timestamps = list(split['baseline_timestamps']) + list(split['eval_timestamps'])

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
        iv0 = shifted_intervals[0]
        print(f"    [{scenario_name}] idx {iv0['start_idx']}-{iv0['end_idx']}  {iv0['description']}")

        for method in list_methods():
            thr = thresholds[method]
            if method == "MD":
                raw_anomalies = run_method(
                    method, full_features, full_timestamps, threshold=thr,
                    baseline_features=split['baseline_features'],
                    baseline_timestamps=split['baseline_timestamps'],
                )
            else:
                raw_anomalies = run_method(method, full_features, full_timestamps, threshold=thr)

            report = evaluate_raw(
                method_name=method,
                raw_anomalies=raw_anomalies,
                timestamps=full_timestamps,
                intervals=shifted_intervals,
                total_minutes=len(split['eval_timestamps']),
            )
            per_method_scenario_reports[method].append(report)

    # 5. Combine the three independent scenario reports per method
    reports = {
        method: _combine_scenario_reports(method, per_method_scenario_reports[method])
        for method in list_methods()
    }

    # 6. Print comparison report
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    for method, report in reports.items():
        print()
        print(format_report(report))

    # 7. Save results to CSV
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
