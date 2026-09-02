# Cleaning the data based on the method
import csv
import json
from collections import defaultdict


FEATURE_KEYS = {
    "cpu_user_pct": ("zbx_cpu", "user.pct"),
    "cpu_system_pct": ("zbx_cpu", "system.pct"),
    "cpu_iowait_pct": ("zbx_cpu", "iowait.pct"),
    "cpu_switches": ("zbx_cpu", "switches"),
    "cpu_interrupts": ("zbx_cpu", "interrupts"),
    "mem_util_pct": ("zbx_memory", "util.pct"),
    "mem_committed_as_kbytes": ("zbx_memory", "committed_as.kbytes"),
    "sys_load_avg_1": ("zbx_system", "load_avg_1"),
    "sys_load_avg_15": ("zbx_system", "load_avg_15"),
    "sys_proc_count": ("zbx_system", "proc_count"),
    "sys_swap_used_pct": ("zbx_system", "swap_used.pct"),
}


def extract_minute_data(path):
    # Collect all module metrics grouped by minute timestamp.
    minute_data = defaultdict(dict)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            dt = row["datetime"]
            module = row["module"]

            if not row["metrics"]:
                continue


            metrics = json.loads(row["metrics"])
            if metrics is None:
                continue

            for feature_name, (expected_module, metric_key) in FEATURE_KEYS.items():
                if module == expected_module:
                    minute_data[dt][feature_name] = metrics.get(metric_key)

    return minute_data


def extract_all_features(path):
    """Extract all features and return as a dict of equal-length lists."""
    minute_data = extract_minute_data(path)
    result = {feature_name: [] for feature_name in FEATURE_KEYS.keys()}
    timestamps = []

    for dt in sorted(minute_data.keys()):
        # skip the whole minute if any feature is missing, so all lists stay aligned
        if any(minute_data[dt].get(feature_name) is None for feature_name in FEATURE_KEYS.keys()):
            continue
        timestamps.append(dt)
        for feature_name in FEATURE_KEYS.keys():
            result[feature_name].append(minute_data[dt][feature_name])

    return result, timestamps
