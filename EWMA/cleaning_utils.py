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


def _extract_minute_data(path):
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

            for feature_name, (expected_module, metric_key) in FEATURE_KEYS.items():
                if module == expected_module:
                    minute_data[dt][feature_name] = metrics.get(metric_key)

    return minute_data


def extract_cpu_user_pct(path):
    minute_data = _extract_minute_data(path)
    return [minute_data[dt].get("cpu_user_pct") for dt in sorted(minute_data.keys()) if minute_data[dt].get("cpu_user_pct") is not None]


def extract_cpu_system_pct(path):
    minute_data = _extract_minute_data(path)
    return [minute_data[dt].get("cpu_system_pct") for dt in sorted(minute_data.keys()) if minute_data[dt].get("cpu_system_pct") is not None]


def extract_cpu_iowait_pct(path):
    minute_data = _extract_minute_data(path)
    return [minute_data[dt].get("cpu_iowait_pct") for dt in sorted(minute_data.keys()) if minute_data[dt].get("cpu_iowait_pct") is not None]


def extract_cpu_switches(path):
    minute_data = _extract_minute_data(path)
    return [minute_data[dt].get("cpu_switches") for dt in sorted(minute_data.keys()) if minute_data[dt].get("cpu_switches") is not None]


def extract_cpu_interrupts(path):
    minute_data = _extract_minute_data(path)
    return [minute_data[dt].get("cpu_interrupts") for dt in sorted(minute_data.keys()) if minute_data[dt].get("cpu_interrupts") is not None]


def extract_mem_util_pct(path):
    minute_data = _extract_minute_data(path)
    return [minute_data[dt].get("mem_util_pct") for dt in sorted(minute_data.keys()) if minute_data[dt].get("mem_util_pct") is not None]


def extract_mem_committed_as_kbytes(path):
    minute_data = _extract_minute_data(path)
    return [minute_data[dt].get("mem_committed_as_kbytes") for dt in sorted(minute_data.keys()) if minute_data[dt].get("mem_committed_as_kbytes") is not None]


def extract_sys_load_avg_1(path):
    minute_data = _extract_minute_data(path)
    return [minute_data[dt].get("sys_load_avg_1") for dt in sorted(minute_data.keys()) if minute_data[dt].get("sys_load_avg_1") is not None]


def extract_sys_load_avg_15(path):
    minute_data = _extract_minute_data(path)
    return [minute_data[dt].get("sys_load_avg_15") for dt in sorted(minute_data.keys()) if minute_data[dt].get("sys_load_avg_15") is not None]


def extract_sys_proc_count(path):
    minute_data = _extract_minute_data(path)
    return [minute_data[dt].get("sys_proc_count") for dt in sorted(minute_data.keys()) if minute_data[dt].get("sys_proc_count") is not None]


def extract_sys_swap_used_pct(path):
    minute_data = _extract_minute_data(path)
    return [minute_data[dt].get("sys_swap_used_pct") for dt in sorted(minute_data.keys()) if minute_data[dt].get("sys_swap_used_pct") is not None]

# Execute
# debugging aggregated_arrays = extract_important_features('./Test_Data/data-1786192670480.csv')
# print(aggregated_arrays[0])