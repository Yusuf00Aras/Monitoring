import csv
import json

def load_test_data(path):
    rows = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "datetime": row["datetime"],
                "module": row["module"],
                "tags": json.loads(row["tags"]) if row["tags"] else {},
                "features": json.loads(row["metrics"])
            })
    print(rows[0])
    return rows

def aggregate_features(data):
    aggregated_data = []
    i = 0
    while i <= 11:
        aggregated_data.append({
            "datetime": data[i]["datetime"],
            "module": data[i]["module"],
            "tags": data[i]["tags"],
            "features": {
                "cpu_user_pct": data[i]["features"]["cpu_user_pct"],
                "cpu_system_pct": data[i]["features"]["cpu_system_pct"],
                "cpu_iowait_pct": data[i]["features"]["cpu_iowait_pct"],
                "cpu_switches": data[i]["features"]["cpu_switches"],
                "cpu_interrupts": data[i]["features"]["cpu_interrupts"],
                "mem_util_pct": data[i]["features"]["mem_util_pct"],
                "mem_committed_as_kbytes": data[i]["features"]["mem_committed_as_kbytes"],
                "sys_load_avg_1": data[i]["features"]["sys_load_avg_1"],
                "sys_load_avg_15": data[i]["features"]["sys_load_avg_15"],
                "sys_proc_count": data[i]["features"]["sys_proc_count"],
                "sys_swap_used_pct": data[i]["features"]["sys_swap_used_pct"]
            }
        })
        i += 12
    return aggregated_data

def extract_important_features(data):
    important_features = []
    for row in data:
        important_features.append([
            row["features"]["cpu_user_pct"],
            row["features"]["cpu_system_pct"],
            row["features"]["cpu_iowait_pct"],
            row["features"]["cpu_switches"],
            row["features"]["cpu_interrupts"],
            row["features"]["mem_util_pct"],
            row["features"]["mem_committed_as_kbytes"],
            row["features"]["sys_load_avg_1"],
            row["features"]["sys_load_avg_15"],
            row["features"]["sys_proc_count"],
            row["features"]["sys_swap_used_pct"]
        ])
    return important_features

load_test_data('./Test_Data/data-1786192670480.csv')