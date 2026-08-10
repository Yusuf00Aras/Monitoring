import csv
import json
from collections import defaultdict

def extract_important_features(path):
    # defaultdict to aggregate metrics for each minute, makes the handling easier with dictionary keys
    minute_data = defaultdict(dict)
    

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            dt = row["datetime"]
            module = row["module"]
            
            # Skip empty metrics
            if not row["metrics"]:
                continue
                
            metrics = json.loads(row["metrics"])

            # Sort the metrics into the single minute bucket based on the module
            # adds relevant cpu metrics to the current minute's dictionary
            if module == "zbx_cpu":
                minute_data[dt]["cpu_user_pct"] = metrics.get("user.pct")
                minute_data[dt]["cpu_system_pct"] = metrics.get("system.pct")
                minute_data[dt]["cpu_iowait_pct"] = metrics.get("iowait.pct")
                minute_data[dt]["cpu_switches"] = metrics.get("switches")
                minute_data[dt]["cpu_interrupts"] = metrics.get("interrupts")

            # adds relevant memory metrics to the current minute's dictionary
            elif module == "zbx_memory":
                minute_data[dt]["mem_util_pct"] = metrics.get("util.pct")
                minute_data[dt]["mem_committed_as_kbytes"] = metrics.get("committed_as.kbytes")

            # adds relevant system metrics to the current minute's dictionary
            elif module == "zbx_system":
                minute_data[dt]["sys_load_avg_1"] = metrics.get("load_avg_1")
                minute_data[dt]["sys_load_avg_15"] = metrics.get("load_avg_15")
                minute_data[dt]["sys_proc_count"] = metrics.get("proc_count")
                minute_data[dt]["sys_swap_used_pct"] = metrics.get("swap_used.pct")

    final_features = []
    for dt in sorted(minute_data.keys()):
        # print(dt) debugging
        features = minute_data[dt]
        
        # This creates exactly one array per minute
        # takes all the values from the dictionary and puts them into a list in the order specified
        # makes the handling easier later by making sure each value is a feature (important for mahalanobis distance)
        try:
            # skipping NaN values, if any of the features are missing for a minute, we skip that minute
            if features.get("cpu_user_pct") is None or features.get("cpu_system_pct") is None or features.get("cpu_iowait_pct") is None or features.get("cpu_switches") is None or features.get("cpu_interrupts") is None or features.get("mem_util_pct") is None or features.get("mem_committed_as_kbytes") is None or features.get("sys_load_avg_1") is None or features.get("sys_load_avg_15") is None or features.get("sys_proc_count") is None or features.get("sys_swap_used_pct") is None:
                print(f"Missing feature for datetime {dt}, skipping this minute.")
                continue
            minute_array = [
                features.get("cpu_user_pct"),
                features.get("cpu_system_pct"),
                features.get("cpu_iowait_pct"),
                features.get("cpu_switches"),
                features.get("cpu_interrupts"),
                features.get("mem_util_pct"),
                features.get("mem_committed_as_kbytes"),
                features.get("sys_load_avg_1"),
                features.get("sys_load_avg_15"),
                features.get("sys_proc_count"),
                features.get("sys_swap_used_pct")
            ]
            final_features.append(minute_array)
        except KeyError as e:
            print(f"Missing feature for datetime {dt}: {e}")
            continue

    return final_features

# Execute
# debugging aggregated_arrays = extract_important_features('./Test_Data/data-1786192670480.csv')
# print(aggregated_arrays[0])