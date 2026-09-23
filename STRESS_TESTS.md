# stress-ng end-to-end tests

End-to-end check of the monitoring pipeline (Zabbix agent → Postgres DB →
EWMA / MD / OOL monitors) on the dev-server, run on 22.09.2026 from 14:20.
The tests are not used to compare the methods (the exact start of the load
cannot be aligned with the Zabbix polling second); the comparison is done
with the Python mask in `evaluation/`.

The tests were run one after another with a 5-minute break in between:

```bash
# Test 1: One CPU worker — partial CPU pressure
stress-ng --cpu 1 --timeout 5m --metrics-brief

# Wait 5 minutes

# Test 2: Two CPU workers — full CPU pressure
stress-ng --cpu 2 --timeout 5m --metrics-brief

# Wait 5 minutes

# Test 3: Memory pressure
stress-ng --vm 1 --vm-bytes 60% --timeout 5m --metrics-brief

# Wait 5 minutes

# Test 4: I/O pressure
stress-ng --io 2 --timeout 5m --metrics-brief
```
