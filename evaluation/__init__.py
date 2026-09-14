"""Evaluation framework for the comparative anomaly detection PoC.

Modules
-------
injection.py      — Python mask: injects spike, drift, correlation break
metrics.py        — detection rate, detection delay, false alarm rate
calibration.py    — calibrate all methods to the same false alarm rate
window_split.py   — split data into baseline / evaluation windows
method_loader.py  — unified interface to run EWMA, MD, or OOL in batch
run_evaluation.py — main entry point that runs the full pipeline
"""
