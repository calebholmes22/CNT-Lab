"""
Scorecard: lead-time, FAR/hr, AUC-PR, calibration — plus null/placebo tests.
This is a scaffold; fill the dataset loaders locally.
"""
import json, hashlib, time, math
from dataclasses import dataclass
import numpy as np

def median_lead_time(pred_times, event_times, horizon_s):
    """
    pred_times: list of timestamps where alarm turns ON
    event_times: list of event start timestamps
    Return median lead (seconds) for events captured inside horizon.
    """
    leads = []
    j = 0
    event_times = sorted(event_times)
    for p in sorted(pred_times):
        while j < len(event_times) and event_times[j] < p:
            j += 1
        if j < len(event_times):
            dt = event_times[j] - p
            if 0 <= dt <= horizon_s:
                leads.append(dt)
    if not leads:
        return None
    return float(np.median(leads))

def far_per_hour(alarm_on_mask, fs_hz):
    # count ON transitions per hour
    on_edges = (alarm_on_mask.astype(int)[1:] > alarm_on_mask.astype(int)[:-1]).sum()
    hours = len(alarm_on_mask)/fs_hz/3600.0
    return on_edges/max(hours,1e-9)

def sha256_file(path):
    m = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            m.update(chunk)
    return m.hexdigest()
