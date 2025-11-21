"""
CNT Transform → R(t)
- Band-pass (or wavelet) the multivariate series into M bands.
- Compute analytic signal phases θ_{m,k}(t) via Hilbert transform.
- Per band, compute order parameter r_m(t) = |(1/N) Σ_k exp(i θ_{m,k}(t))|.
- Combine across bands using fixed weights to get R(t) ∈ [0,1].
- Smooth + hysteresis to produce stable alarms.
"""
from dataclasses import dataclass
import numpy as np
import scipy.signal as sps

@dataclass
class CNTConfig:
    fs_hz: float
    bands: list
    weights: list
    smooth_tau_s: float
    hysteresis_on: float
    hysteresis_off: float
    min_on_s: float

def _bandpass(x, fs, f_lo, f_hi):
    # simple zero-phase IIR bandpass
    sos = sps.butter(4, [f_lo/(fs/2), f_hi/(fs/2)], btype="bandpass", output="sos")
    return sps.sosfiltfilt(sos, x, axis=0)

def _analytic_phase(x):
    z = sps.hilbert(x, axis=0)
    return np.angle(z)

def _order_parameter(phases):
    # phases: [T, N]
    v = np.exp(1j*phases)
    r = np.abs(v.mean(axis=1))
    return r  # [T]

def _ew_smooth(x, tau_s, fs):
    if tau_s <= 0:
        return x
    alpha = 1 - np.exp(-1/(tau_s*fs))
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = (1-alpha)*y[t-1] + alpha*x[t]
    return y

def compute_R(X, cfg: CNTConfig):
    """
    X: np.ndarray [T, N] multivariate time series
    cfg: CNTConfig
    Returns: R(t) ∈ [0,1] as np.ndarray [T]
    """
    fs = cfg.fs_hz
    bands = np.array(cfg.bands, dtype=float)
    assert len(cfg.weights) == len(bands), "weights must match bands"
    # bands are edges: interpret as center freqs with ×3 bandwidth if len==len(weights)
    # or as explicit edge pairs if even length. Here we assume center freqs and build edges.
    rs = []
    for f in bands:
        f_lo = max(1e-6, f/np.sqrt(3))
        f_hi = min(0.49*fs, f*np.sqrt(3))
        Xb = _bandpass(X, fs, f_lo, f_hi)
        phases = _analytic_phase(Xb)
        r = _order_parameter(phases)
        rs.append(r)
    R = np.average(np.stack(rs, axis=1), axis=1, weights=np.array(cfg.weights))
    R = _ew_smooth(R, cfg.smooth_tau_s, cfg.fs_hz)
    return np.clip(R, 0.0, 1.0)
