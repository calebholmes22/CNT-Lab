
"""
CNT Transform Probe (v0.2)
A micro‑module to "transform first, solve second": try linearizing transforms and score the nonlinearity drop.

Implements:
  - Cole–Hopf for 1D viscous Burgers:  u_t + u u_x = nu u_xx
  - Fourier domain scaffold (for LTI operators)
  - Kelvin inversion scaffold in 2D (geometry transform only)

Scoring idea (field-only, no PDE object):
  Given u(x), estimate nonlinearity ~ || u * u_x ||_2.
  After a transform T, map back if applicable and recompute.
  Report Δ = before − after.

Author: CNT Lab (Telos × Aetheron)
"""
import numpy as np

def grad_1d(u, dx):
    g = np.zeros_like(u)
    g[1:-1] = (u[2:] - u[:-2])/(2*dx)
    g[0] = (u[1]-u[0])/dx
    g[-1] = (u[-1]-u[-2])/dx
    return g

def lap_1d(u, dx):
    L = np.zeros_like(u)
    L[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2])/(dx*dx)
    L[0] = L[1]
    L[-1] = L[-2]
    return L

# ---------- Cole–Hopf ----------
def cole_hopf_step_heat(phi, nu, dx, dt):
    # Heat eq: phi_t = nu phi_xx (explicit)
    lap = lap_1d(phi, dx)
    return phi + dt*nu*lap

def u_from_phi(phi, nu, dx):
    # u = -2 nu (phi_x / phi)
    gx = grad_1d(phi, dx)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = -2.0*nu * (gx / np.maximum(phi, 1e-12))
    return u

def burgers_step(u, nu, dx, dt):
    # Explicit scheme (upwind-ish): u_t = nu u_xx - u u_x
    ux = grad_1d(u, dx)
    lap = lap_1d(u, dx)
    return u + dt*(nu*lap - u*ux)

def cole_hopf_evolve(u0, nu, dx, dt, steps):
    # Initialize phi from u0 by integrating u = -2 nu (phi_x/phi) => log phi' = -(1/(2nu)) * integral u dx
    x = np.arange(len(u0))*dx
    # crude integral for seed
    integ = np.cumsum(u0)*dx
    integ_centered = integ - integ.mean()
    phi0 = np.exp(-integ_centered/(2*nu))
    phi = phi0.copy()
    for _ in range(steps):
        phi = cole_hopf_step_heat(phi, nu, dx, dt)
        m = np.mean(phi)
        if not np.isfinite(m) or m == 0:
            m = 1.0
        phi = phi / m  # keep scale stable
        phi = np.maximum(phi, 1e-6)
    u = u_from_phi(phi, nu, dx)
    return u, phi

def burgers_evolve(u0, nu, dx, dt, steps):
    u = u0.copy()
    for _ in range(steps):
        u = burgers_step(u, nu, dx, dt)
    return u

def nonlin_score(u, dx):
    ux = grad_1d(u, dx)
    return float(np.sqrt(np.mean((u*ux)**2)))

# ---------- Fourier scaffold ----------
def fourier_linearize(u):
    # Returns FFT and a callable inverse
    U = np.fft.rfft(u)
    def inv(V): 
        return np.fft.irfft(V, n=u.size)
    return U, inv

# ---------- Kelvin inversion (2D) scaffold ----------
def kelvin_inversion(xy, eps=1e-12):
    # xy: (..., 2), maps (x,y) -> (x,y)/r^2
    r2 = np.sum(xy**2, axis=-1, keepdims=True)
    r2 = np.maximum(r2, eps)
    return xy / r2

# ---------- Probe orchestration ----------
def probe_transforms(u0, nu, dx, dt, steps):
    baseline = nonlin_score(u0, dx)
    out = {"baseline": baseline, "transforms": {}}

    # Cole–Hopf evolve then score (map result's score vs initial for reduction sense)
    u_ch, _ = cole_hopf_evolve(u0, nu, dx, dt, steps)
    out["transforms"]["cole_hopf"] = {
        "score_after": nonlin_score(u_ch, dx),
        "u_after": u_ch.copy()
    }

    # Fourier: zero out high frequencies as a crude "linearize" proxy
    U, inv = fourier_linearize(u0)
    k = np.arange(U.size)
    cutoff = max(1, U.size//10)
    Uf = U.copy()
    Uf[cutoff:] = 0
    u_f = inv(Uf)
    out["transforms"]["fourier_lowpass"] = {
        "score_after": nonlin_score(u_f, dx),
        "u_after": u_f.copy(),
        "cutoff_bins": int(cutoff)
    }
    return out
