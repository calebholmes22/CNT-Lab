
import os
import sys
import json
import glob
import argparse
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.signal import welch, csd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt

# ------------------------------
# Utility / IO
# ------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def read_channels(path_txt: str, n_fallback: int) -> List[str]:
    if os.path.exists(path_txt):
        with open(path_txt, "r") as f:
            chans = [ln.strip() for ln in f if ln.strip()]
        return chans
    return [f"ch{i}" for i in range(n_fallback)]

# ------------------------------
# Spectral Connectivity
# ------------------------------

def band_limits() -> Dict[str, Tuple[float, float]]:
    return {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }

def band_mask(fs: float, nfft: int, fmin: float, fmax: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    return (freqs >= fmin) & (freqs <= fmax)

def imag_coh_matrix(X: np.ndarray, fs: float, band: str, nperseg: int = 512, noverlap: int = 256) -> np.ndarray:
    """
    X: [n_channels, n_time]
    Returns W (n_channels x n_channels) of average imaginary coherence in the band.
    """
    n_ch, _ = X.shape
    bl = band_limits()
    if band not in bl:
        raise ValueError(f"Unknown band: {band}")
    fmin, fmax = bl[band]

    # Welch autospectra
    Sxx = np.zeros((n_ch,), dtype=object)
    Pxx = [None]*n_ch
    freqs, _ = welch(X[0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    # Compute auto-spectra for normalization
    for i in range(n_ch):
        freqs, Pxx_i = welch(X[i], fs=fs, nperseg=nperseg, noverlap=noverlap)
        Pxx[i] = Pxx_i

    band_sel = (freqs >= fmin) & (freqs <= fmax)
    W = np.zeros((n_ch, n_ch), dtype=float)

    # Cross-spectra and imag coherence
    for i in range(n_ch):
        for j in range(i+1, n_ch):
            _, Pxy = csd(X[i], X[j], fs=fs, nperseg=nperseg, noverlap=noverlap)
            # Imag coherence: |Im(Pxy)| / sqrt(Pxx * Pyy)
            num = np.abs(np.imag(Pxy[band_sel]))
            den = np.sqrt(Pxx[i][band_sel] * Pxx[j][band_sel] + 1e-12)
            ic = num / den
            val = float(np.nanmean(ic))
            W[i, j] = val
            W[j, i] = val

    # no self-edges
    np.fill_diagonal(W, 0.0)
    return W

# ------------------------------
# Spectral Clustering Core
# ------------------------------

def laplacian(W: np.ndarray, normalized: bool = True):
    d = W.sum(axis=1)
    if not normalized:
        L = np.diag(d) - W
        return L
    # Symmetric normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}
    d_safe = np.where(d <= 1e-12, 1.0, d)
    Dmh = np.diag(1.0 / np.sqrt(d_safe))
    Lsym = np.eye(W.shape[0]) - Dmh @ W @ Dmh
    return Lsym

def pick_k_by_eigengap(Lsym: np.ndarray, kmax: int = 5) -> int:
    evals, evecs = np.linalg.eigh(Lsym)
    # skip trivial 0 (smallest)
    gaps = []
    for k in range(1, min(kmax+1, len(evals)-1)):
        gaps.append((k, evals[k+1] - evals[k]))
    if not gaps:
        return 2
    # choose k with largest gap
    k_star = max(gaps, key=lambda x: x[1])[0]
    return max(2, min(k_star, kmax))

def spectral_cluster(W: np.ndarray, k: int) -> np.ndarray:
    Lsym = laplacian(W, normalized=True)
    evals, evecs = np.linalg.eigh(Lsym)
    U = evecs[:, 1:k+1]  # skip trivial eigenvector
    # Row-normalize
    U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    labels = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(U_norm)
    return labels, evals

# ------------------------------
# Consensus & Stability
# ------------------------------

def consensus_from_labels(all_labels: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple co-association consensus (majority threshold).
    Returns (consensus_labels, coassoc_matrix)
    """
    n = len(all_labels[0])
    m = len(all_labels)
    co = np.zeros((n, n), dtype=float)
    for labels in all_labels:
        for i in range(n):
            for j in range(n):
                co[i, j] += 1.0 if labels[i] == labels[j] else 0.0
    co /= m

    # Build a graph by thresholding co-association at 0.5 and take components as clusters
    thr = 0.5
    A = (co >= thr).astype(int)
    # Connected components by DFS
    visited = np.zeros(n, dtype=bool)
    cons = -1 * np.ones(n, dtype=int)
    cid = 0
    for i in range(n):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            cons[i] = cid
            while stack:
                u = stack.pop()
                for v in range(n):
                    if A[u, v] and not visited[v]:
                        visited[v] = True
                        cons[v] = cid
                        stack.append(v)
            cid += 1
    return cons, co

def ari_loso(all_labels: List[np.ndarray]) -> float:
    """LOSO median ARI vs consensus"""
    cons, _ = consensus_from_labels(all_labels)
    aris = []
    for s in range(len(all_labels)):
        leave = [lab for i, lab in enumerate(all_labels) if i != s]
        cons_leave, _ = consensus_from_labels(leave)
        aris.append(adjusted_rand_score(cons, cons_leave))
    return float(np.median(aris))

def circular_shift_null(X: np.ndarray, max_shift: int = None) -> np.ndarray:
    """
    Circularly shift each channel by a random amount to destroy phase relations.
    """
    n_ch, n_t = X.shape
    if max_shift is None:
        max_shift = n_t - 1
    Y = np.zeros_like(X)
    for c in range(n_ch):
        s = np.random.randint(0, max_shift+1)
        Y[c] = np.roll(X[c], s)
    return Y

# ------------------------------
# Plot helpers
# ------------------------------

def plot_matrix(M: np.ndarray, title: str, out_png: str):
    plt.figure()
    plt.imshow(M, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_eigs(evals: np.ndarray, title: str, out_png: str):
    plt.figure()
    plt.plot(np.arange(len(evals)), evals, marker='o')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# ------------------------------
# Main pipeline
# ------------------------------

@dataclass
class SubjectResult:
    subject: str
    band: str
    k: int
    labels: List[int]
    eigenvalues: List[float]

def run_pipeline(
    data_files: List[str],
    fs: float,
    out_root: str,
    bands: List[str],
    kmax: int = 4,
    nperseg: int = 512,
    noverlap: int = 256,
    null_perms: int = 200,
) -> Dict:

    rng = np.random.default_rng(42)
    out_fig = ensure_dir(os.path.join(out_root, "figures"))
    out_tab = ensure_dir(os.path.join(out_root, "tables"))
    out_met = ensure_dir(os.path.join(out_root, "metrics"))

    # Load all subjects
    subjects = []
    X_list = []
    chans = None
    for f in data_files:
        X = np.load(f)  # [n_ch, n_t]
        ch_txt = f.replace(".npy", ".channels.txt")
        chs = read_channels(ch_txt, X.shape[0])
        if chans is None:
            chans = chs
        subjects.append(os.path.splitext(os.path.basename(f))[0])
        X_list.append(X)

    results: Dict[str, Dict[str, SubjectResult]] = {}
    for band in bands:
        # Per subject clustering
        subj_labels = []
        eig_store = []
        for s, X in zip(subjects, X_list):
            W = imag_coh_matrix(X, fs=fs, band=band, nperseg=nperseg, noverlap=noverlap)
            Lsym = laplacian(W, normalized=True)
            k = pick_k_by_eigengap(Lsym, kmax=kmax)
            labels, evals = spectral_cluster(W, k)
            subj_labels.append(labels)
            eig_store.append(evals.tolist())

            plot_matrix(W, f"{s} {band} | iCoh", os.path.join(out_fig, f"{s}__{band}__icoherence.png"))
            plot_eigs(evals, f"{s} {band} | Laplacian eigs", os.path.join(out_fig, f"{s}__{band}__eigs.png"))

        # Consensus and LOSO
        cons, co = consensus_from_labels(subj_labels)
        plot_matrix(co, f"Co-association | {band}", os.path.join(out_fig, f"consensus__{band}__coassoc.png"))
        loso_med_ari = ari_loso(subj_labels)

        # Null test (circular shift)
        null_aris = []
        for p in range(null_perms):
            null_labels = []
            for X in X_list:
                Y = circular_shift_null(X)
                Wn = imag_coh_matrix(Y, fs=fs, band=band, nperseg=nperseg, noverlap=noverlap)
                Ln = laplacian(Wn, normalized=True)
                kn = pick_k_by_eigengap(Ln, kmax=kmax)
                lbl_n, _ = spectral_cluster(Wn, kn)
                null_labels.append(lbl_n)
            cons_n, _ = consensus_from_labels(null_labels)
            # ARI between real consensus and null consensus as "similarity"; smaller is better separation
            null_aris.append(adjusted_rand_score(cons, cons_n))
        null_aris = np.array(null_aris)
        p_val = float((np.sum(null_aris >= loso_med_ari) + 1) / (len(null_aris) + 1))

        # Save band metrics
        met = {
            "band": band,
            "n_subjects": len(subjects),
            "loso_median_ari": float(loso_med_ari),
            "null_ari_mean": float(null_aris.mean()),
            "null_ari_p_value": p_val,
            "subjects": subjects,
            "channels": chans,
        }
        save_json(met, os.path.join(out_met, f"band__{band}__metrics.json"))
        # Save consensus labels
        np.save(os.path.join(out_tab, f"band__{band}__consensus_labels.npy"), cons)
        np.save(os.path.join(out_tab, f"band__{band}__coassoc.npy"), co)

    return {"subjects": subjects, "channels": chans, "bands": bands}

# ------------------------------
# Synthetic demo
# ------------------------------

def synth_rest(n_subj=4, n_ch=32, fs=250.0, seconds=16, band="alpha", seed=7) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs*seconds))/fs
    # base noise
    Xs = []
    for s in range(n_subj):
        X = rng.normal(0, 1, size=(n_ch, len(t)))
        # inject alpha hub in posterior (channels 16:28)
        omega = 2*np.pi*10.0  # 10 Hz
        driver = np.sin(omega*t + rng.uniform(0, 2*np.pi))
        hub_idx = slice(n_ch//2, n_ch - 4)
        for c in range(hub_idx.start, hub_idx.stop):
            X[c] += 0.8*driver + 0.2*rng.normal(0, 1, size=len(t))
        Xs.append(X)
    return Xs, fs

# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="", help="Directory with *.npy subject files")
    ap.add_argument("--glob", type=str, default="*.npy", help="Glob for subject files")
    ap.add_argument("--fs", type=float, default=250.0, help="Sampling rate")
    ap.add_argument("--out_root", type=str, default="cnt_artifacts", help="Output root")
    ap.add_argument("--bands", type=str, default="alpha,theta", help="Comma-separated bands to analyze")
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--null_perms", type=int, default=200)
    ap.add_argument("--demo", nargs="*", help="Run synthetic demo: N_SUBJ N_CH SECONDS (defaults 4 32 16)")
    args = ap.parse_args()

    out_root = ensure_dir(args.out_root)
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]

    if args.demo is not None:
        if len(args.demo) >= 1:
            n_subj = int(args.demo[0])
        else:
            n_subj = 4
        n_ch = int(args.demo[1]) if len(args.demo) >= 2 else 32
        seconds = int(args.demo[2]) if len(args.demo) >= 3 else 16
        Xs, fs = synth_rest(n_subj=n_subj, n_ch=n_ch, seconds=seconds, seed=7)
        tmp_dir = ensure_dir(os.path.join(out_root, "demo_data"))
        subs = []
        for i, X in enumerate(Xs):
            f = os.path.join(tmp_dir, f"subject_{i:02d}.npy")
            np.save(f, X)
            with open(f.replace(".npy", ".channels.txt"), "w") as g:
                for c in range(X.shape[0]):
                    g.write(f"ch{c}\n")
            subs.append(f)
        print(f"[demo] wrote {len(subs)} subjects to {tmp_dir}")
        data_files = subs
        fs_use = fs
    else:
        if not args.data_dir:
            print("ERROR: --data_dir required unless using --demo")
            sys.exit(2)
        data_files = sorted(glob.glob(os.path.join(args.data_dir, args.glob)))
        if not data_files:
            print(f"ERROR: no files matched in {args.data_dir} with {args.glob}")
            sys.exit(2)
        fs_use = args.fs

    print(f"subjects: {len(data_files)} | bands: {bands} | out: {out_root}")
    summary = run_pipeline(
        data_files=data_files,
        fs=fs_use,
        out_root=out_root,
        bands=bands,
        kmax=args.kmax,
        null_perms=args.null_perms,
    )
    save_json(summary, os.path.join(out_root, "summary.json"))
    print("done.")

if __name__ == "__main__":
    main()
