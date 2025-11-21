
# === CNT "3I Atlas" — Mega Check‑In (single cell, v3) ========================
# (See prior cell content for full documentation; this is the complete code.)
import os, re, sys, json, glob, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
USE_POLARS = False
try:
    import pandas as pd
except Exception:
    pd = None
if USE_POLARS:
    try:
        import polars as pl
    except Exception:
        USE_POLARS = False
try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None
try:
    import umap
except Exception:
    umap = None
try:
    from fpdf import FPDF
except Exception:
    FPDF = None
def ts_utc(): return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
def ts_local(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True); return p
def normalize_pack_dir(p: Path) -> Path:
    parts = list(p.parts)
    if len(parts) >= 2 and parts[-1].lower() == "vector_embedding" and parts[-2].lower() == "vector_embedding":
        return Path(*parts[:-1])
    name = p.name.lower()
    if name.endswith("_vector_embedding_vector_embedding"):
        return p.with_name(p.name[: -len("_vector_embedding")])
    return p
def list_datafiles(root: Path):
    patterns = []
    for base in ("out", "data", ""):
        basep = (root / base) if base else root
        patterns += [str(basep / "**/*.csv"), str(basep / "**/*.tsv"),
                     str(basep / "**/*.parquet"), str(basep / "**/*.feather"),
                     str(basep / "**/*.npz"), str(basep / "**/*.npy")]
    hits = []
    for pat in patterns: hits.extend([Path(p) for p in glob.glob(pat, recursive=True)])
    hits = [h for h in hits if h.is_file()]
    hits.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return hits
def read_table_any(path: Path, max_rows=None):
    suff = path.suffix.lower()
    if USE_POLARS:
        if 'pl' not in globals(): raise RuntimeError("Polars not available")
        if suff in (".csv", ".tsv"):
            sep = "," if suff == ".csv" else "\t"; df = pl.read_csv(str(path), separator=sep)
            return df if max_rows is None else df.head(max_rows)
        elif suff == ".parquet": df = pl.read_parquet(str(path)); return df if max_rows is None else df.head(max_rows)
        elif suff == ".feather": df = pl.read_ipc(str(path)); return df if max_rows is None else df.head(max_rows)
        elif suff in (".npz", ".npy"):
            arr = np.load(str(path)); 
            if isinstance(arr, np.lib.npyio.NpzFile): key = next(iter(arr.files)); arr = arr[key]
            if arr.ndim == 2:
                df = pl.DataFrame(arr); df = df.with_columns(pl.Series("gene", [f"g{i}" for i in range(arr.shape[0])]))
                df = df.select(["gene"] + [c for c in df.columns if c != "gene"]); return df if max_rows is None else df.head(max_rows)
            raise RuntimeError(f"Unsupported NPZ/NPY shape in {path}: {arr.shape}")
        else: raise RuntimeError(f"Unsupported file type: {suff}")
    else:
        if pd is None: raise RuntimeError("pandas not available")
        if suff in (".csv", ".tsv"):
            sep = "," if suff == ".csv" else "\t"; return pd.read_csv(path, nrows=max_rows, sep=sep)
        elif suff == ".parquet": return pd.read_parquet(path)
        elif suff == ".feather": return pd.read_feather(path)
        elif suff in (".npz", ".npy"):
            arr = np.load(str(path))
            if isinstance(arr, np.lib.npyio.NpzFile): key = next(iter(arr.files)); arr = arr[key]
            if arr.ndim == 2:
                cols = [f"col_{j}" for j in range(arr.shape[1])]; df = pd.DataFrame(arr, columns=cols)
                df.insert(0, "gene", [f"g{i}" for i in range(arr.shape[0])]); return df if max_rows is None else df.head(max_rows)
            raise RuntimeError(f"Unsupported NPZ/NPY shape in {path}: {arr.shape}")
        else: raise RuntimeError(f"Unsupported file type: {suff}")
def to_pandas(df):
    if pd is None: raise RuntimeError("pandas not available")
    if USE_POLARS: return df.to_pandas()
    return df
def infer_matrix(df):
    meta = {"format": None, "value_col": None, "gene_col": None, "tissue_col": None}
    gene_cols = [c for c in df.columns if str(c).lower() in ("gene","gene_id","gene_name","symbol","ensembl","ensembl_id","id")]
    tissue_cols = [c for c in df.columns if str(c).lower() in ("tissue","organ","celltype","cell_type","sample","sample_id")]
    val_keys = ("value","expression","expr","count","tpms","fpkm","reads","abundance","intensity")
    value_cols = [c for c in df.columns if str(c).lower() in val_keys]
    emb_like = [c for c in df.columns if re.match(r"(emb(ed(ding)?)?_?\d+)$", str(c).lower())]
    if gene_cols and tissue_cols and (value_cols or emb_like):
        g = gene_cols[0]; t = tissue_cols[0]; v = (value_cols[0] if value_cols else emb_like[0])
        pivot = df.pivot_table(index=g, columns=t, values=v, aggfunc="mean").sort_index()
        E = pivot.to_numpy(dtype=float); gene_names = pivot.index.astype(str).to_list(); sample_names = [str(c) for c in pivot.columns.to_list()]
        meta.update({"format":"long/tidy","gene_col":g,"tissue_col":t,"value_col":v}); return E, gene_names, sample_names, meta
    if gene_cols:
        g = gene_cols[0]; sub = df.copy().drop_duplicates(subset=[g]).set_index(g)
        num = sub.select_dtypes(include=[np.number]); 
        if num.shape[1]==0: num = sub.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(how="all", axis=1)
        E = num.to_numpy(dtype=float); gene_names = [str(i) for i in num.index.to_list()]; sample_names = [str(c) for c in num.columns.to_list()]
        meta.update({"format":"wide","gene_col":g}); return E, gene_names, sample_names, meta
    sub = df.copy().dropna(how="all", axis=1)
    if sub.shape[1] < 2: raise RuntimeError("Table has <2 columns; can't infer matrix.")
    g = sub.columns[0]; sub = sub.drop_duplicates(subset=[g]).set_index(g)
    num = sub.select_dtypes(include=[np.number]); 
    if num.shape[1]==0: num = sub.apply(pd.to_numeric, errors="coerce")
    num = num.dropna(how="all", axis=1)
    E = num.to_numpy(dtype=float); gene_names = [str(i) for i in num.index.to_list()]; sample_names = [str(c) for c in num.columns.to_list()]
    meta.update({"format":"wide/fallback","gene_col":str(g)}); return E, gene_names, sample_names, meta
def summarize_matrix(E, gene_names, sample_names, k_top=25):
    X = E.copy(); 
    if np.nanmin(X) < 0: X = X - np.nanmin(X)
    X = np.nan_to_num(X, nan=0.0)
    var  = np.nanvar(X, axis=1); mean = np.nanmean(X, axis=1) + 1e-12; cv = np.sqrt(var) / mean
    def gini(row, eps=1e-12):
        r = np.asarray(row, dtype=float); mn = np.nanmin(r)
        if mn < 0: r = r - mn
        r = np.nan_to_num(r, nan=0.0); mu = r.mean() + eps
        diff_sum = np.abs(r[:, None] - r[None, :]).mean(); return 0.5 * diff_sum / mu
    def Hn_row(p, eps=1e-12):
        p = np.clip(p, eps, None); p = p / p.sum(); H = float(-(p * np.log(p)).sum())
        return H / (np.log(X.shape[1]) if X.shape[1] > 1 else 1.0)
    gini_v = np.array([gini(row) for row in X]); Hn = np.array([Hn_row(row) for row in X])
    idx_g = np.argsort(-gini_v)[:k_top]; idx_lo = np.argsort(Hn)[:k_top]; idx_hi = np.argsort(-Hn)[:k_top]
    def take(idx): return [(gene_names[i], float(gini_v[i]), float(Hn[i]), float(cv[i]), float(mean[i])) for i in idx]
    tops = {"top_gini": take(idx_g), "top_specialized_low_entropy": take(idx_lo), "top_housekeeping_high_entropy": take(idx_hi)}
    summary = {"n_genes": int(X.shape[0]), "n_samples": int(X.shape[1]),
               "gini_mean": float(np.nanmean(gini_v)), "gini_median": float(np.nanmedian(gini_v)),
               "entropy_mean": float(np.nanmean(Hn)), "entropy_median": float(np.nanmedian(Hn)), "cv_mean": float(np.nanmean(cv))}
    per_gene = {"var": var.tolist(), "mean": mean.tolist(), "cv": cv.tolist(), "gini": gini_v.tolist(), "H_norm": Hn.tolist()}
    return summary, per_gene, tops
def to_csv(path, rows, header):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows: f.write(",".join(map(lambda x: str(x).replace(",",";"), r)) + "\n")
def try_pca(E, n=2, random_state=42):
    if PCA is None: return None, None
    X = np.nan_to_num(E, nan=0.0); X = X - X.mean(axis=1, keepdims=True)
    p = PCA(n_components=min(n, min(X.shape)-1), random_state=random_state)
    try: Y = p.fit_transform(X.T); return Y, p.explained_variance_ratio_.tolist()
    except Exception: return None, None
def try_umap(E, n=2, random_state=42):
    if umap is None: return None
    X = np.nan_to_num(E, nan=0.0); X = X - X.mean(axis=1, keepdims=True)
    try: return umap.UMAP(n_components=n, random_state=random_state).fit_transform(X.T)
    except Exception: return None
def plot_hist(arr, path, title, xlabel):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ensure_dir(Path(path).parent); plt.figure(); plt.hist([a for a in arr if not np.isnan(a)], bins=50)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Count"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
def plot_bar(items, path, title, ylabel):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as _np
    ensure_dir(Path(path).parent); labels = [i[0] for i in items]; vals = [i[1] for i in items]
    plt.figure(figsize=(10, max(3, 0.3*len(items)))); y = _np.arange(len(items)); plt.barh(y, vals); plt.yticks(y, labels)
    plt.title(title); plt.xlabel(ylabel); plt.ylabel("Gene"); plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
def plot_scatter(Y, path, title, xlabel="Dim 1", ylabel="Dim 2"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ensure_dir(Path(path).parent); plt.figure(); plt.scatter(Y[:,0], Y[:,1], s=12, alpha=0.8)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
def write_pdf(report_md_path: Path, images, out_pdf: Path, title="3I Atlas Check-In"):
    if FPDF is None: return False
    try:
        from fpdf.enums import XPos, YPos; HAVE_ENUMS = True
    except Exception:
        HAVE_ENUMS = False
    REPL = {"\u2011":"-","\u2013":"-","\u2014":"-","\u2018":"'","\u2019":"'","\u201c":'"',"\u201d":'"',"\u2026":"..."}
    def ascii_fallback(s: str):
        for k,v in REPL.items(): s = s.replace(k, v)
        return s
    ttf_candidates = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\DejaVuSans.ttf",
                      r"C:\Windows\Fonts\Calibri.ttf", r"C:\Windows\Fonts\segoeui.ttf"]
    pdf = FPDF(orientation="P", unit="mm", format="A4"); pdf.set_auto_page_break(auto=True, margin=12); pdf.add_page()
    used_unicode = False
    for ttf in ttf_candidates:
        if Path(ttf).exists():
            try:
                try: pdf.add_font("U", "", ttf, uni=True)
                except TypeError: pdf.add_font("U", "", ttf)
                pdf.set_font("U", "", 16); used_unicode = True; break
            except Exception: pass
    if not used_unicode: pdf.set_font("helvetica", "", 16)
    safe_title = title if used_unicode else ascii_fallback(title)
    if HAVE_ENUMS: pdf.cell(0, 10, safe_title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    else:          pdf.cell(0, 10, safe_title, ln=1)
    pdf.set_font("U" if used_unicode else "helvetica", "", 10)
    with open(report_md_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("!"): continue
            pdf.multi_cell(0, 5, line if used_unicode else ascii_fallback(line))
    for img in images:
        if img and Path(img).exists():
            pdf.add_page(); pdf.image(str(img), x=10, y=20, w=180)
            if HAVE_ENUMS: pdf.cell(0, 6, Path(img).name, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:          pdf.ln(6)
    ensure_dir(Path(out_pdf).parent); pdf.output(str(out_pdf)); return True
# ---- Main
PACK_DIR = None
ROOT_HINTS = [r"C:\Users\caleb\CNT_Lab", r"E:\CNT", r"E:\CNT\notebooks\archive", r"D:\CNT", r"C:\CNT", str(Path.cwd())]
RUN_BASE = r"C:\Users\caleb\CNT_Lab\notebooks\archive\cnt_runs\3i_atlas_checkin"
if not Path(RUN_BASE).exists(): RUN_BASE = str(Path.cwd() / "cnt_runs" / "3i_atlas_checkin")
STAMP = ts_utc(); RUN_DIR = ensure_dir(Path(RUN_BASE) / STAMP)
print(f"[{ts_local()}] 3I Atlas Check-In v3 starting…"); print(f"  Run dir: {RUN_DIR}")
candidates = []
def score_candidate(path: Path) -> int:
    s = str(path).lower(); sc = 0
    if path.is_dir(): sc += 3
    if "vector" in s and "embed" in s: sc += 5
    if "cnt_3i_atlas_all" in s: sc += 3
    if s.endswith((".csv",".tsv",".parquet",".feather")): sc += 1
    if "vector_embedding_vector_embedding" in s: sc -= 4
    try: sc += min(6, len(list_datafiles(path)) if path.is_dir() else 1)
    except Exception: pass
    try: sc += int(path.stat().st_mtime // 3600) % 10
    except Exception: pass
    return sc
def gather_candidates(root: Path):
    pats = ["**/*3i*atlas*vector*embed*","**/*3i*atlas*embed*","**/*3i*atlas*","**/cnt_3i_atlas*","**/*3i*atlas*.csv"]
    for pat in pats:
        for hit in root.glob(pat):
            if ".ipynb_checkpoints" in str(hit): continue
            candidates.append(hit)
def all_roots(): return [Path(p) for p in ROOT_HINTS if Path(p).exists()]
if PACK_DIR:
    pack = normalize_pack_dir(Path(PACK_DIR)); print(f"  PACK_DIR override: {pack}")
else:
    for r in all_roots(): print(f"  Scanning: {r}"); gather_candidates(r)
    if not candidates: raise SystemExit("No 3I Atlas candidates found. Set PACK_DIR to the pack root.")
    candidates = [normalize_pack_dir(c) for c in candidates]
    uniq, seen = [], set()
    for c in candidates:
        k = str(c).lower()
        if k not in seen: seen.add(k); uniq.append(c)
    candidates = uniq; candidates.sort(key=score_candidate, reverse=True); pack = candidates[0]
print(f"  Candidate pack: {pack}")
INCLUDE_PATTERNS = ["atlas","gene","expr","tpm","fpkm","counts"]; EXCLUDE_PATTERNS = ["noaa","mag","weather","test","debug"]
def choose_data_root(p: Path):
    files = list_datafiles(p)
    if files: return p, files
    par = p.parent
    if par and par.exists():
        files = list_datafiles(par)
        if files: print(f"  Recovery: using parent of candidate ({par})"); return par, files
    return p, []
pack, data_files = choose_data_root(pack)
if not data_files: raise SystemExit(f"No supported data files under {pack}. Set PACK_DIR to the pack root with out/ or data/.")
def file_rank(p: Path):
    ext = p.suffix.lower(); base = p.name.lower()
    order = {".csv":3, ".tsv":3, ".parquet":2, ".feather":2, ".npz":1, ".npy":1}
    bonus = sum(1 for w in INCLUDE_PATTERNS if w in base) - sum(1 for w in EXCLUDE_PATTERNS if w in base)
    sniff = 0
    try:
        tmp = read_table_any(p, max_rows=32); tmp_pd = to_pandas(tmp)
        cols = [str(c).lower() for c in tmp_pd.columns]
        if any(c in cols for c in ["gene","gene_id","gene_name","symbol","ensembl","ensembl_id"]): sniff += 4
        if tmp_pd.shape[1] >= 20: sniff += 1
    except Exception: pass
    return (order.get(ext,0), bonus, sniff, p.stat().st_size)
data_files.sort(key=file_rank, reverse=True); chosen = data_files[0]
print(f"  Using data file: {chosen} ({chosen.stat().st_size/1_048_576:.2f} MiB)")
df_any = read_table_any(chosen, max_rows=None); df = to_pandas(df_any)
E, gene_names, sample_names, meta = infer_matrix(df)
print(f"  Inferred matrix: genes={len(gene_names)}, samples={len(sample_names)}  format={meta['format']}")
summary, per_gene, tops = summarize_matrix(E, gene_names, sample_names, k_top=25)
plots = {}
to_csv(Path(RUN_DIR/"top_gini_genes.csv"), tops["top_gini"], ["gene","gini","H_norm","cv","mean"])
to_csv(Path(RUN_DIR/"top_specialized_low_entropy.csv"), tops["top_specialized_low_entropy"], ["gene","gini","H_norm","cv","mean"])
to_csv(Path(RUN_DIR/"top_housekeeping_high_entropy.csv"), tops["top_housekeeping_high_entropy"], ["gene","gini","H_norm","cv","mean"])
to_csv(Path(RUN_DIR/"summary_stats.csv"), [[k, v] for k, v in summary.items()], ["metric","value"])
def plot_all():
    global plots
    plot_hist(per_gene["gini"], Path(RUN_DIR/"plots/gini_hist.png"), "Gini distribution (gene specialization)", "Gini"); plots["gini_hist"] = str(Path(RUN_DIR/"plots/gini_hist.png"))
    plot_hist(per_gene["H_norm"], Path(RUN_DIR/"plots/entropy_hist.png"), "Normalized entropy across samples", "H_norm"); plots["entropy_hist"] = str(Path(RUN_DIR/"plots/entropy_hist.png"))
    plot_bar(tops["top_gini"], Path(RUN_DIR/"plots/top_gini_bar.png"), "Top specialized genes (by Gini)", "Gini"); plots["top_gini_bar"] = str(Path(RUN_DIR/"plots/top_gini_bar.png"))
    pca_pts, pca_var = try_pca(E, n=2, random_state=42)
    if pca_pts is not None:
        plot_scatter(pca_pts, Path(RUN_DIR/"plots/pca_scatter.png"), f"PCA on samples (var={sum(pca_var):.2%})", "PC1", "PC2"); plots["pca_scatter"] = str(Path(RUN_DIR/"plots/pca_scatter.png"))
    else: print("  PCA not available or failed; skipping PCA plot.")
    umap_pts = try_umap(E, n=2, random_state=42)
    if umap_pts is not None:
        plot_scatter(umap_pts, Path(RUN_DIR/"plots/umap_scatter.png"), "UMAP on samples", "UMAP-1", "UMAP-2"); plots["umap_scatter"] = str(Path(RUN_DIR/"plots/umap_scatter.png"))
plot_all()
def read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return None
def write_json(path: Path, obj):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=2)
def last_snapshot(dir_base: Path):
    files = glob.glob(str(dir_base / "*" / "snapshot.json"))
    if not files: return None, None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True); path = Path(files[0]); return path, read_json(path)
SNAPSHOT_PATH = Path(RUN_DIR/"snapshot.json"); prev_path, prev = last_snapshot(Path(RUN_BASE)); deltas = None
if prev:
    deltas = {"n_genes_delta": summary["n_genes"] - int(prev.get("summary",{}).get("n_genes", 0)),
              "n_samples_delta": summary["n_samples"] - int(prev.get("summary",{}).get("n_samples", 0)),
              "gini_mean_delta": summary["gini_mean"] - float(prev.get("summary",{}).get("gini_mean", 0.0)),
              "entropy_mean_delta": summary["entropy_mean"] - float(prev.get("summary",{}).get("entropy_mean", 0.0)),
              "cv_mean_delta": summary["cv_mean"] - float(prev.get("summary",{}).get("cv_mean", 0.0)),
              "changed_samples": False, "added_samples": [], "removed_samples": []}
    try:
        prev_samples = set(prev.get("sample_names", [])); cur_samples = set(sample_names)
        add = sorted(cur_samples - prev_samples); rem = sorted(prev_samples - cur_samples)
        if add or rem: deltas["changed_samples"] = True; deltas["added_samples"] = add; deltas["removed_samples"] = rem
    except Exception: pass
    write_json(Path(RUN_DIR/"delta_summary.json"), deltas); print(f"  Δ written: {Path(RUN_DIR/'delta_summary.json')}")
else: print("  No prior snapshot found; this will serve as the baseline.")
snapshot = {"meta": {"stamp_utc": ts_utc(), "stamp_local": ts_local(), "host": platform.node(),
                     "python": sys.version.split()[0], "pack_dir": str(pack), "data_file": str(chosen)},
            "summary": summary, "sample_names": sample_names[:5000],
            "top_gini": tops["top_gini"], "top_housekeeping_high_entropy": tops["top_housekeeping_high_entropy"]}
write_json(SNAPSHOT_PATH, snapshot)
def write_report_md(path: Path, info):
    ensure_dir(path.parent); L = []
    L.append(f"# 3I Atlas Check-In — {info['meta']['stamp_local']}"); L.append("")
    L.append(f"- **Pack**: `{info['meta']['pack']}`"); L.append(f"- **Run dir**: `{info['meta']['run_dir']}`")
    L.append(f"- **Rows (genes)**: **{info['summary']['n_genes']}**, **Samples**: **{info['summary']['n_samples']}**")
    L.append(f"- Gini (mean/median): **{info['summary']['gini_mean']:.4f} / {info['summary']['gini_median']:.4f}**")
    L.append(f"- Entropy_n (mean/median): **{info['summary']['entropy_mean']:.4f} / {info['summary']['entropy_median']:.4f}**")
    L.append(f"- CV (mean): **{info['summary']['cv_mean']:.4f}**"); L.append("")
    for key in ("gini_hist","entropy_hist","top_gini_bar","pca_scatter","umap_scatter"):
        p = info["plots"].get(key); if p: L.append(f"![{key}]({Path(p).name})")
    L.append(""); L.append("## Top specialized (by Gini) — preview")
    for (name,g,h,cv,mu) in info["tops"]["top_gini"][:10]:
        L.append(f"- {name}: Gini={g:.4f}, H_n={h:.4f}, CV={cv:.3f}, mean={mu:.3g}")
    L.append(""); L.append("## Top housekeeping (high normalized entropy) — preview")
    for (name,g,h,cv,mu) in info["tops"]["top_housekeeping_high_entropy"][:10]:
        L.append(f"- {name}: H_n={h:.4f}, Gini={g:.4f}, CV={cv:.3f}, mean={mu:.3g}")
    if info.get("deltas"):
        d = info["deltas"]; L.append(""); L.append("## Delta vs last snapshot")
        L.append(f"- Genes: **{d.get('n_genes_delta',0):+d}**, Samples: **{d.get('n_samples_delta',0):+d}**")
        if "gini_mean_delta" in d: L.append(f"- Δ Gini mean: **{d['gini_mean_delta']:+.4f}**, Δ Entropy_n mean: **{d.get('entropy_mean_delta',0):+.4f}**")
        if d.get("changed_samples"): L.append(f"- Changed sample set: +{len(d['added_samples'])} / -{len(d['removed_samples'])}")
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(L))
info = {"meta": {"stamp_local": ts_local(), "pack": str(pack), "run_dir": str(RUN_DIR)},
        "summary": summary, "tops": tops, "deltas": deltas, "plots": plots}
REPORT_MD = Path(RUN_DIR/"report.md"); write_report_md(REPORT_MD, info); print(f"  Wrote: {REPORT_MD}")
REPORT_PDF = Path(RUN_DIR/"report.pdf")
ok_pdf = write_pdf(REPORT_MD,
                   images=[plots.get("gini_hist"), plots.get("entropy_hist"), plots.get("top_gini_bar"),
                           plots.get("pca_scatter"), plots.get("umap_scatter")],
                   out_pdf=REPORT_PDF, title="3I Atlas Check-In")
print("  PDF:   {}".format(REPORT_PDF if ok_pdf else "(skipped; fpdf missing)"))
print(f"[{ts_local()}] Done. Keep the field humming.")
