
# === CNT "3I Atlas" — Mega Check‑In (single cell, v2 resilient) =============
# Fixes:
#  - Search ALL roots; don't stop at the first existing one.
#  - Normalize duplicate suffix dirs (…\vector_embedding\vector_embedding).
#  - Accept CSV/TSV/Parquet/Feather/NPZ/NPY, not just CSV.
#  - If target dir has no tables, try its parent once.
#  - Better candidate scoring: prefer dirs with actual data files.
# ============================================================================

import os, re, sys, json, glob, math, time, uuid, platform, textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Prefer pandas; fall back to polars by toggling USE_POLARS=True
USE_POLARS = False
try:
    import pandas as pd
except Exception as e:
    pd = None

if USE_POLARS:
    try:
        import polars as pl
    except Exception:
        USE_POLARS = False

# Optional libs
try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

try:
    import umap
except Exception:
    umap = None

# Optional PDF
try:
    from fpdf import FPDF
except Exception:
    FPDF = None


# ----------------------------- Helpers --------------------------------------

def ts_utc():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")

def ts_local():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def normalize_pack_dir(p: Path) -> Path:
    """
    Collapse duplicate 'vector_embedding' suffixes and strip trailing empty segments.
    e.g., .../vector_embedding/vector_embedding -> .../vector_embedding
    """
    parts = list(p.parts)
    if len(parts) >= 2 and parts[-1].lower() == "vector_embedding" and parts[-2].lower() == "vector_embedding":
        return Path(*parts[:-1])
    # Sometimes the duplication appears in a single folder name as "..._vector_embedding_vector_embedding"
    name = p.name.lower()
    if name.endswith("_vector_embedding_vector_embedding"):
        return p.with_name(p.name[: -len("_vector_embedding")])
    return p

def first_existing(paths):
    return [Path(p) for p in paths if Path(p).exists()]

def list_datafiles(root: Path):
    """
    Return a list of candidate data files under root with supported suffixes.
    We search typical subfolders: out/, data/, current dir.
    """
    patterns = []
    for base in ("out", "data", ""):
        basep = (root / base) if base else root
        patterns += [
            str(basep / "**/*.csv"),
            str(basep / "**/*.tsv"),
            str(basep / "**/*.parquet"),
            str(basep / "**/*.feather"),
            str(basep / "**/*.npz"),
            str(basep / "**/*.npy"),
        ]
    hits = []
    for pat in patterns:
        hits.extend([Path(p) for p in glob.glob(pat, recursive=True)])
    # files only
    hits = [h for h in hits if h.is_file()]
    # prefer larger files first
    hits.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return hits

def read_table_any(path: Path, max_rows=None):
    suff = path.suffix.lower()
    if USE_POLARS:
        if suff in (".csv", ".tsv"):
            sep = "," if suff == ".csv" else "\t"
            df = pl.read_csv(str(path), separator=sep)
            return df if max_rows is None else df.head(max_rows)
        elif suff == ".parquet":
            df = pl.read_parquet(str(path))
            return df if max_rows is None else df.head(max_rows)
        elif suff == ".feather":
            df = pl.read_ipc(str(path))
            return df if max_rows is None else df.head(max_rows)
        elif suff in (".npz", ".npy"):
            arr = np.load(str(path))
            if isinstance(arr, np.lib.npyio.NpzFile):
                # choose first array-like
                key = next(iter(arr.files))
                arr = arr[key]
            if arr.ndim == 2:
                # synthesize a DataFrame-like table with index + numbered columns
                df = pl.DataFrame(arr)
                df = df.with_columns(pl.Series("gene", [f"g{i}" for i in range(arr.shape[0])]))
                df = df.select(["gene"] + [c for c in df.columns if c != "gene"])
                return df if max_rows is None else df.head(max_rows)
            raise RuntimeError(f"Unsupported NPZ/NPY shape in {path}: {arr.shape}")
        else:
            raise RuntimeError(f"Unsupported file type: {suff}")
    else:
        if pd is None:
            raise RuntimeError("pandas not available; install pandas or set USE_POLARS=True")
        if suff in (".csv", ".tsv"):
            sep = "," if suff == ".csv" else "\t"
            try:
                return pd.read_csv(path, nrows=max_rows, sep=sep)
            except Exception as e:
                raise RuntimeError(f"Failed to read {path}: {e}")
        elif suff == ".parquet":
            return pd.read_parquet(path)
        elif suff == ".feather":
            return pd.read_feather(path)
        elif suff in (".npz", ".npy"):
            arr = np.load(str(path))
            if isinstance(arr, np.lib.npyio.NpzFile):
                key = next(iter(arr.files))
                arr = arr[key]
            if arr.ndim == 2:
                # build a DataFrame with 'gene' + col_*
                cols = [f"col_{j}" for j in range(arr.shape[1])]
                df = pd.DataFrame(arr, columns=cols)
                df.insert(0, "gene", [f"g{i}" for i in range(arr.shape[0])])
                return df if max_rows is None else df.head(max_rows)
            raise RuntimeError(f"Unsupported NPZ/NPY shape in {path}: {arr.shape}")
        else:
            raise RuntimeError(f"Unsupported file type: {suff}")

def to_pandas(df):
    if pd is None:
        raise RuntimeError("pandas not available")
    if USE_POLARS:
        return df.to_pandas()
    return df

def infer_matrix(df: 'pd.DataFrame'):
    """
    Infer a (genes x samples) numeric matrix from common 3I Atlas shapes.
    """
    meta = {"format": None, "value_col": None, "gene_col": None, "tissue_col": None}
    cols = [str(c).lower() for c in df.columns]

    # candidate id columns
    gene_cols = [c for c in df.columns if str(c).lower() in ("gene","gene_id","gene_name","symbol","ensembl","ensembl_id","id")]
    tissue_cols = [c for c in df.columns if str(c).lower() in ("tissue","organ","celltype","cell_type","sample","sample_id")]

    # likely value columns
    val_keys = ("value","expression","expr","count","tpms","fpkm","reads","abundance","intensity")
    value_cols = [c for c in df.columns if str(c).lower() in val_keys]

    # Embedding-shaped (e.g., embedding_0, embedding_1, …)
    emb_like = [c for c in df.columns if re.match(r"(emb(ed(ding)?)?_?\d+)$", str(c).lower())]

    # Tidy form?
    if gene_cols and tissue_cols and (value_cols or emb_like):
        g = gene_cols[0]; t = tissue_cols[0]
        v = (value_cols[0] if value_cols else emb_like[0])
        meta.update({"format":"long/tidy","gene_col":g,"tissue_col":t,"value_col":v})
        pivot = df.pivot_table(index=g, columns=t, values=v, aggfunc="mean")
        pivot = pivot.sort_index()
        E = pivot.to_numpy(dtype=float)
        gene_names = pivot.index.astype(str).to_list()
        sample_names = [str(c) for c in pivot.columns.to_list()]
        return E, gene_names, sample_names, meta

    # Wide form with known gene column
    if gene_cols:
        g = gene_cols[0]
        sub = df.copy().drop_duplicates(subset=[g]).set_index(g)
        num = sub.select_dtypes(include=[np.number])
        if num.shape[1]==0:
            num = sub.apply(pd.to_numeric, errors="coerce")
        num = num.dropna(how="all", axis=1)
        E = num.to_numpy(dtype=float)
        gene_names = [str(i) for i in num.index.to_list()]
        sample_names = [str(c) for c in num.columns.to_list()]
        meta.update({"format":"wide","gene_col":g})
        return E, gene_names, sample_names, meta

    # Fallback: first column id, rest numeric
    sub = df.copy().dropna(how="all", axis=1)
    if sub.shape[1] < 2:
        raise RuntimeError("Table has <2 columns; can't infer matrix.")
    g = sub.columns[0]
    sub = sub.drop_duplicates(subset=[g]).set_index(g)
    num = sub.select_dtypes(include=[np.number])
    if num.shape[1]==0:
        num = sub.apply(pd.to_numeric, errors="coerce")
    num = num.dropna(how="all", axis=1)
    E = num.to_numpy(dtype=float)
    gene_names = [str(i) for i in num.index.to_list()]
    sample_names = [str(c) for c in num.columns.to_list()]
    meta.update({"format":"wide/fallback","gene_col":str(g)})
    return E, gene_names, sample_names, meta

def summarize_matrix(E: np.ndarray, gene_names, sample_names, k_top=25):
    n_genes, n_samp = E.shape
    # zero-floor for stats
    X = E.copy()
    if np.nanmin(X) < 0:
        X = X - np.nanmin(X)
    X = np.nan_to_num(X, nan=0.0)
    # per-gene
    var = np.nanvar(X, axis=1)
    mean = np.nanmean(X, axis=1) + 1e-12
    cv = np.sqrt(var) / mean
    # gini and entropy
    def gini_coefficient(row, eps=1e-12):
        r = np.asarray(row, dtype=float)
        mn = np.nanmin(r)
        if mn < 0:
            r = r - mn
        r = np.nan_to_num(r, nan=0.0)
        mu = r.mean() + eps
        diff_sum = np.abs(r[:, None] - r[None, :]).mean()
        return 0.5 * diff_sum / mu
    def shannon_entropy(p, eps=1e-12):
        p = np.clip(p, eps, None)
        p = p / p.sum()
        return float(-(p * np.log(p)).sum())
    gini = np.array([gini_coefficient(row) for row in X])
    H = np.array([shannon_entropy(row) for row in X])
    H_norm = H / (np.log(X.shape[1]) if X.shape[1] > 1 else 1.0)  # 0..1

    idx_gini = np.argsort(-gini)[:k_top]
    idx_entropy_low = np.argsort(H_norm)[:k_top]
    idx_entropy_high = np.argsort(-H_norm)[:k_top]

    def take(idx):
        return [(gene_names[i], float(gini[i]), float(H_norm[i]), float(cv[i]), float(mean[i])) for i in idx]

    top_gini = take(idx_gini)
    top_spec = take(idx_entropy_low)
    top_house = take(idx_entropy_high)

    summary = {
        "n_genes": int(n_genes),
        "n_samples": int(n_samp),
        "gini_mean": float(np.nanmean(gini)),
        "gini_median": float(np.nanmedian(gini)),
        "entropy_mean": float(np.nanmean(H_norm)),
        "entropy_median": float(np.nanmedian(H_norm)),
        "cv_mean": float(np.nanmean(cv)),
    }
    per_gene = {
        "var": var.tolist(),
        "mean": mean.tolist(),
        "cv": cv.tolist(),
        "gini": gini.tolist(),
        "H_norm": H_norm.tolist(),
    }
    tops = {
        "top_gini": top_gini,
        "top_specialized_low_entropy": top_spec,
        "top_housekeeping_high_entropy": top_house,
    }
    return summary, per_gene, tops

def to_csv(path: Path, rows, header):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(lambda x: str(x).replace(",",";"), r)) + "\n")

def try_pca(E: np.ndarray, n=2, random_state=42):
    if PCA is None:
        return None, None
    X = np.nan_to_num(E, nan=0.0)
    X = X - X.mean(axis=1, keepdims=True)
    pca = PCA(n_components=min(n, min(X.shape)-1), random_state=random_state)
    try:
        Y = pca.fit_transform(X.T)
        return Y, pca.explained_variance_ratio_.tolist()
    except Exception:
        return None, None

def try_umap(E: np.ndarray, n=2, random_state=42):
    if umap is None:
        return None
    X = np.nan_to_num(E, nan=0.0)
    X = X - X.mean(axis=1, keepdims=True)
    try:
        Y = umap.UMAP(n_components=n, random_state=random_state).fit_transform(X.T)
        return Y
    except Exception:
        return None

def plot_hist(arr, path: Path, title, xlabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure_dir(path.parent)
    plt.figure()
    plt.hist([a for a in arr if not np.isnan(a)], bins=50)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_bar(items, path: Path, title, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure_dir(path.parent)
    labels = [i[0] for i in items]
    vals = [i[1] for i in items]
    plt.figure(figsize=(10, max(3, 0.3*len(items))))
    y = np.arange(len(items))
    plt.barh(y, vals)
    plt.yticks(y, labels)
    plt.title(title)
    plt.xlabel(ylabel); plt.ylabel("Gene")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_scatter(Y, path: Path, title, xlabel="Dim 1", ylabel="Dim 2"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure_dir(path.parent)
    plt.figure()
    plt.scatter(Y[:,0], Y[:,1], s=12, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def write_pdf(report_md_path: Path, images, out_pdf: Path, title="3I Atlas Check‑In"):
    if FPDF is None:
        return False
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=1)
    pdf.set_font("Arial", "", 10)
    with open(report_md_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("!"):
                continue
            pdf.multi_cell(0, 5, line.rstrip())
    for img in images:
        if img and Path(img).exists():
            pdf.add_page()
            pdf.image(str(img), x=10, y=20, w=180)
            pdf.ln(5)
            pdf.set_font("Arial", "I", 9)
            pdf.cell(0, 6, str(Path(img).name), ln=1, align="C")
    ensure_dir(out_pdf.parent)
    pdf.output(str(out_pdf))
    return True

def read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def write_json(path: Path, obj):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def last_snapshot(dir_base: Path):
    files = glob.glob(str(dir_base / "*" / "snapshot.json"))
    if not files:
        return None, None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    path = Path(files[0])
    try:
        return path, read_json(path)
    except Exception:
        return path, None

def write_report_md(path: Path, info):
    ensure_dir(path.parent)
    lines = []
    lines.append(f"# 3I Atlas Check‑In — {info['meta']['stamp_local']}")
    lines.append("")
    lines.append(f"- **Pack**: `{info['meta']['pack']}`")
    lines.append(f"- **Run dir**: `{info['meta']['run_dir']}`")
    lines.append(f"- **Rows (genes)**: **{info['summary']['n_genes']}**, **Samples**: **{info['summary']['n_samples']}**")
    lines.append(f"- Gini (mean/median): **{info['summary']['gini_mean']:.4f} / {info['summary']['gini_median']:.4f}**")
    lines.append(f"- Entropyₙ (mean/median): **{info['summary']['entropy_mean']:.4f} / {info['summary']['entropy_median']:.4f}**")
    lines.append(f"- CV (mean): **{info['summary']['cv_mean']:.4f}**")
    lines.append("")
    for key in ("gini_hist","entropy_hist","top_gini_bar","pca_scatter","umap_scatter"):
        p = info["plots"].get(key)
        if p:
            lines.append(f"![{key}]({Path(p).name})")
    lines.append("")
    tg = info["tops"]["top_gini"][:10]
    lines.append("## Top specialized (by Gini) — preview")
    for (name,g,h,cv,mu) in tg:
        lines.append(f"- {name}: Gini={g:.4f}, Hₙ={h:.4f}, CV={cv:.3f}, mean={mu:.3g}")
    lines.append("")
    th = info["tops"]["top_housekeeping_high_entropy"][:10]
    lines.append("## Top housekeeping (high normalized entropy) — preview")
    for (name,g,h,cv,mu) in th:
        lines.append(f"- {name}: Hₙ={h:.4f}, Gini={g:.4f}, CV={cv:.3f}, mean={mu:.3g}")
    lines.append("")
    if info.get("deltas"):
        d = info["deltas"]
        lines.append("## Delta vs last snapshot")
        lines.append(f"- Genes: **{d.get('n_genes_delta',0):+d}**, Samples: **{d.get('n_samples_delta',0):+d}**")
        if "gini_mean_delta" in d:
            lines.append(f"- Δ Gini mean: **{d['gini_mean_delta']:+.4f}**, Δ Entropyₙ mean: **{d.get('entropy_mean_delta',0):+.4f}**")
        if d.get("changed_samples"):
            lines.append(f"- Changed sample set: +{len(d['added_samples'])} / -{len(d['removed_samples'])}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------- Main -----------------------------------------

# === Config ===
PACK_DIR = None  # Optionally set this to the exact pack folder to skip discovery.
ROOT_HINTS = [
    r"C:\Users\caleb\CNT_Lab",
    r"E:\CNT",
    r"D:\CNT",
    r"C:\CNT",
    str(Path.cwd()),
]

RUN_BASE = r"C:\Users\caleb\CNT_Lab\notebooks\archive\cnt_runs\3i_atlas_checkin"
if not Path(RUN_BASE).exists():
    RUN_BASE = str(Path.cwd() / "cnt_runs" / "3i_atlas_checkin")

STAMP = ts_utc()
RUN_DIR = ensure_dir(Path(RUN_BASE) / STAMP)

print(f"[{ts_local()}] 3I Atlas Check‑In v2 starting…")
print(f"  Run dir: {RUN_DIR}")

# ---- Discover pack across ALL roots
candidates = []

def score_candidate(path: Path) -> int:
    s = str(path).lower()
    sc = 0
    if path.is_dir(): sc += 3
    if "vector" in s and "embed" in s: sc += 5
    if "cnt_3i_atlas_all" in s: sc += 3
    if s.endswith(".csv") or s.endswith(".tsv") or s.endswith(".parquet") or s.endswith(".feather"): sc += 1
    if "vector_embedding_vector_embedding" in s: sc -= 4  # penalize duplicate suffix
    try:
        dcount = len(list_datafiles(path)) if path.is_dir() else 1
        sc += min(6, dcount)
    except Exception:
        pass
    try:
        sc += int(path.stat().st_mtime // 3600) % 10
    except Exception:
        pass
    return sc

def gather_candidates(root: Path):
    pats = [
        "**/*3i*atlas*vector*embed*",
        "**/*3i*atlas*embed*",
        "**/*3i*atlas*",
        "**/cnt_3i_atlas*",
        "**/*3i*atlas*.csv",
    ]
    for pat in pats:
        for hit in root.glob(pat):
            if ".ipynb_checkpoints" in str(hit):
                continue
            candidates.append(hit)

roots = first_existing(ROOT_HINTS)
if PACK_DIR:
    pack = normalize_pack_dir(Path(PACK_DIR))
    print(f"  PACK_DIR override: {pack}")
else:
    for r in roots:
        print(f"  Scanning: {r}")
        gather_candidates(r)
    if not candidates:
        raise SystemExit("No 3I Atlas candidates found under configured roots. Set PACK_DIR manually.")
    candidates = [normalize_pack_dir(c) for c in candidates]
    uniq = []
    seen = set()
    for c in candidates:
        key = str(c).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    candidates = uniq
    candidates.sort(key=score_candidate, reverse=True)
    pack = candidates[0]

print(f"  Candidate pack: {pack}")

# ---- Find data files; if none, try parent once
def choose_data_root(p: Path) -> Path:
    files = list_datafiles(p)
    if files:
        return p, files
    par = p.parent
    if par and par.exists():
        files = list_datafiles(par)
        if files:
            print(f"  Recovery: using parent of candidate ({par})")
            return par, files
    return p, []

pack, data_files = choose_data_root(pack)
if not data_files:
    raise SystemExit(f"No supported data files under {pack}. Set PACK_DIR to the pack root that contains out/ or data/.")

# Prefer CSV/TSV first, then parquet/feather, then NPZ/NPY
def file_rank(p: Path):
    ext = p.suffix.lower()
    order = {".csv":3, ".tsv":3, ".parquet":2, ".feather":2, ".npz":1, ".npy":1}
    return (order.get(ext,0), p.stat().st_size)

data_files.sort(key=file_rank, reverse=True)
chosen = data_files[0]
print(f"  Using data file: {chosen} ({chosen.stat().st_size/1_048_576:.2f} MiB)")

# ---- Load and infer
df_any = read_table_any(chosen, max_rows=None)
df = to_pandas(df_any)
E, gene_names, sample_names, meta = infer_matrix(df)
print(f"  Inferred matrix: genes={len(gene_names)}, samples={len(sample_names)}  format={meta['format']}")

# ---- Summarize
summary, per_gene, tops = summarize_matrix(E, gene_names, sample_names, k_top=25)

# ---- Outputs
REPORT_DIR = Path(RUN_DIR)
plots = {}

def plot_hist(arr, path: Path, title, xlabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure_dir(path.parent)
    plt.figure()
    plt.hist([a for a in arr if not np.isnan(a)], bins=50)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def plot_bar(items, path: Path, title, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure_dir(path.parent)
    labels = [i[0] for i in items]
    vals = [i[1] for i in items]
    plt.figure(figsize=(10, max(3, 0.3*len(items))))
    y = np.arange(len(items))
    plt.barh(y, vals)
    plt.yticks(y, labels)
    plt.title(title)
    plt.xlabel(ylabel); plt.ylabel("Gene")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_scatter(Y, path: Path, title, xlabel="Dim 1", ylabel="Dim 2"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure_dir(path.parent)
    plt.figure()
    plt.scatter(Y[:,0], Y[:,1], s=12, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def write_pdf(report_md_path: Path, images, out_pdf: Path, title="3I Atlas Check‑In"):
    if FPDF is None:
        return False
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=1)
    pdf.set_font("Arial", "", 10)
    with open(report_md_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("!"):
                continue
            pdf.multi_cell(0, 5, line.rstrip())
    for img in images:
        if img and Path(img).exists():
            pdf.add_page()
            pdf.image(str(img), x=10, y=20, w=180)
            pdf.ln(5)
            pdf.set_font("Arial", "I", 9)
            pdf.cell(0, 6, str(Path(img).name), ln=1, align="C")
    ensure_dir(out_pdf.parent)
    pdf.output(str(out_pdf))
    return True

def to_csv(path: Path, rows, header):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(lambda x: str(x).replace(",",";"), r)) + "\n")

# CSVs
to_csv(Path(REPORT_DIR/"top_gini_genes.csv"), tops["top_gini"], ["gene","gini","H_norm","cv","mean"])
to_csv(Path(REPORT_DIR/"top_specialized_low_entropy.csv"), tops["top_specialized_low_entropy"], ["gene","gini","H_norm","cv","mean"])
to_csv(Path(REPORT_DIR/"top_housekeeping_high_entropy.csv"), tops["top_housekeeping_high_entropy"], ["gene","gini","H_norm","cv","mean"])
to_csv(Path(REPORT_DIR/"summary_stats.csv"), [[k, v] for k, v in summary.items()], ["metric","value"])

# Plots
plot_hist(per_gene["gini"], Path(REPORT_DIR/"plots/gini_hist.png"), "Gini distribution (gene specialization)", "Gini")
plots["gini_hist"] = str(Path(REPORT_DIR/"plots/gini_hist.png"))
plot_hist(per_gene["H_norm"], Path(REPORT_DIR/"plots/entropy_hist.png"), "Normalized entropy across samples", "H_norm")
plots["entropy_hist"] = str(Path(REPORT_DIR/"plots/entropy_hist.png"))
plot_bar(tops["top_gini"], Path(REPORT_DIR/"plots/top_gini_bar.png"), "Top specialized genes (by Gini)", "Gini")
plots["top_gini_bar"] = str(Path(REPORT_DIR/"plots/top_gini_bar.png"))

# Embeddings
pca_pts, pca_var = try_pca(E, n=2, random_state=42)
if pca_pts is not None:
    plot_scatter(pca_pts, Path(REPORT_DIR/"plots/pca_scatter.png"),
                 f"PCA on samples (var={sum(pca_var):.2%})", "PC1", "PC2")
    plots["pca_scatter"] = str(Path(REPORT_DIR/"plots/pca_scatter.png"))
else:
    print("  PCA not available or failed; skipping PCA plot.")
umap_pts = try_umap(E, n=2, random_state=42)
if umap_pts is not None:
    plot_scatter(umap_pts, Path(REPORT_DIR/"plots/umap_scatter.png"),
                 "UMAP on samples", "UMAP-1", "UMAP-2")
    plots["umap_scatter"] = str(Path(REPORT_DIR/"plots/umap_scatter.png"))

# Snapshot & delta
SNAPSHOT_PATH = Path(REPORT_DIR/"snapshot.json")
prev_path, prev = last_snapshot(Path(RUN_BASE))
deltas = None
if prev:
    deltas = {
        "n_genes_delta": summary["n_genes"] - int(prev.get("summary",{}).get("n_genes", 0)),
        "n_samples_delta": summary["n_samples"] - int(prev.get("summary",{}).get("n_samples", 0)),
        "gini_mean_delta": summary["gini_mean"] - float(prev.get("summary",{}).get("gini_mean", 0.0)),
        "entropy_mean_delta": summary["entropy_mean"] - float(prev.get("summary",{}).get("entropy_mean", 0.0)),
        "cv_mean_delta": summary["cv_mean"] - float(prev.get("summary",{}).get("cv_mean", 0.0)),
        "changed_samples": False,
        "added_samples": [],
        "removed_samples": [],
    }
    try:
        prev_samples = set(prev.get("sample_names", []))
        cur_samples = set(sample_names)
        add = sorted(cur_samples - prev_samples)
        rem = sorted(prev_samples - cur_samples)
        if add or rem:
            deltas["changed_samples"] = True
            deltas["added_samples"] = add
            deltas["removed_samples"] = rem
    except Exception:
        pass
    write_json(Path(REPORT_DIR/"delta_summary.json"), deltas)
    print(f"  Δ written: {Path(REPORT_DIR/'delta_summary.json')}")
else:
    print("  No prior snapshot found; this will serve as the baseline.")

snapshot = {
    "meta": {
        "stamp_utc": ts_utc(),
        "stamp_local": ts_local(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "pack_dir": str(pack),
        "data_file": str(chosen),
    },
    "summary": summary,
    "sample_names": sample_names[:5000],
    "top_gini": tops["top_gini"],
    "top_housekeeping_high_entropy": tops["top_housekeeping_high_entropy"],
}
write_json(SNAPSHOT_PATH, snapshot)

# Report
info = {
    "meta": {
        "stamp_local": ts_local(),
        "pack": str(pack),
        "run_dir": str(REPORT_DIR),
    },
    "summary": summary,
    "tops": tops,
    "deltas": deltas,
    "plots": plots,
}
REPORT_MD = Path(REPORT_DIR/"report.md")
write_report_md(REPORT_MD, info)
print(f"  Wrote: {REPORT_MD}")

# Lightweight PDF
REPORT_PDF = Path(REPORT_DIR/"report.pdf")
ok_pdf = write_pdf(REPORT_MD, images=[plots.get("gini_hist"), plots.get("entropy_hist"),
                                      plots.get("top_gini_bar"), plots.get("pca_scatter"), plots.get("umap_scatter")],
                   out_pdf=REPORT_PDF, title="3I Atlas Check‑In")
if ok_pdf:
    print(f"  PDF:   {REPORT_PDF}")
else:
    print("  PDF:   (skipped; fpdf missing)")

print(f"[{ts_local()}] Done. — The field answers when you listen.")
