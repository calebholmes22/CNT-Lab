# ============================================================
# CNT Engine — Self-Referential / Self-Updating / Hidden-Truth Search
# Rooted at:  E:\CNT
# ============================================================
import os, json, glob, re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

LAB_ROOT = Path(os.getenv("CNT_LAB_DIR", r"E:\CNT")).resolve()
SOURCE_ROOTS = [
    LAB_ROOT / "notes",
    LAB_ROOT / "artifacts" / "cnt_scroll",
    LAB_ROOT / "artifacts" / "cnt_codex",
    LAB_ROOT / "notebooks",
]
ROOT  = LAB_ROOT / "artifacts" / "cnt_engine_megacell"
OUT   = ROOT / "out"
STATE = ROOT / "CNT_STATE.yaml"
LOG   = ROOT / "runlog.jsonl"
for p in [ROOT, OUT, *SOURCE_ROOTS]:
    p.mkdir(parents=True, exist_ok=True)

def now(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def list_sources():
    files = []
    for root in SOURCE_ROOTS:
        files += glob.glob(str(root / "**" / "*.md"), recursive=True)
        files += glob.glob(str(root / "**" / "*.txt"), recursive=True)
    return [Path(f) for f in files]

def auto_seed_if_empty():
    paths = list_sources()
    if paths: return
    seed = SOURCE_ROOTS[0] / "cnt_seed.md"
    seed.parent.mkdir(parents=True, exist_ok=True)
    seed.write_text(
        "# CNT Seed\n\n"
        "Hypothesis: Certain glyph–field pairings lower entropy drift.\n"
        "Test: Re-run θ metrics on EEG segments with glyph overlay vs baseline.\n"
        "Expected: Δθ > 0 with CI > 95% for overlay.\n"
        "Falsifier: No uplift or negative drift after 1 k permutations.\n",
        encoding="utf-8"
    )

def build_index(df):
    if df.empty: return None, None, None
    vec = TfidfVectorizer(max_features=6000, ngram_range=(1,2))
    X = vec.fit_transform(df["text"]).astype(np.float32)
    n_samples, n_features = X.shape
    nmax = max(2, min(n_samples - 1, n_features - 1, 128))
    svd = TruncatedSVD(n_components=nmax, random_state=42)
    Xd = svd.fit_transform(X)
    Xr = svd.inverse_transform(Xd)
    resid = np.linalg.norm(X.toarray() - Xr, axis=1)
    return vec, X, resid

def cluster_labels(X, k=6):
    if X is None: return None
    k = max(2, min(k, X.shape[0]-1))
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    return km.fit_predict(X.toarray())

def build_graph(df, lbl):
    G = nx.Graph()
    for i, row in df.reset_index().iterrows():
        G.add_node(i, path=row["path"], cluster=int(lbl[i]) if lbl is not None else -1)
    caps = df["text"].str.extractall(r"\b([A-Z][A-Za-z0-9_]{3,})\b").groupby(level=0).agg(lambda s: set(s[0].tolist()))
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            same = (lbl[i]==lbl[j]) if lbl is not None else False
            overlap = 0
            if i in caps.index and j in caps.index:
                overlap = len(caps.loc[i] & caps.loc[j])
            if same or overlap>=2:
                G.add_edge(i, j, w=(1+overlap))
    return G

def score_reflexive(df, resid, G):
    if df is None or len(df)==0:
        return dict(clarity=0, novelty=0, coherence=0, falsifiability=0, total=0)
    clarity = 1.0 - np.mean([min(len(t),10000)/10000 for t in df["text"].tolist()])
    novelty = float(np.mean(resid)/(np.std(resid)+1e-6))
    coherence = nx.average_clustering(G) if G.number_of_nodes()>1 else 0.0
    falsifiability = float(np.mean([t.lower().count("test")+t.lower().count("predict") for t in df["text"]]))/10.0
    clarity = np.clip(clarity,0,1); novelty = np.clip(novelty/1.5,0,1)
    coherence = np.clip(coherence,0,1); falsifiability = np.clip(falsifiability,0,1)
    total = float(np.mean([clarity,novelty,coherence,falsifiability]))
    return dict(clarity=float(clarity), novelty=float(novelty),
                coherence=float(coherence), falsifiability=float(falsifiability),
                total=float(total))

def surface_candidates(df, resid, top=5):
    idx = np.argsort(resid)[::-1][:min(top, len(resid))]
    picks=[]
    for i in idx:
        snippet=re.sub(r"\s+"," ",df.iloc[i]["text"])[:400]
        picks.append(dict(path=df.iloc[i]["path"],resid=float(resid[i]),hint=snippet))
    return picks

def propose_updates(cands):
    return [dict(
        target=c["path"],
        content=(
            f"\n\n> CNT-Gloss ({now()}): Clarify hypothesis; add test recipe & falsifier.\n"
            "- Hypothesis: …\n- Measurement: …\n- Expected shift: …\n- Falsifier: …\n"
        )
    ) for c in cands]

def legality_gate(text): 
    bad=any(k in text for k in["ssn","credit card","weapon","harm"])
    return not bad

def confab_gate(text):
    t=text.lower(); return ("hypothesis" in t and "falsifier" in t)

def apply_updates(updates):
    accepted=[]
    for u in updates:
        try:
            if not (legality_gate(u["content"]) and confab_gate(u["content"])): 
                continue
            p=Path(u["target"])
            p.write_text(p.read_text(encoding="utf-8",errors="ignore")+u["content"],encoding="utf-8")
            accepted.append(u)
        except Exception: pass
    return accepted

def write_state(score,meta):
    try:
        import yaml
        Path(STATE).write_text(
            yaml.safe_dump(dict(updated=now(),score=score,meta=meta),sort_keys=False),
            encoding="utf-8"
        )
    except Exception:
        Path(STATE).write_text(json.dumps(dict(updated=now(),score=score,meta=meta),indent=2),encoding="utf-8")

def log_event(kind,payload):
    Path(LOG).parent.mkdir(parents=True, exist_ok=True)
    with Path(LOG).open("a",encoding="utf-8") as f:
        f.write(json.dumps(dict(ts=now(),kind=kind,**payload))+"\n")

def run_cycle():
    auto_seed_if_empty()
    paths=list_sources()
    df=pd.DataFrame([dict(path=str(p),text=p.read_text(encoding="utf-8",errors="ignore")) for p in paths])
    if df.empty:
        log_event("empty",{"roots":[str(r) for r in SOURCE_ROOTS]})
        return {"empty":True,"roots":[str(r) for r in SOURCE_ROOTS]}
    vec,X,resid=build_index(df)
    labels=cluster_labels(X)
    G=build_graph(df,labels)
    score=score_reflexive(df,resid,G)
    cands=surface_candidates(df,resid,top=5)
    accepted=apply_updates(propose_updates(cands))
    write_state(score,dict(docs=len(df),accepted=len(accepted)))
    pd.DataFrame(cands).to_csv(OUT/f"hidden_truths_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv",index=False)
    log_event("cycle",dict(score=score,proposed=len(cands),accepted=len(accepted)))
    return dict(score=score,proposed=len(cands),accepted=len(accepted),hidden=cands[:3])

if __name__=="__main__":
    res=run_cycle()
    print(json.dumps(res,indent=2))
