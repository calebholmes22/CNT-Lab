# CALM Eval (minimal) — drop-in module
import os, json, numpy as np, pandas as pd
from pathlib import Path

def _safe_csv(p):
    try:
        if p.exists() and p.stat().st_size>0:
            return pd.read_csv(p)
    except Exception:
        pass
    return None

def evaluate(bundle_root):
    rep = Path(r"E:\CNT\artifacts\cog_alphabet_report_hybrid_v1".replace("\\","/"))
    out = {}
    # Identity / Function
    eoec = _safe_csv(rep/"generalization"/"eoec_iaf_summary.csv")
    task = _safe_csv(rep/"generalization"/"task_fbcsp_summary.csv")
    if eoec is not None and len(eoec): out["EOEC_mean"] = float(eoec["acc"].mean())
    if task is not None and len(task): out["TaskAUC_mean"] = float(task["AUC_task"].mean())
    # Grammar (predictive)
    k3 = Path(r"E:\CNT\artifacts\cog_alphabet_hybrid_k3".replace("\\","/"))/"state_assignments.csv"
    if k3.exists():
        L = pd.read_csv(k3)["state"].astype(int).to_numpy()
        K = int(L.max()+1)
        T = np.zeros((K,K)); 
        for a,b in zip(L[:-1], L[1:]): T[a,b]+=1
        eps=1e-6; T=(T+eps)/(T.sum(axis=1,keepdims=True)+eps*K)
        pi=T.sum(axis=0); pi=pi/(pi.sum()+1e-12)
        H_T1 = -np.mean([np.log(T[a,b]+1e-12) for a,b in zip(L[:-1],L[1:])])
        H_pi = -np.mean([np.log(pi[b]+1e-12) for b in L[1:]])
        out["DXENT_K3"] = float(H_pi - H_T1)
    # Mechanism (MI & CMI) and generator
    gb = rep/"groundbreaking"
    for f in ["microstate_epoch_stats.json","microstate_epoch_stats_subjects.json","cmi_microstate_conditioned.json","generative_adequacy.json"]:
        p = gb/f
        if p.exists():
            try: out[f.replace(".json","")] = json.loads(p.read_text(encoding="utf-8"))
            except Exception: pass
    # S1 zoom summary
    zsum = rep/"analysis"/"zoom_S1_summary.json"
    if zsum.exists():
        out["zoom_S1"] = json.loads(zsum.read_text(encoding="utf-8"))
    # Save and return
    outp = rep/"analysis"/"calm_eval_min.json"; outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out

if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv)>1 else Path(".")
    res = evaluate(root)
    print(json.dumps(res, indent=2))
