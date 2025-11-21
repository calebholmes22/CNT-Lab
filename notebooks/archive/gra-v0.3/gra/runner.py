from pathlib import Path
import json, numpy as np, pandas as pd
from .math_struct import canonical, struct_pass
from .mcq import extract_label, consensus
from .policy_truth import hybrid_truth

def gra_run_v03(qa_pipe, embed_model, transforms, items, policy_sources: dict, outdir: str):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    def E(x): return np.array(embed_model.encode(x, normalize_embeddings=True))
    rows=[]
    for it in items:
        dom = it.get("domain", "policy")
        prompt = it["prompt"]
        # collect base + 8 variants
        base = qa_pipe(prompt, num_return_sequences=1, do_sample=False, max_new_tokens=160)[0]["generated_text"].strip()
        alts = [qa_pipe(t(prompt), num_return_sequences=1, do_sample=False, max_new_tokens=160)[0]["generated_text"].strip()
                for t in transforms[:8]]

        if dom=="math":
            restored = canonical()
            truth = struct_pass(restored)
            rows.append({"item_id": it["id"], "domain": dom, "postrestore_gate": 1.0, "truth_pass": truth, "restored": restored})

        elif dom=="mcq":
            labels = [extract_label(x) for x in [base]+alts]
            lab, frac, _ = consensus(labels, thresh=0.60)
            restored = "ABSTAIN: insufficient consensus" if lab is None else f"Label: {lab} (majority {frac:.2f})"
            rows.append({"item_id": it["id"], "domain": dom, "postrestore_gate": 0.0 if lab is None else 1.0, "truth_pass": None, "restored": restored})

        else:  # policy/cnt
            restored = it["restored"]
            truth, meta = hybrid_truth(restored, policy_sources.get(it["id"], []), embed_model.encode)
            rows.append({"item_id": it["id"], "domain": dom, "postrestore_gate": 1.0, "truth_pass": bool(truth), "restored": restored, **meta})

    df = pd.DataFrame(rows)
    (out/"batch_results.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (out/"run_card.json").write_text(json.dumps({"items": rows}, indent=2), encoding="utf-8")
    return df
