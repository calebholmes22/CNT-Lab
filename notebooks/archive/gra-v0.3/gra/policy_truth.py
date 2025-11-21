import re
from numpy import array, max as npmax
from sklearn.metrics.pairwise import cosine_similarity

ALIASES_RX = {
    "24/7 access":       [r"\b24[\/x]7\b", r"\balways-?on\b", r"\b24\/7 access\b"],
    "hallucinations":    [r"\bhallucin\w*\b", r"\bfabricat\w*\b", r"\bmade-?up\b"],
    "bias and fairness": [r"\bbias(ed)?\b", r"\bfairness\b", r"\bdisparit(y|ies)\b"],
    "safety guardrail":  [r"\bsafety guardrail(s)?\b", r"\bguardrail(s)? for safety\b"],
    "false invariance":  [r"\bfalse invariance\b", r"\bspurious invariance\b"],
    "invariance to rewording":[r"\binvariance to rewording\b", r"\bprompt-?invariant\b", r"\bgauge-?restor\w*\b"],
    "faster triage":[r"\bfaster\b.*\btriage\b", r"\breduce wait time(s)?\b"],
}

def split_2x2(text: str):
    parts = re.split(r"\bRisks:\s*", text, flags=re.I)
    if len(parts)!=2: return [], []
    bpart = re.sub(r"\bBenefits:\s*", "", parts[0], flags=re.I)
    rpart = parts[1]
    def parse(block):
        out=[]
        for line in block.splitlines():
            if line.strip().startswith("-"):
                raw = line.strip()[1:].strip()
                ids = [int(x) for x in re.findall(r"\[(\d+)\]", raw)]
                txt = re.sub(r"\s*(\[\d+\])+", "", raw).strip()
                out.append((txt, ids))
        return out
    return parse(bpart), parse(rpart)

def alias_or_stem_match(bullet_txt, source_txt):
    if bullet_txt.lower() in source_txt.lower(): return True
    for rx in ALIASES_RX.get(bullet_txt, []):
        if re.search(rx, source_txt, flags=re.I): return True
    return False

def hybrid_truth(restored_2x2: str, sources: list[str], embed, sim_thr=0.50, min_cites=1):
    if not sources:
        return False, {"reason":"no_sources", "failed_bullets":[]}
    B, R = split_2x2(restored_2x2)
    failed = []
    def side_ok(side):
        ok=0
        for txt, ids in side:
            if len(ids) < min_cites: failed.append(("no_citations", txt)); continue
            cited = [sources[i-1] for i in ids if 1 <= i <= len(sources)]
            if not cited: failed.append(("bad_ids", txt)); continue
            try:
                sb = cosine_similarity(array(embed([txt])), array(embed(cited)))[0]
                sem_ok = float(npmax(sb)) >= sim_thr
            except Exception:
                sem_ok = False
            phr_ok = any(alias_or_stem_match(txt, s) for s in cited)
            if sem_ok or phr_ok: ok += 1
            else: failed.append(("weak_support", txt))
        return ok
    okB = side_ok(B); okR = side_ok(R)
    return (okB >= 2 and okR >= 2), {"okB":okB, "okR":okR, "failed_bullets":failed}
