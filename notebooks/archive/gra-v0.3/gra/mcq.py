import re
from collections import Counter

def extract_label(text: str):
    m = re.search(r"\b([ABCD])\b", text.strip())
    if m: return m.group(1)
    t=text.lower()
    if "paris" in t:  return "B"
    if "berlin" in t: return "A"
    if "rome" in t:   return "C"
    if "madrid" in t: return "D"
    return None

def consensus(labels, thresh=0.60):
    labels = [x for x in labels if x]
    if not labels: return None, 0.0, Counter()
    c = Counter(labels)
    lab, cnt = c.most_common(1)[0]
    frac = cnt/len(labels)
    return (None, frac, c) if frac < thresh else (lab, frac, c)
