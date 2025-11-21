"""
Utilities: hash-chaining for artifacts, JSON logging.
"""
import json, hashlib, time, os

def hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def write_hashed_json(obj, path):
    payload = {
        "data": obj,
        "ts": time.time(),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    h = hashlib.sha256(blob).hexdigest()
    with open(path, "wb") as f:
        f.write(blob)
    with open(path + ".sha256", "w") as f:
        f.write(h+"
")
    return h
