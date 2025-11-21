# scripts/hash_file.py
import sys, hashlib, pathlib

p = pathlib.Path(sys.argv[1])
h = hashlib.sha256(p.read_bytes()).hexdigest()
print(h)
