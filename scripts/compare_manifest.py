"""Compare downloaded weight hashes with the pinned manifest. Exit 1 when a pinned file differs."""
import json
import sys

pinned = json.load(open(sys.argv[1]))["profiles"]
now = json.load(open(sys.argv[2]))["profiles"]
bad = checked = 0
for prof, spec in pinned.items():
    if prof not in now:
        continue
    for name, f in spec.get("files", {}).items():
        if name.startswith(".cache/"):  # download records with timestamps, not model files
            continue
        g = now[prof]["files"].get(name)
        if g is None:
            continue
        checked += 1
        if g["sha256"] != f["sha256"]:
            bad += 1
            print(f"differs: {prof}/{name}")
print(f"{checked} files checked against the tested versions, {bad} differ")
sys.exit(1 if bad else 0)
