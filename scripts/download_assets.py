"""Download pinned weights into assets/weights/ and write assets/manifest.json with a sha256
per file. The only module allowed to touch the network; never run during an edit.

Usage: python scripts/download_assets.py [--profile klein-4b-bf16] [--sam sam2] [--grounder gdino-base] [--only renderer sam grounder]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "assets" / "weights"
MANIFEST = ROOT / "assets" / "manifest.json"

# Klein in the Diffusers layout only; the 7.75 GB single-file duplicate is skipped.
KLEIN = {
    "repo_id": "black-forest-labs/FLUX.2-klein-4B",
    "allow_patterns": ["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*", "LICENSE*", "README.md"],
    "ignore_patterns": ["*.ckpt", "flux-2-klein-4b.safetensors", "*.png", "*.jpg", "assets/*"],
}
SAM = {
    "sam2": {"repo_id": "facebook/sam2.1-hiera-large", "backend": "sam2"},
    "sam3": {"repo_id": "facebook/sam3", "backend": "sam3_tracker"},
}
# Selection from words. sam3 is the same download as above and also serves concept prompts.
GROUNDERS = {
    "gdino-base": {"repo_id": "IDEA-Research/grounding-dino-base", "backend": "dino", "ignore_patterns": ["*.bin", "*.pt", "*.ckpt", "*.msgpack", "*.h5"]},
    "sam3": {"repo_id": "facebook/sam3", "backend": "sam3", "ignore_patterns": ["*.pt"]},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def hash_tree(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if rel.parts and rel.parts[0] == ".cache":  # Hugging Face's own download records: timestamps, differ per download
            continue
        if p.is_file() and not p.is_symlink():
            out[str(rel)] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="klein-4b-bf16")
    ap.add_argument("--sam", default="sam2", choices=sorted(SAM))
    ap.add_argument("--grounder", default="gdino-base", choices=sorted(GROUNDERS))
    ap.add_argument("--only", nargs="+", choices=["renderer", "sam", "grounder"], default=["renderer", "sam", "grounder"])
    args = ap.parse_args()
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"schema_version": 1, "profiles": {}}
    manifest.pop("note", None)

    t0 = time.time()
    klein_dir = WEIGHTS / "flux2-klein-4b"
    if "renderer" in args.only:
        snapshot_download(KLEIN["repo_id"], local_dir=klein_dir, allow_patterns=KLEIN["allow_patterns"], ignore_patterns=KLEIN["ignore_patterns"])
    t1 = time.time()
    sam_dir = WEIGHTS / args.sam
    if "sam" in args.only:
        snapshot_download(SAM[args.sam]["repo_id"], local_dir=sam_dir)
    t2 = time.time()
    if "grounder" in args.only:
        g = GROUNDERS[args.grounder]
        g_dir = WEIGHTS / args.grounder
        snapshot_download(g["repo_id"], local_dir=g_dir, ignore_patterns=g["ignore_patterns"])
        manifest["profiles"][f"grounder-{args.grounder}"] = {
            "kind": "grounder", "repo_id": g["repo_id"], "backend": g["backend"], "path": str(g_dir.relative_to(ROOT)),
            "files": hash_tree(g_dir), "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "download_seconds": round(time.time() - t2, 1)}
    if "renderer" not in args.only and "sam" not in args.only:
        manifest["last_verified"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        print(f"manifest written: grounder {args.grounder}")
        return

    manifest["profiles"][args.profile] = {
        "kind": "renderer", "repo_id": KLEIN["repo_id"], "path": str(klein_dir.relative_to(ROOT)),
        "dtype": "bfloat16", "pipelines": ["Flux2KleinPipeline", "Flux2KleinInpaintPipeline"],
        "files": hash_tree(klein_dir), "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "download_seconds": round(t1 - t0, 1),
    }
    manifest["profiles"][args.sam] = {
        "kind": "segmenter", "repo_id": SAM[args.sam]["repo_id"], "backend": SAM[args.sam]["backend"],
        "path": str(sam_dir.relative_to(ROOT)), "files": hash_tree(sam_dir),
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "download_seconds": round(t2 - t1, 1),
    }
    manifest["last_verified"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    total = sum(f["bytes"] for prof in manifest["profiles"].values() for f in prof["files"].values())
    print(f"manifest written: {len(manifest['profiles'])} profiles, {total / 1e9:.2f} GB, klein {t1 - t0:.0f}s, sam {t2 - t1:.0f}s")


if __name__ == "__main__":
    main()
