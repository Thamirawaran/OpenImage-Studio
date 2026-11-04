"""Command line: the end-to-end gate before the UI exists.

  lis edit photo.jpg --click 400,300 --protect-click 120,90 --instruction "make the jacket copper" [--fake]
  lis verify            full sha256 of every weight file against assets/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _xy(s: str) -> tuple[int, int, int]:
    parts = [int(v) for v in s.split(",")]
    if len(parts) == 2:
        parts.append(1)
    if len(parts) != 3 or parts[2] not in (0, 1):
        raise argparse.ArgumentTypeError("expected x,y or x,y,label with label 0 or 1")
    return parts[0], parts[1], parts[2]


def cmd_verify(args) -> int:
    manifest = json.loads(Path(args.manifest).read_text())
    bad = 0
    t0 = time.time()
    for prof, v in manifest["profiles"].items():
        base = ROOT / v["path"]
        for rel, meta in v["files"].items():
            if rel.startswith(".cache/"):  # Hugging Face download records, not model files
                continue
            p = base / rel
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            ok = "sha256:" + h.hexdigest() == meta["sha256"] and p.stat().st_size == meta["bytes"]
            bad += not ok
            if not ok or args.verbose:
                print(("ok   " if ok else "FAIL ") + f"{prof}/{rel}")
    if bad == 0:
        manifest["last_verified"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    print(f"{'all files match' if bad == 0 else str(bad) + ' files differ'} ({time.time() - t0:.0f}s)")
    return 1 if bad else 0


def cmd_edit(args) -> int:
    from lis.engine import Engine, EngineConfig
    from lis.render import WorkerConfig

    if args.fake:
        worker = WorkerConfig(module="lis.worker_fake", weights="", python=sys.executable)
        seg, seg_w = "fake", ""
    else:
        for k, v in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}.items():
            os.environ.setdefault(k, v)
        worker = WorkerConfig(module="lis.worker_klein", weights=str(ROOT / "assets" / "weights" / "flux2-klein-4b"), offload=args.offload, profile=args.profile, python=sys.executable)
        seg, seg_w = args.segmenter, str(ROOT / "assets" / "weights" / args.segmenter)
    eng = Engine(EngineConfig(projects_dir=Path(args.projects), manifest_path=Path(args.manifest), worker=worker, segmenter=seg, segmenter_weights=seg_w,
                              profile_id="fake" if args.fake else "klein-4b-bf16", color_match=not args.no_color_match))
    if args.fake:
        eng.manifest.setdefault("profiles", {})["fake"] = {"files": {}}
    try:
        t0 = time.time()
        o = eng.open(args.photo)
        pid = o["project_id"]
        print(f"opened {o['width']}x{o['height']} as {pid} ({time.time() - t0:.1f}s incl. embeddings)", file=sys.stderr)
        t0 = time.time()
        sel = eng.select(pid, "target", args.click, tuple(int(v) for v in args.box.split(",")) if args.box else None)
        if not sel["candidates"]:
            print("nothing selected here, try another click or a box", file=sys.stderr)
            return 2
        print(f"{len(sel['candidates'])} candidates in {time.time() - t0:.2f}s, scores {[c['score'] for c in sel['candidates']]}", file=sys.stderr)
        pick = sel["candidates"][0]
        if args.pick == "largest":
            import numpy as _np
            from PIL import Image as _Image
            pick = max(sel["candidates"], key=lambda c: int((_np.asarray(_Image.open(c["png_path"])) > 127).sum()))
        target = eng.commit_mask(pid, "target", pick["candidate_id"], "target", "target")["mask_id"]
        protected = []
        for i, pc in enumerate(args.protect_click):
            s = eng.select(pid, f"prot{i}", [pc], None)
            if s["candidates"]:
                protected.append(eng.commit_mask(pid, f"prot{i}", s["candidates"][0]["candidate_id"], "protected", f"protected-{i}")["mask_id"])
        for i, png in enumerate(args.protect):
            protected.append(eng.commit_mask(pid, f"protpng{i}", None, "protected", Path(png).stem, brush_add_png=png)["mask_id"])
        plan = {"schema_version": 2, "project_id": pid, "parent_revision": None,
                "operation": {"kind": "native_instruction_edit", "target_mask_id": target, "instruction": args.instruction, "band_px": args.band, "strategy": args.strategy, "blend": args.blend},
                "protected_mask_ids": protected, "seed": args.seed, "steps": args.steps, "strength": args.strength,
                "backend_profile_id": "fake" if args.fake else "klein-4b-bf16", "network_policy": "offline"}
        cancel = threading.Event()
        result = None
        for msg in eng.render(plan, cancel):
            if msg["type"] == "progress":
                print(f"  step {msg['step']}/{msg['of']}" + (f" ({msg['note']})" if msg.get("note") else ""), file=sys.stderr)
            else:
                result = msg
        print(json.dumps(result, indent=2))
        if result["status"] == "ok" and args.out:
            ex = eng.export(pid, result["revision_id"], args.out, "png")
            print(f"exported {args.out} ({ex['bytes']} bytes, check {ex['protected_check']['status']})", file=sys.stderr)
        return 0 if result["status"] == "ok" else 1
    finally:
        eng.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="lis")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", default=str(ROOT / "assets" / "manifest.json"))
    v.add_argument("--verbose", action="store_true")
    v.set_defaults(fn=cmd_verify)
    e = sub.add_parser("edit")
    e.add_argument("photo")
    e.add_argument("--click", type=_xy, action="append", default=[], help="x,y[,label] on the target; repeatable")
    e.add_argument("--box", help="x0,y0,x1,y1 on the target")
    e.add_argument("--protect-click", type=_xy, action="append", default=[], help="x,y of an object to protect; repeatable")
    e.add_argument("--protect", action="append", default=[], help="PNG mask to protect; repeatable")
    e.add_argument("--instruction", required=True)
    e.add_argument("--strategy", default="auto", choices=["auto", "edit", "inpaint"])
    e.add_argument("--seed", type=int, default=1234)
    e.add_argument("--steps", type=int, default=4)
    e.add_argument("--strength", type=float, default=0.8)
    e.add_argument("--band", type=int, default=8)
    e.add_argument("--blend", default="auto", choices=["auto", "tight", "context"])
    e.add_argument("--pick", default="best", choices=["best", "largest"], help="which SAM candidate to keep for the target")
    e.add_argument("--no-color-match", action="store_true")
    e.add_argument("--segmenter", default="sam2", choices=["sam2", "sam3_tracker"])
    e.add_argument("--offload", action="store_true", help="alias for --profile offload")
    e.add_argument("--profile", default="auto", help="memory profile: auto (default: resident, fast or group, by free video memory) | group | group_fp8 | fast | resident | offload | offload_fp8")
    e.add_argument("--fake", action="store_true", help="fake worker and segmenter; no models")
    e.add_argument("--projects", default=str(ROOT / "projects"))
    e.add_argument("--manifest", default=str(ROOT / "assets" / "manifest.json"))
    e.add_argument("--out", help="export the result PNG here")
    e.set_defaults(fn=cmd_edit)
    args = ap.parse_args(argv)
    if args.cmd == "edit" and not args.click and not args.box:
        ap.error("edit needs at least one --click or a --box")
    raise SystemExit(args.fn(args))


if __name__ == "__main__":
    main()
