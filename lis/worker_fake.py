"""A worker that speaks the render protocol without any model. For client tests and UI work.

Each job: emits `steps` progress messages spaced by --step-seconds, then writes the input
image back out (tinted inside the mask if one is given) and reports done. Honors cancel.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--step-seconds", type=float, default=float(os.environ.get("LIS_FAKE_STEP_SECONDS", "0.05")))
    ap.add_argument("--load-seconds", type=float, default=float(os.environ.get("LIS_FAKE_LOAD_SECONDS", "0.0")))
    ap.add_argument("--hang", action="store_true", help="never answer jobs (for timeout tests)")
    args = ap.parse_args()
    out = os.fdopen(os.dup(1), "w", buffering=1)
    sys.stdout = sys.stderr
    q: "queue.Queue[dict]" = queue.Queue()
    cancels: set[str] = set()

    def reader():
        for line in sys.stdin:
            if line.strip():
                q.put(json.loads(line))
        q.put({"type": "eof"})

    threading.Thread(target=reader, daemon=True).start()

    def send(**m):
        out.write(json.dumps(m) + "\n")
        out.flush()

    while True:
        msg = q.get()
        t = msg.get("type")
        if t in ("eof", "quit"):
            return
        if t == "load":
            time.sleep(args.load_seconds)
            send(type="ready", profile_id="fake", load_ms=int(args.load_seconds * 1000), vram_bytes=0, offload=False, is_distilled=True)
        elif t == "cancel":
            cancels.add(msg.get("id"))
        elif t == "job":
            if args.hang:
                continue
            jid, steps = msg["id"], int(msg.get("steps", 4))
            cancelled = False
            for i in range(steps):
                time.sleep(args.step_seconds)
                while not q.empty():  # pick up cancels that arrived mid-job
                    m2 = q.get()
                    if m2.get("type") == "cancel":
                        cancels.add(m2.get("id"))
                if jid in cancels:
                    cancelled = True
                    break
                send(type="progress", id=jid, step=i + 1, of=steps)
            if cancelled:
                send(type="failed", id=jid, cause="cancelled", detail="cancelled by the user")
                continue
            from PIL import Image
            import numpy as np

            if msg.get("strategy") == "generate":
                rng = np.random.default_rng(int(msg.get("seed", 0)))
                w, h = int(msg["width"]), int(msg["height"])
                yy, xx = np.mgrid[0:h, 0:w]
                c = rng.integers(0, 255, size=(2, 3))
                t = (xx / max(1, w - 1))[..., None]
                img = (c[0] * (1 - t) + c[1] * t).astype(np.uint8)
                if msg.get("ref_paths"):  # a reference: its colours, so tests can tell it was used
                    ref = np.asarray(Image.open(msg["ref_paths"][0]).convert("RGB").resize((w, h)))
                    img = (ref * 0.7 + img * 0.3).astype(np.uint8)
            else:
                img = np.asarray(Image.open(msg["image_path"]).convert("RGB")).copy()
            if msg.get("strategy") == "generate":
                pass
            elif msg.get("mask_path"):
                m = np.asarray(Image.open(msg["mask_path"]).convert("L")) >= 128
                img[m] = (img[m] * 0.5 + np.array([200, 120, 40]) * 0.5).astype(np.uint8)
            else:
                img = (img * 0.5 + np.array([200, 120, 40]) * 0.5).astype(np.uint8)
            Image.fromarray(img).save(msg["out_path"])
            send(type="done", id=jid, out_path=msg["out_path"], timing_ms={"encode": 1, "denoise": int(steps * args.step_seconds * 1000), "decode": 1, "total": 2},
                 peak_vram_bytes=0, strategy=msg.get("strategy"), out_size=[img.shape[1], img.shape[0]])


if __name__ == "__main__":
    main()
