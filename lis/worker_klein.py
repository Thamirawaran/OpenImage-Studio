"""Render worker process: FLUX.2 Klein 4B through Diffusers.

Loads the pipeline once, serves jobs from stdin, two strategies over shared components:
`edit` (Flux2KleinPipeline) and `inpaint` (Flux2KleinInpaintPipeline with the repaint
mask). Accepts only pre-sized working-region images (multiples of 16, at most 1024x1024
area). Softness, protection and the proof live in composite.py and engine.py.

Protocol (one JSON object per line; stdout carries only these, everything else to stderr):
  in : {"type":"load"} | {"type":"job",...} | {"type":"cancel","id":...} | {"type":"quit"}
  out: {"type":"ready",...} | {"type":"progress",...} | {"type":"done",...} | {"type":"failed",...}
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

MAX_AREA = 1024 * 1024
MIN_FREE_BYTES = int(1.2 * 2**30)
PROGRESS_MIN_INTERVAL_S = 0.1
PROGRESS_MIN_FRACTION = 0.005


class Protocol:
    """Owns the real stdout; everything else in the process writes to stderr."""

    def __init__(self):
        self._out = os.fdopen(os.dup(1), "w", buffering=1)
        sys.stdout = sys.stderr
        self._lock = threading.Lock()

    def send(self, **msg) -> None:
        with self._lock:
            self._out.write(json.dumps(msg) + "\n")
            self._out.flush()


def reader(q: "queue.Queue[dict]", on_cancel=None) -> None:
    """Cancels are handled here, at once: the main loop is busy inside the job they cancel."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            q.put({"type": "bad_json", "raw": line[:200]})
            continue
        if msg.get("type") == "cancel" and on_cancel is not None:
            on_cancel(msg.get("id"))
            continue
        q.put(msg)
    q.put({"type": "eof"})


class Worker:
    def __init__(self, weights: Path, profile: str, proto: Protocol):
        self.weights = weights
        self.profile = profile
        self.proto = proto
        self.pipe = None
        self.inpaint = None
        self.profile_id = "klein-4b-bf16"
        self.cancel_ids: set[str] = set()
        self.cancel_lock = threading.Lock()
        # Text encodings by instruction. The same tensors the pipeline would compute, so the
        # output is identical; a retry or another seed skips the 8 GB text encoder entirely.
        self.embeds: "dict[str, object]" = {}
        self.embeds_max = 16

    # ---- lifecycle -------------------------------------------------------------------

    def load(self) -> None:
        import torch
        from diffusers import Flux2KleinInpaintPipeline, Flux2KleinPipeline

        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        from lis.profiles import apply_profile, choose, shorten_text_encoder

        free_gb = torch.cuda.mem_get_info()[0] / 2**30
        cap = os.environ.get("LIS_VRAM_CAP_GB")
        if cap:  # a test limit: the driver's free memory is not what this process may use
            free_gb = min(free_gb, float(cap))
        if self.profile.startswith("auto"):  # the fastest lossless profile that fits what is free now
            self.profile = choose(free_gb, self.profile.partition("/")[2] or "group")
        pipe = Flux2KleinPipeline.from_pretrained(self.weights, dtype=torch.bfloat16, local_files_only=True)
        dropped = shorten_text_encoder(pipe)  # before offloading: the unused layers never take memory
        apply_profile(pipe, self.profile)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.inpaint = Flux2KleinInpaintPipeline(**pipe.components)
        self.inpaint.set_progress_bar_config(disable=True)
        load_ms = int((time.time() - t0) * 1000)
        # One tiny render so the first real one does not pay for CUDA start-up and the
        # offload hooks' first pass (about 6 s measured on the first job otherwise).
        t1 = time.time()
        with torch.no_grad():
            emb = self.encode("a photo")
            self.pipe(prompt=None, prompt_embeds=emb, height=256, width=256, num_inference_steps=1, guidance_scale=1.0,
                      generator=torch.Generator("cuda").manual_seed(0))
        self._free()
        self.proto.send(type="ready", profile_id=self.profile_id, load_ms=load_ms, warmup_ms=int((time.time() - t1) * 1000),
                        vram_bytes=int(torch.cuda.memory_allocated()), memory_profile=self.profile, free_gb_at_load=round(free_gb, 1),
                        text_layers_dropped=dropped, is_distilled=bool(pipe.config.get("is_distilled", False)))

    def encode(self, prompt: str):
        """Prompt embeddings, bit-identical to the pipeline's encode_prompt."""
        from lis.profiles import encode_short

        return encode_short(self.pipe, prompt)

    def prepare(self, text: str) -> None:
        """Encode a sentence ahead of its job (sent when the user presses Find)."""
        import torch

        if not text or text in self.embeds:
            return
        try:
            with torch.no_grad():
                self.embeds[text] = self.encode(text).detach()
            while len(self.embeds) > self.embeds_max:
                self.embeds.pop(next(iter(self.embeds)))
        except Exception:  # noqa: BLE001  (the job encodes it again)
            traceback.print_exc()
        finally:
            self._free()

    def cancel(self, job_id: str) -> None:
        with self.cancel_lock:
            self.cancel_ids.add(job_id)

    def _cancelled(self, job_id: str) -> bool:
        with self.cancel_lock:
            return job_id in self.cancel_ids

    # ---- jobs ------------------------------------------------------------------------

    def run_job(self, job: dict) -> None:
        import torch
        from PIL import Image

        job_id = job["id"]
        try:
            w, h = int(job["width"]), int(job["height"])
            if w % 16 or h % 16 or w * h > MAX_AREA:
                raise ValueError(f"working region {w}x{h} must be multiples of 16 with area <= 1024x1024")
            strategy = job["strategy"]
            img = None
            if strategy != "generate":
                img = Image.open(job["image_path"]).convert("RGB")
                if img.size != (w, h):
                    raise ValueError(f"image is {img.size}, job says {(w, h)}")
            free, _total = torch.cuda.mem_get_info()
            if free < MIN_FREE_BYTES:
                raise MemoryError(f"only {free / 2**20:.0f} MB free on the GPU before the job; need {MIN_FREE_BYTES / 2**20:.0f}")
            steps = int(job.get("steps", 4))
            seed = int(job.get("seed", 0))
            gen = torch.Generator("cuda").manual_seed(seed)
            torch.cuda.reset_peak_memory_stats()

            state = {"last_emit": 0.0, "last_frac": -1.0, "first_step_at": None, "last_step_at": None}
            t_start = time.time()

            def on_step(pipeline, i, t, kwargs):
                now = time.time()
                if state["first_step_at"] is None:
                    state["first_step_at"] = now
                state["last_step_at"] = now
                if self._cancelled(job_id):
                    pipeline._interrupt = True
                frac = (i + 1) / steps
                if (now - state["last_emit"] >= PROGRESS_MIN_INTERVAL_S and frac - state["last_frac"] >= PROGRESS_MIN_FRACTION) or i + 1 == steps:
                    self.proto.send(type="progress", id=job_id, step=i + 1, of=steps)
                    state["last_emit"], state["last_frac"] = now, frac
                return kwargs

            emb = self.embeds.pop(job["instruction"], None)
            if emb is None:
                with torch.no_grad():
                    emb = self.encode(job["instruction"])
                emb = emb.detach()
            self.embeds[job["instruction"]] = emb  # most recent last
            while len(self.embeds) > self.embeds_max:
                self.embeds.pop(next(iter(self.embeds)))
            common = dict(prompt=None, prompt_embeds=emb, height=h, width=w, num_inference_steps=steps, guidance_scale=1.0,
                          generator=gen, callback_on_step_end=on_step)
            if strategy == "generate":  # a new picture from words, with reference photos if given
                refs = [Image.open(r).convert("RGB") for r in (job.get("ref_paths") or [])]
                result = self.pipe(image=refs, **common) if refs else self.pipe(**common)
            elif strategy == "edit":
                result = self.pipe(image=[img], **common)
            elif strategy == "inpaint":
                mask = Image.open(job["mask_path"]).convert("L")
                if mask.size != (w, h):
                    raise ValueError(f"mask is {mask.size}, job says {(w, h)}")
                result = self.inpaint(image=img, mask_image=mask, strength=float(job.get("strength", 0.8)), **common)
            else:
                raise ValueError(f"unknown strategy {strategy!r}")
            torch.cuda.synchronize()
            t_end = time.time()

            if self._cancelled(job_id):
                self.proto.send(type="failed", id=job_id, cause="cancelled", detail="cancelled by the user")
                return
            out = result.images[0]
            out.save(job["out_path"], format="PNG")
            fs, ls = state["first_step_at"] or t_start, state["last_step_at"] or t_end
            self.proto.send(type="done", id=job_id, out_path=job["out_path"],
                            timing_ms={"encode": int((fs - t_start) * 1000), "denoise": int((ls - fs) * 1000), "decode": int((t_end - ls) * 1000),
                                       "total": int((t_end - t_start) * 1000)},
                            peak_vram_bytes=int(torch.cuda.max_memory_allocated()), strategy=strategy, out_size=list(out.size))
        except torch.OutOfMemoryError as e:  # type: ignore[attr-defined]
            print(torch.cuda.memory_summary(), file=sys.stderr)
            self._free()
            self.proto.send(type="failed", id=job_id, cause="out_of_memory", detail=str(e)[:300])
        except MemoryError as e:
            self.proto.send(type="failed", id=job_id, cause="out_of_memory", detail=str(e))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.proto.send(type="failed", id=job_id, cause="error", detail=f"{type(e).__name__}: {e}"[:300])
        finally:
            with self.cancel_lock:
                self.cancel_ids.discard(job_id)
            self._free()

    @staticmethod
    def _free() -> None:
        import torch

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--offload", action="store_true", help="alias for --profile offload")
    ap.add_argument("--profile", default=None, help="auto | resident | fast | group | offload | offload_fp8 | group_fp8")
    args = ap.parse_args()
    cap = os.environ.get("LIS_VRAM_CAP_GB")  # tests: behave like a smaller card (allocator limit; the CUDA context is extra)
    if cap:
        import torch

        torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    proto = Protocol()
    worker = Worker(Path(args.weights), args.profile or ("offload" if args.offload else "auto"), proto)
    q: "queue.Queue[dict]" = queue.Queue()
    threading.Thread(target=reader, args=(q, worker.cancel), daemon=True).start()

    while True:
        msg = q.get()
        kind = msg.get("type")
        if kind in ("eof", "quit"):
            return
        if kind == "load":
            try:
                worker.load()
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                proto.send(type="failed", id=None, cause="error", detail=f"load: {type(e).__name__}: {e}"[:300])
        elif kind == "cancel":
            worker.cancel(msg.get("id"))
        elif kind == "prepare":
            if worker.pipe is not None:
                worker.prepare(str(msg.get("instruction") or ""))
        elif kind == "job":
            if worker.pipe is None:
                proto.send(type="failed", id=msg.get("id"), cause="error", detail="model not loaded")
                continue
            # Drain cancels that arrived while this job was queued, then run it.
            worker.run_job(msg)
        elif kind == "bad_json":
            proto.send(type="failed", id=None, cause="error", detail=f"bad json: {msg.get('raw')}")


if __name__ == "__main__":
    main()
