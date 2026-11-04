"""Gradio UI, the fallback editor (the main one is lis/web.py). One screen.

Handlers are plain functions over a small per-session state dict so they can be tested
without a browser. The UI never calls a model; it only speaks to lis.engine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from lis.engine import Engine, EngineBusy, EngineConfig
from lis.ground import parse_instruction
from lis.render import WorkerConfig

ROOT = Path(__file__).resolve().parents[1]
PROXY_LONG_SIDE = 1200
TARGET_RGB = (255, 140, 0)
PROTECT_RGB = (0, 200, 255)
CANDIDATE_RGB = (255, 255, 0)

_cancel_events: dict[str, threading.Event] = {}


def radio(choices: list[str], value: str | None = None):
    """A Radio update; plain lists are not valid Radio values."""
    return gr.update(choices=choices, value=value)


# ---- state and drawing ------------------------------------------------------------------


def new_state() -> dict:
    return {"project_id": None, "head": None, "mode": "target", "tool": "positive", "object_id": None, "clicks": [],
            "box_start": None, "box": None, "candidates": [], "picked": 0, "objects": [], "scale": 1.0, "last_result": None,
            "proposal": None, "parsed_from": None}


def proxy_scale(w: int, h: int) -> float:
    return min(1.0, PROXY_LONG_SIDE / max(w, h))


def to_proxy(img: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return img
    pil = Image.fromarray(img)
    return np.asarray(pil.resize((max(1, round(img.shape[1] * scale)), max(1, round(img.shape[0] * scale))), Image.BILINEAR))


def mask_to_proxy(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)) >= 128


def draw_overlay(base_proxy: np.ndarray, layers: list[tuple[np.ndarray, tuple[int, int, int], float]], points: list | None = None) -> np.ndarray:
    out = base_proxy.astype(np.float32)
    for mask, rgb, alpha in layers:
        m = mask_to_proxy(mask, out.shape[:2])
        out[m] = out[m] * (1 - alpha) + np.array(rgb, np.float32) * alpha
        # a 1 px outline
        edge = m & ~np.pad(m, 1)[1:-1, 1:-1].__and__(np.pad(m, 1)[2:, 1:-1]).__and__(np.pad(m, 1)[:-2, 1:-1]).__and__(np.pad(m, 1)[1:-1, 2:]).__and__(np.pad(m, 1)[1:-1, :-2])
        out[edge] = rgb
    out = out.clip(0, 255).astype(np.uint8)
    if points:
        pil = Image.fromarray(out)
        from PIL import ImageDraw

        d = ImageDraw.Draw(pil)
        for x, y, label in points:
            r = 6
            d.ellipse([x - r, y - r, x + r, y + r], fill=(0, 220, 0) if label == 1 else (220, 0, 0), outline=(255, 255, 255), width=2)
        out = np.asarray(pil)
    return out


class UI:
    """Binds an Engine to handler functions. One instance serves all sessions; per-session data lives in the state dict."""

    def __init__(self, engine: Engine):
        self.engine = engine

    # ---- helpers -----------------------------------------------------------------------

    def _base(self, st: dict) -> np.ndarray:
        p = self.engine._project(st["project_id"])
        return p.base(p.record.head)

    def _objects_layers(self, st: dict, p) -> list:
        layers = []
        for o in st["objects"]:
            layers.append((p.mask(o["mask_id"]), TARGET_RGB if o["kind"] == "target" else PROTECT_RGB, 0.30))
        return layers

    def canvas(self, st: dict, overlays: bool = True) -> np.ndarray | None:
        """The image on screen. After a render it is shown clean; selection tints return with
        the next selection action."""
        if not st["project_id"]:
            return None
        p = self.engine._project(st["project_id"])
        base = to_proxy(self._base(st), st["scale"])
        if not overlays:
            return base
        layers = self._objects_layers(st, p)
        if st.get("proposal"):
            for kind, m in st["proposal"]["masks"]:
                layers.append((m, TARGET_RGB if kind == "target" else PROTECT_RGB, 0.50))
        if st["candidates"]:
            layers.append((st["candidates"][st["picked"]]["mask"], CANDIDATE_RGB, 0.45))
        pts = [(round(x * st["scale"]), round(y * st["scale"]), l) for x, y, l in st["clicks"]]
        return draw_overlay(base, layers, pts)

    def object_choices(self, st: dict) -> list[str]:
        return [f"{o['name']} ({o['kind']})" for o in st["objects"]]

    def history_choices(self, st: dict) -> tuple[list[str], str | None]:
        if not st["project_id"]:
            return [], None
        h = self.engine.history(st["project_id"])
        rows = ["source"] + [f"{r['revision_id']}  {r['instruction'][:40]}  [{r['protected_check']['status']}]" for r in h["revisions"]]
        head = "source" if h["head"] is None else next(r for r in rows[1:] if r.startswith(h["head"]))
        return rows, head

    def history_update(self, st: dict):
        rows, head = self.history_choices(st)
        return radio(rows, head)

    # ---- handlers ----------------------------------------------------------------------

    def open(self, file_path: str | None, st: dict):
        st = new_state()
        if not file_path:
            return (st, None, "Drop a photo to start.", radio([]), None, radio([]), radio([]), *self._cleared())
        o = self.engine.open(file_path)
        st.update(project_id=o["project_id"], head=o["head"], scale=proxy_scale(o["width"], o["height"]))
        st["object_id"] = f"obj-{int(time.time() * 1000)}"
        return (st, self.canvas(st), f"Opened {o['width']}x{o['height']}. **Step 2:** write what to change and press Find what to edit, or select by hand.",
                radio([]), None, self.history_update(st), radio([]), *self._cleared())

    @staticmethod
    def _cleared():
        """A new photo starts clean: no old result, change map, proposal or fields."""
        return "", None, "", gr.update(value=""), gr.update(value="")

    def set_mode(self, mode: str, st: dict):
        st["mode"] = mode
        return st, f"Mode: {'select target' if mode == 'target' else 'mark protected'}. Click the object."

    def set_tool(self, tool: str, st: dict):
        st["tool"] = {"positive click": "positive", "negative click": "negative", "box (two clicks)": "box"}[tool]
        st["box_start"] = None
        return st

    def click(self, st: dict, evt: gr.SelectData):
        if not st["project_id"]:
            return st, None, "Open a photo first.", radio([])
        x, y = int(round(evt.index[0] / st["scale"])), int(round(evt.index[1] / st["scale"]))
        if st["tool"] == "box":
            if st["box_start"] is None:
                st["box_start"] = (x, y)
                return st, self.canvas(st), "Box started. Click the opposite corner.", radio([])
            x0, y0 = st["box_start"]
            st["box"] = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
            st["box_start"] = None
        else:
            st["clicks"].append((x, y, 1 if st["tool"] == "positive" else 0))
        try:
            sel = self.engine.select(st["project_id"], st["object_id"], st["clicks"], tuple(st["box"]) if st["box"] else None)
        except EngineBusy as e:
            return st, self.canvas(st), str(e), radio([])
        cands = []
        for c in sel["candidates"]:
            m = np.asarray(Image.open(c["png_path"]).convert("L")) >= 128
            cands.append({"candidate_id": c["candidate_id"], "score": c["score"], "mask": m})
        st["candidates"], st["picked"] = cands, 0
        if not cands:
            return st, self.canvas(st), "Nothing selected here, try another click or a box.", radio([])
        choices = self.candidate_choices(st)
        return st, self.canvas(st), f"{len(cands)} candidates. Keep clicking to refine, or keep the highlighted one.", radio(choices, choices[0])

    def candidate_choices(self, st: dict) -> list[str]:
        return [f"{i + 1}. {c.get('label') or 'candidate'} (score {c['score']:.2f})" for i, c in enumerate(st["candidates"])]

    def find(self, text: str, st: dict):
        """Select by typing what it is: 'the red car', 'faces'. Candidates appear like click candidates."""
        if not st["project_id"]:
            return st, None, "Open a photo first.", radio([])
        if not (text or "").strip():
            return st, self.canvas(st), "Type what to select, for example 'the red car' or 'faces'.", radio([])
        st.update(clicks=[], box=None, box_start=None, object_id=f"obj-{int(time.time() * 1000)}")
        try:
            r = self.engine.ground(st["project_id"], st["object_id"], text)
        except (EngineBusy, RuntimeError) as e:
            return st, self.canvas(st), str(e), radio([])
        cands = []
        for c in r["candidates"]:
            m = np.asarray(Image.open(c["png_path"]).convert("L")) >= 128
            cands.append({"candidate_id": c["candidate_id"], "score": c["score"], "mask": m, "label": c["label"], "box": c.get("box")})
        st["candidates"], st["picked"] = cands, 0
        if not cands:
            return st, self.canvas(st), f"Nothing matching {text!r} was found. Try other words, or click the object.", radio([])
        if cands[0].get("box"):
            st["box"] = cands[0]["box"]  # clicks now refine this candidate
        choices = self.candidate_choices(st)
        weak = "" if r["candidates"][0].get("confident", True) else " The best match is weak: check it."
        return st, self.canvas(st), f"Found {len(cands)} candidate(s) for {r['phrase']!r}.{weak} Pick one, click to refine, then Keep.", radio(choices, choices[0])

    def suggest(self, instruction: str, target_text: str, protect_text: str, st: dict):
        """Read the instruction and propose what to edit and what to keep. Shown, not kept: the
        user presses 'Use this selection' or fixes it first."""
        if not st["project_id"]:
            return st, None, "Open a photo first.", gr.update(), gr.update(), ""
        if not (instruction or "").strip():
            return st, self.canvas(st), "Write the instruction first.", gr.update(), gr.update(), ""
        edited = st.get("parsed_from") == instruction  # same instruction: the fields carry the user's corrections
        try:
            r = self.engine.propose(st["project_id"], instruction, target_text if edited else None, protect_text if edited else None)
        except (EngineBusy, RuntimeError) as e:
            return st, self.canvas(st), str(e), gr.update(), gr.update(), ""
        st["parsed_from"] = instruction
        masks, lines = [], []
        t = r["target"]
        if t.get("candidate_id"):
            masks.append(("target", np.asarray(Image.open(t["png_path"]).convert("L")) >= 128))
            what = "everything except the kept objects" if t["background"] else ", ".join(t["phrases"])
            lines.append(f"**edit:** {what}" + (" (weak match, check it)" if t["weak"] else ""))
        else:
            lines.append("**edit:** nothing found. Type it in the Target field or click the object.")
        for e in r["protect"]:
            if e.get("candidate_id"):
                masks.append(("protected", np.asarray(Image.open(e["png_path"]).convert("L")) >= 128))
                lines.append(f"**keep:** {e['phrase']} ({e['used']} found)" + (" (weak match, check it)" if e["weak"] else ""))
            else:
                lines.append(f"**keep:** {e['phrase']}: not found, click it instead")
        for n in r["notes"]:
            lines.append(f"note: {n}")
        if r["edit"] != instruction.strip():
            lines.append(f"sent to the model: \"{r['edit']}\"")
        st["proposal"] = {"result": r, "masks": masks}
        tfield = "background" if t["background"] else ", ".join(t["phrases"])
        pfield = ", ".join(e["phrase"] for e in r["protect"])
        status = "Proposed selection shown: orange is edited, blue is kept. Press 'Use this selection', or correct the fields and press 'Select from instruction' again."
        return st, self.canvas(st), status, gr.update(value=tfield), gr.update(value=pfield), "  \n".join(lines)

    def use_proposal(self, st: dict):
        prop = st.get("proposal")
        if not prop:
            return st, self.canvas(st), "Press 'Select from instruction' first.", radio(self.object_choices(st))
        r = prop["result"]
        kept = []
        for e in r["protect"]:
            if e.get("candidate_id"):
                m = self.engine.commit_mask(st["project_id"], f"prop-{e['candidate_id']}", e["candidate_id"], "protected", e["phrase"])
                st["objects"].append({"mask_id": m["mask_id"], "kind": "protected", "name": e["phrase"]})
                kept.append(f"{e['phrase']} (keep)")
        t = r["target"]
        if t.get("candidate_id"):
            name = "background" if t["background"] else " and ".join(t["phrases"])
            m = self.engine.commit_mask(st["project_id"], f"prop-{t['candidate_id']}", t["candidate_id"], "target", name)
            st["objects"].append({"mask_id": m["mask_id"], "kind": "target", "name": name})
            kept.append(f"{name} (edit)")
        st["proposal"] = None
        if not kept:
            return st, self.canvas(st), "Nothing to use: the proposal found no objects.", radio(self.object_choices(st))
        return st, self.canvas(st), "Kept " + ", ".join(kept) + ". Render when ready.", radio(self.object_choices(st), self.object_choices(st)[-1])

    def pick(self, choice: str | None, st: dict):
        if choice and st["candidates"]:
            st["picked"] = int(choice.split(".")[0]) - 1
            box = st["candidates"][st["picked"]].get("box")
            if box and not st["clicks"]:
                st["box"] = box  # refine clicks start from the picked object's box
        return st, self.canvas(st)

    def keep_inverted(self, name: str, st: dict):
        return self.keep(name, st, invert=True)

    def keep(self, name: str, st: dict, invert: bool = False):
        if not st["candidates"]:
            return st, self.canvas(st), "Select something first.", radio(self.object_choices(st)), radio([])
        c = st["candidates"][st["picked"]]
        kind = st["mode"]
        name = (name or "").strip() or (f"everything except {kind}-{len(st['objects']) + 1}" if invert else f"{kind}-{len(st['objects']) + 1}")
        r = self.engine.commit_mask(st["project_id"], st["object_id"], c["candidate_id"], kind, name, invert=invert)
        st["objects"].append({"mask_id": r["mask_id"], "kind": kind, "name": name})
        st.update(clicks=[], box=None, box_start=None, candidates=[], picked=0, object_id=f"obj-{int(time.time() * 1000)}")
        return st, self.canvas(st), f"Kept {name!r} as {kind}" + (" (inverted: everything except the selection)." if invert else "."), radio(self.object_choices(st)), radio([])

    def clear_selection(self, st: dict):
        st.update(clicks=[], box=None, box_start=None, candidates=[], picked=0, object_id=f"obj-{int(time.time() * 1000)}")
        return st, self.canvas(st), "Selection cleared.", radio([])

    # brush: the editor's single layer alpha becomes a delta
    def brush_open(self, st: dict):
        if not st["project_id"]:
            return st, None
        return st, {"background": self.canvas(st), "layers": [], "composite": None}

    def brush_apply(self, editor_value, object_choice: str | None, op: str, st: dict):
        if not st["project_id"] or not object_choice or editor_value is None or not editor_value.get("layers"):
            return st, self.canvas(st), "Paint on the canvas, choose an object, then apply."
        idx = self.object_choices(st).index(object_choice)
        obj = st["objects"][idx]
        layer = editor_value["layers"][0]
        alpha = (np.asarray(layer)[..., 3] > 0) if np.asarray(layer).ndim == 3 and np.asarray(layer).shape[2] == 4 else (np.asarray(layer).sum(axis=-1) > 0)
        p = self.engine._project(st["project_id"])
        full = np.asarray(Image.fromarray(alpha.astype(np.uint8) * 255).resize((p.record.width, p.record.height), Image.NEAREST)) >= 128
        delta_path = p.root / "masks" / f"brush-{int(time.time() * 1000)}.png"
        Image.fromarray(full.astype(np.uint8) * 255).save(delta_path)
        if op == "add":
            self.engine.update_mask(st["project_id"], obj["mask_id"], brush_add_png=str(delta_path))
        else:
            self.engine.update_mask(st["project_id"], obj["mask_id"], brush_subtract_png=str(delta_path))
        return st, self.canvas(st), f"Brush {op} applied to {obj['name']!r} ({int(full.sum())} px)."

    def render(self, instruction: str, seed: int, steps: int, band: int, strategy: str, strength: float, target_choice: str | None, st: dict, blend: str = "auto", progress=None):
        """Generator: yields (state, slider, result_md, status, history_update, canvas, change_map)."""
        for out in self._render6(instruction, seed, steps, band, strategy, strength, target_choice, st, blend, progress):
            yield (*out, self._change_map_update(st, out))

    def _change_map_update(self, st: dict, out: tuple):
        last = st.get("last_result") or {}
        if isinstance(out[1], tuple) and last.get("change_map_path"):  # a finished, accepted render
            return to_proxy(np.asarray(Image.open(last["change_map_path"]).convert("RGB")), st["scale"])
        return gr.update()

    def another_seed(self, instruction, seed, steps, band, strategy, strength, target_choice, st, blend="auto", progress=None):
        """Same plan, next seed, from the same starting image as the last result."""
        last = st.get("last_result")
        if not last or last.get("status") != "ok":
            yield st, gr.update(), gr.update(), "Render once first.", self.history_update(st), gr.update(), gr.update(), gr.update()
            return
        h = self.engine.history(st["project_id"])
        rev = next(r for r in h["revisions"] if r["revision_id"] == last["revision_id"])
        self.engine.set_head(st["project_id"], rev["parent"])
        new_seed = int(seed) + 1
        for out in self.render(instruction, new_seed, steps, band, strategy, strength, target_choice, st, blend, progress):
            yield (*out, gr.update(value=new_seed))

    def _render6(self, instruction: str, seed: int, steps: int, band: int, strategy: str, strength: float, target_choice: str | None, st: dict, blend: str = "auto", progress=None):
        if not st["project_id"]:
            yield st, None, "", "Open a photo first.", radio([]), gr.update()
            return
        targets = [o for o in st["objects"] if o["kind"] == "target"]
        if not targets:
            msg = ("A selection is proposed but not kept yet: press 'Use this selection' or fix it first." if st.get("proposal")
                   else "Keep a target object first: click it, find it by text, or press 'Select from instruction'.")
            yield st, None, "", msg, self.history_update(st), gr.update()
            return
        sent = parse_instruction(instruction or "").edit  # keep clauses are enforced by masks, not sent to the model
        target = targets[-1]
        if target_choice and target_choice in self.object_choices(st):
            cand = st["objects"][self.object_choices(st).index(target_choice)]
            if cand["kind"] == "target":
                target = cand
        p = self.engine._project(st["project_id"])
        plan = {"schema_version": 2, "project_id": st["project_id"], "parent_revision": p.record.head,
                "operation": {"kind": "native_instruction_edit", "target_mask_id": target["mask_id"], "instruction": sent, "band_px": int(band), "strategy": strategy, "blend": blend},
                "protected_mask_ids": [o["mask_id"] for o in st["objects"] if o["kind"] == "protected"], "seed": int(seed), "steps": int(steps),
                "strength": float(strength), "backend_profile_id": self.engine.cfg.profile_id, "network_policy": "offline"}
        ev = _cancel_events.setdefault(st["project_id"], threading.Event())
        ev.clear()
        before = to_proxy(self._base(st), st["scale"])
        result = None
        try:
            for msg in self.engine.render(plan, ev):
                if msg["type"] == "progress":
                    if progress is not None:
                        progress(msg["step"] / max(1, msg["of"]), desc=f"step {msg['step']}/{msg['of']}" + (" (retry)" if msg.get("attempt") else ""))
                    yield st, gr.update(), gr.update(), f"Rendering step {msg['step']}/{msg['of']}" + (f" ({msg['note']})" if msg.get("note") else ""), gr.update(), gr.update()
                else:
                    result = msg
        except EngineBusy as e:
            yield st, gr.update(), gr.update(), str(e), gr.update(), gr.update()
            return
        st["last_result"] = result
        if result["status"] != "ok":
            yield st, gr.update(), f"**{result['status']}**  \n{result['message']}", result["message"], self.history_update(st), gr.update()
            return
        after = to_proxy(np.asarray(Image.open(result["image_path"]).convert("RGB")), st["scale"])
        wr = result["working_region"]
        pc = result["protected_check"]
        oc = result.get("outside_check") or {}
        ec = result.get("edit_check") or {}
        ok = pc["status"] in ("pass", "not_applicable") and oc.get("status") in ("pass", "not_applicable", None)
        prot_line = ("**Protected areas:** none marked" if pc["status"] == "not_applicable" else
                     f"**Protected areas: unchanged** ({pc['mismatched_pixels']} px differ)")
        out_line = (f"**Outside the edit area: unchanged** ({oc.get('mismatched_pixels', 0)} px differ; "
                    f"{oc.get('changed_fraction', 0) * 100:.1f}% of the photo changed, all inside the edit area)" if oc.get("status") == "pass" else "")
        edit_word = {"verified": "verified", "not_detected": "NOT DETECTED", "changed": "changed", "unchanged": "unchanged", "not_checked": "not checked"}.get(ec.get("status"), "not checked")
        md = (f"**check: {'PASS' if ok else 'FAIL'}** (on the saved file)  \n{prot_line}  \n" + (f"{out_line}  \n" if out_line else "") +
              f"**Edit:** {edit_word}" + (f": {ec['message']}" if ec.get("message") else "") + "  \n"
              f"edited at {wr['scale'] * 100:.0f}% ({wr['worker_size'][0]}x{wr['worker_size'][1]}), {result['timing_ms'].get('total', 0) / 1000:.1f} s, "
              f"edge {result.get('blend_used')}")
        if ec.get("status") in ("not_detected", "unchanged"):
            md += "  \nPress **Try another seed**, or reword the instruction."
        if sent != (instruction or "").strip():
            md += f"  \nsent to the model: \"{sent}\""
        if result.get("expansion_hint_mask_path"):
            md += "  \nThe model wanted to change pixels just outside the selection. Grow the mask with the brush if that is intended."
        status = f"Rendered {result['revision_id']}. " + (result["message"] or "")
        yield st, (before, after), md, status, self.history_update(st), self.canvas(st, overlays=False)  # the new head, clean

    def grow_and_rerender(self, instruction, seed, steps, band, strategy, strength, target_choice, st, blend="auto", progress=None):
        """Add the last result's expansion hint to its target, step back to that result's parent and
        render again with the same settings."""
        last = st.get("last_result")
        if not last or last.get("status") != "ok" or not last.get("expansion_hint_mask_path"):
            yield st, gr.update(), gr.update(), "There is no expansion hint on the last result.", self.history_update(st), gr.update(), gr.update()
            return
        h = self.engine.history(st["project_id"])
        rev = next(r for r in h["revisions"] if r["revision_id"] == last["revision_id"])
        p = self.engine._project(st["project_id"])
        meta = json.loads((p.root / "runs" / last["revision_id"] / "meta.json").read_text())
        target_id = meta["operation"]["target_mask_id"]
        self.engine.update_mask(st["project_id"], target_id, brush_add_png=last["expansion_hint_mask_path"])
        self.engine.set_head(st["project_id"], rev["parent"])
        yield from self.render(instruction, seed, steps, band, strategy, strength, target_choice, st, blend, progress)

    def cancel(self, st: dict):
        if st["project_id"] and st["project_id"] in _cancel_events:
            _cancel_events[st["project_id"]].set()
        return "Cancelling after the current step."

    def set_head(self, choice: str | None, st: dict):
        if not st["project_id"] or not choice:
            return st, self.canvas(st), "", self.history_update(st)
        rev = None if choice == "source" else choice.split()[0]
        self.engine.set_head(st["project_id"], rev)
        return st, self.canvas(st), f"Showing {choice.split()[0]}.", self.history_update(st)

    def undo(self, st: dict):
        if not st["project_id"]:
            return st, None, "", radio([])
        h = self.engine.history(st["project_id"])
        if h["head"] is None:
            return st, self.canvas(st), "Nothing to undo.", self.history_update(st)
        parent = next(r["parent"] for r in h["revisions"] if r["revision_id"] == h["head"])
        self.engine.set_head(st["project_id"], parent)
        return st, self.canvas(st), "Undone.", self.history_update(st)

    def export(self, fmt: str, st: dict):
        if not st["project_id"]:
            return None, "Open a photo first."
        p = self.engine._project(st["project_id"])
        rev = p.record.head
        out_dir = p.root / "exports"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"{rev or 'source'}.{ 'jpg' if fmt == 'jpeg' else 'png'}"
        r = self.engine.export(st["project_id"], rev, path, fmt)
        note = " JPEG is lossy: the byte-identical guarantee does not survive it." if fmt == "jpeg" else ""
        return str(path), f"Exported {path.name} ({r['bytes']} bytes, check {r['protected_check']['status']}).{note}"


# ---- layout ---------------------------------------------------------------------------------


def build(ui: UI):
    with gr.Blocks(title="OpenImage Studio", analytics_enabled=False) as demo:
        st = gr.State(new_state())
        gr.Markdown("## OpenImage Studio  \nEdits photos on this computer only. What you mark to keep stays exactly the same, and every result proves it.")
        status = gr.Markdown("**Start:** drop a photo in step 1.")
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 1. Photo")
                photo = gr.File(label="Photo", file_types=["image"], type="filepath")
                gr.Markdown("### 2. Say what to change")
                instruction = gr.Textbox(label="Instruction", lines=3,
                                         placeholder="make the red car blue, keep the people unchanged")
                suggest_btn = gr.Button("Find what to edit", variant="secondary")
                with gr.Row():
                    target_field = gr.Textbox(label="Edit (found in the instruction)", placeholder="red car", scale=1)
                    protect_field = gr.Textbox(label="Keep unchanged (comma separated)", placeholder="people", scale=1)
                proposal_md = gr.Markdown("")
                use_btn = gr.Button("Use this selection", variant="primary")
                objects = gr.Radio([], label="Selected (orange is edited, blue is kept)")
                with gr.Accordion("Or select by hand: click, box, type, brush", open=False):
                    mode = gr.Radio(["target", "protected"], value="target", label="Next selection is")
                    tool = gr.Radio(["positive click", "negative click", "box (two clicks)"], value="positive click", label="Tool")
                    with gr.Row():
                        find_text = gr.Textbox(label="Find by text", placeholder="the red car, faces, the dog", scale=3)
                        find_btn = gr.Button("Find", scale=1)
                    cand_pick = gr.Radio([], label="Candidates")
                    obj_name = gr.Textbox(label="Name", placeholder="jacket, face, ...")
                    with gr.Row():
                        keep_btn = gr.Button("Keep", variant="primary")
                        keep_inv_btn = gr.Button("Keep everything else")
                        clear_btn = gr.Button("Clear selection")
                    brush_open_btn = gr.Button("Open brush canvas")
                    editor = gr.ImageEditor(label="Paint, then apply", sources=[], transforms=[], layers=False, type="numpy", image_mode="RGBA",
                                            brush=gr.Brush(colors=["#ff8c00"], color_mode="fixed", default_size=25), eraser=gr.Eraser(default_size=25))
                    with gr.Row():
                        brush_add = gr.Button("Apply as add")
                        brush_sub = gr.Button("Apply as subtract")
            with gr.Column(scale=5):
                canvas = gr.Image(label="Photo (click to select by hand)", type="numpy", interactive=False)
                slider = gr.ImageSlider(label="Before / after", type="numpy")
                change_map = gr.Image(label="What changed: orange changed, blue kept, white line is the edit area", type="numpy", interactive=False)
            with gr.Column(scale=3):
                gr.Markdown("### 3. Render")
                render_btn = gr.Button("Render", variant="primary")
                with gr.Row():
                    another_btn = gr.Button("Try another seed")
                    cancel_btn = gr.Button("Cancel")
                grow_btn = gr.Button("Grow to the hint and re-render")
                result_md = gr.Markdown("")
                gr.Markdown("### History")
                history = gr.Radio([], label="Versions (select to show)")
                undo_btn = gr.Button("Undo")
                gr.Markdown("### Export")
                export_fmt = gr.Radio(["png", "jpeg"], value="png", label="Format (PNG keeps the guarantee)")
                export_btn = gr.Button("Export")
                export_file = gr.File(label="Exported file")
                with gr.Accordion("Advanced", open=False):
                    with gr.Row():
                        seed = gr.Number(value=1234, label="Seed", precision=0)
                        steps = gr.Number(value=4, label="Steps", precision=0)
                    with gr.Row():
                        band = gr.Number(value=8, label="Edge px", precision=0)
                        strength = gr.Number(value=0.8, label="Inpaint strength")
                    blend = gr.Radio(["auto", "tight", "context"], value="auto", label="Edge (auto: context for removals, additions, backgrounds)")
                    strategy = gr.Radio(["auto", "edit", "inpaint"], value="auto", label="Strategy (auto means edit)")

        photo.change(ui.open, [photo, st], [st, canvas, status, cand_pick, slider, history, objects, result_md, change_map, proposal_md, target_field, protect_field])
        mode.change(ui.set_mode, [mode, st], [st, status])
        tool.change(ui.set_tool, [tool, st], [st])
        canvas.select(ui.click, [st], [st, canvas, status, cand_pick])
        cand_pick.change(ui.pick, [cand_pick, st], [st, canvas])
        find_btn.click(ui.find, [find_text, st], [st, canvas, status, cand_pick])
        find_text.submit(ui.find, [find_text, st], [st, canvas, status, cand_pick])
        suggest_btn.click(ui.suggest, [instruction, target_field, protect_field, st], [st, canvas, status, target_field, protect_field, proposal_md])
        use_btn.click(ui.use_proposal, [st], [st, canvas, status, objects])
        keep_btn.click(ui.keep, [obj_name, st], [st, canvas, status, objects, cand_pick])
        keep_inv_btn.click(ui.keep_inverted, [obj_name, st], [st, canvas, status, objects, cand_pick])
        clear_btn.click(ui.clear_selection, [st], [st, canvas, status, cand_pick])
        brush_open_btn.click(ui.brush_open, [st], [st, editor])
        brush_add.click(lambda ev, o, s: ui.brush_apply(ev, o, "add", s), [editor, objects, st], [st, canvas, status])
        brush_sub.click(lambda ev, o, s: ui.brush_apply(ev, o, "subtract", s), [editor, objects, st], [st, canvas, status])
        render_inputs = [instruction, seed, steps, band, strategy, strength, objects, st, blend]
        render_outputs = [st, slider, result_md, status, history, canvas, change_map]
        render_btn.click(ui.render, render_inputs, render_outputs)
        another_btn.click(ui.another_seed, render_inputs, render_outputs + [seed])
        grow_btn.click(ui.grow_and_rerender, render_inputs, render_outputs)
        cancel_btn.click(ui.cancel, [st], [status])
        history.select(ui.set_head, [history, st], [st, canvas, status, history])
        undo_btn.click(ui.undo, [st], [st, canvas, status, history])
        export_btn.click(ui.export, [export_fmt, st], [export_file, status])
    return demo


GROUNDER_DIRS = {"dino": "gdino-base", "sam3": "sam3"}


def gpu_total_gb() -> float | None:
    try:
        import torch

        return torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


SMALL_GPU_GB = 12.0  # below this, the SAM models leave the GPU while the editing model renders


def make_engine(fake: bool, offload: bool, segmenter: str, projects: Path, profile: str | None = None, grounder: str = "auto",
                small_gpu: str = "auto") -> Engine:
    if fake:
        worker = WorkerConfig(module="lis.worker_fake", weights="", python=sys.executable)
        eng = Engine(EngineConfig(projects_dir=projects, manifest_path=ROOT / "assets" / "manifest.json", worker=worker, segmenter="fake", profile_id="fake",
                                  grounder="fake"))
        eng.manifest.setdefault("profiles", {})["fake"] = {"files": {}}
        return eng
    # expandable_segments: the CUDA cache grows in place instead of keeping stranded blocks; the
    # worker's real peak fell from 5.1 to 4.0 GB with byte-identical output.
    for k, v in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "GRADIO_ANALYTICS_ENABLED": "False",
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}.items():
        os.environ.setdefault(k, v)
    worker = WorkerConfig(module="lis.worker_klein", weights=str(ROOT / "assets" / "weights" / "flux2-klein-4b"), offload=offload, profile=profile, python=sys.executable)
    if grounder == "auto":  # SAM 3 when its weights are present, else Grounding DINO, else none
        grounder = next((g for g in ("sam3", "dino") if (ROOT / "assets" / "weights" / GROUNDER_DIRS[g] / "config.json").exists()), "none")
    total = gpu_total_gb()
    park = small_gpu == "on" or (small_gpu == "auto" and total is not None and total < SMALL_GPU_GB)
    return Engine(EngineConfig(projects_dir=projects, manifest_path=ROOT / "assets" / "manifest.json", worker=worker, segmenter=segmenter,
                               segmenter_weights=str(ROOT / "assets" / "weights" / segmenter), grounder=grounder,
                               grounder_weights=str(ROOT / "assets" / "weights" / GROUNDER_DIRS.get(grounder, "")), park_selection=park))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--offload", action="store_true", help="alias for --profile offload")
    ap.add_argument("--profile", default="auto", help="memory profile: auto (default: resident, fast or group, by free video memory) | group | group_fp8 | fast | resident | offload | offload_fp8")
    ap.add_argument("--segmenter", default="sam2", choices=["sam2", "sam3_tracker"])
    ap.add_argument("--grounder", default="auto", choices=["auto", "dino", "sam3", "none"], help="selection from words; auto picks sam3, then dino")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--projects", default=str(ROOT / "projects"))
    args = ap.parse_args(argv)
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("GRADIO_TEMP_DIR", str(Path(args.projects) / ".gradio"))
    Path(args.projects).mkdir(parents=True, exist_ok=True)
    engine = make_engine(args.fake, args.offload, args.segmenter, Path(args.projects), args.profile, args.grounder)
    print("starting render worker (this loads the model once)...", file=sys.stderr)
    ready = engine.client.start()
    print(f"worker ready: {json.dumps(ready)}", file=sys.stderr)
    demo = build(UI(engine))
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False, theme=gr.themes.Base())


if __name__ == "__main__":
    main()
