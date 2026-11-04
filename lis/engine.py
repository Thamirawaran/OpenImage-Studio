"""The only module the UI imports. Implements the edit plan contract (schema 2): open,
select, commit_mask, update_mask, render, history, set_head, export. Owns the working
region, the OOM single retry, worker lifecycle, and the file-level protected check that
re-reads out.png, the base and the masks. A failed check is returned as failed, never ok.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import threading
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from lis import composite as C
from lis.history import History, Revision
from lis.policy import REMOVAL_PATTERN, resolve
from lis.project import Project, decode_canonical, load_mask_png, new_id, save_mask_png, save_rgb_png, sha256_array, sha256_file
from lis.render import RenderClient, WorkerConfig, WorkerUnavailable
from lis.ground import FakeGrounder, Phrase, default_pick, find_phrase, make_phrase, parse_instruction, singular, _split_list
from lis.segment import Candidate, FakeSegmenter, ObjectState, SamSegmenter
from lis.shadow import find_shadow, shadow_factor
from lis.verify import verify_edit

MAX_AREA = 1024 * 1024
MIN_SIDE = 512
OOM_SHRINK = 0.75


@dataclass
class EngineConfig:
    projects_dir: Path
    manifest_path: Path
    worker: WorkerConfig
    segmenter: str = "fake"  # fake | sam2 | sam3_tracker
    segmenter_weights: str = ""
    profile_id: str = "klein-4b-bf16"
    color_match: bool = True
    grounder: str = "none"  # none | fake | dino | sam3
    grounder_weights: str = ""
    grounder_table: dict | None = None  # FakeGrounder boxes, tests only
    park_selection: bool = False  # small GPUs: SAM models wait in RAM while the editing model renders


class EngineBusy(RuntimeError):
    pass


# ---- geometry -------------------------------------------------------------------------


def working_region(shape: tuple[int, int], t_core: np.ndarray, feather: int, grow: int) -> tuple[C.WorkingRegion, tuple[int, int, int, int]]:
    """Returns (in-image region for the compositor, worker box
    (x, y, w, h) with w and h multiples of 16, which may extend past the image and is then
    padded by edge replication)."""
    H, W = shape
    ys, xs = np.nonzero(t_core)
    reach = grow + feather
    by0, by1 = max(0, int(ys.min()) - reach), min(H, int(ys.max()) + 1 + reach)
    bx0, bx1 = max(0, int(xs.min()) - reach), min(W, int(xs.max()) + 1 + reach)
    pad = max(max(H, W) / 16.0, ((by1 - by0) + (bx1 - bx0)) / 4.0)  # Krita: max(longest/16, avg side/2)
    x0, x1 = int(math.floor(bx0 - pad)), int(math.ceil(bx1 + pad))
    y0, y1 = int(math.floor(by0 - pad)), int(math.ceil(by1 + pad))
    # Minimum side, centred.
    if x1 - x0 < MIN_SIDE:
        c = (x0 + x1) // 2
        x0, x1 = c - MIN_SIDE // 2, c + MIN_SIDE // 2
    if y1 - y0 < MIN_SIDE:
        c = (y0 + y1) // 2
        y0, y1 = c - MIN_SIDE // 2, c + MIN_SIDE // 2
    # Clamp by shifting, shrink only if larger than the image.
    x0, x1 = _shift_inside(x0, x1, W)
    y0, y1 = _shift_inside(y0, y1, H)
    # Snap up to multiples of 16, extending right/down then shifting; overflow is padded later.
    w16, h16 = _ceil16(x1 - x0), _ceil16(y1 - y0)
    x0 = max(0, min(x0, W - w16))
    y0 = max(0, min(y0, H - h16))
    w_in, h_in = min(w16, W - x0), min(h16, H - y0)
    return C.WorkingRegion(x0, y0, w_in, h_in, 1.0), (x0, y0, w16, h16)


def _shift_inside(a: int, b: int, limit: int) -> tuple[int, int]:
    if b - a >= limit:
        return 0, limit
    if a < 0:
        b, a = b - a, 0
    if b > limit:
        a, b = a - (b - limit), limit
    return max(0, a), b


def still_there(base: np.ndarray, out: np.ndarray, obj: np.ndarray) -> bool:
    """Did the removal leave the object in place? Its inside barely changed (mean largest channel
    difference under 18 of 255), measured away from its edge."""
    from scipy import ndimage

    core = ndimage.binary_erosion(obj, iterations=3)
    if core.sum() < 50:
        core = obj
    if not core.any():
        return False
    d = np.abs(out[core].astype(np.int16) - base[core].astype(np.int16)).max(-1)
    return float(d.mean()) < 18.0


def _ceil16(v: int) -> int:
    return max(16, int(math.ceil(v / 16.0)) * 16)


def worker_dims(w16: int, h16: int, scale: float) -> tuple[int, int]:
    return max(16, int(w16 * scale) // 16 * 16), max(16, int(h16 * scale) // 16 * 16)


def crop_with_edge_pad(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, w, h = box
    H, W = img.shape[:2]
    crop = img[y0:min(H, y0 + h), x0:min(W, x0 + w)]
    pad_h, pad_w = h - crop.shape[0], w - crop.shape[1]
    if pad_h or pad_w:
        pads = [(0, pad_h), (0, pad_w)] + ([(0, 0)] if img.ndim == 3 else [])
        crop = np.pad(crop, pads, mode="edge" if img.ndim == 3 else "constant")
    return crop


CHANGE_MAP_MAX = 2048  # long side of the change map picture (the page shows it scaled to the photo)

# ---- engine ---------------------------------------------------------------------------


class Engine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.manifest = json.loads(Path(cfg.manifest_path).read_text()) if Path(cfg.manifest_path).exists() else {"profiles": {}}
        self.projects: dict[str, Project] = {}
        self.objects: dict[tuple[str, str], ObjectState] = {}
        self.candidates: "OrderedDict[str, Candidate]" = OrderedDict()  # recent ones in memory; all are on disk
        self.segmenter = FakeSegmenter() if cfg.segmenter == "fake" else SamSegmenter(cfg.segmenter_weights, cfg.segmenter)
        self._embedded_for: tuple[str, str | None] | None = None
        self.candidate_meta: dict[str, dict] = {}
        self.grounder = self._make_grounder(cfg)
        self._grounded_for: tuple[str, str | None] | None = None
        self.client = RenderClient(cfg.worker)
        self._render_lock = threading.Lock()

    @staticmethod
    def _make_grounder(cfg: EngineConfig):
        if cfg.grounder == "none":
            return None
        if cfg.grounder == "fake":
            return FakeGrounder(cfg.grounder_table)
        if cfg.grounder == "dino":
            from lis.ground import DinoGrounder

            return DinoGrounder(cfg.grounder_weights)
        if cfg.grounder == "sam3":
            from lis.ground import Sam3Grounder

            return Sam3Grounder(cfg.grounder_weights)
        raise ValueError(f"unknown grounder {cfg.grounder!r}")

    # ---- projects ------------------------------------------------------------------

    def open(self, path: str | Path, embed: bool = True) -> dict:
        """`embed=False` only copies the photo into a new project (safe while a render runs);
        the selection embeddings are then computed on the first selection."""
        p = Project.create(Path(self.cfg.projects_dir), Path(path))
        self.projects[p.record.project_id] = p
        if embed:
            self._ensure_embeddings(p)
        return {"project_id": p.record.project_id, "width": p.record.width, "height": p.record.height, "head": p.record.head}

    def _project(self, project_id: str) -> Project:
        if project_id not in self.projects:
            root = Path(self.cfg.projects_dir) / project_id
            if not (root / "project.json").exists():
                raise KeyError(project_id)
            self.projects[project_id] = Project(root)
        return self.projects[project_id]

    def _ensure_embeddings(self, p: Project) -> None:
        key = (p.record.project_id, p.record.head)
        if self._embedded_for != key:
            self.segmenter.set_image(p.base(p.record.head))
            self._embedded_for = key

    # ---- selection -----------------------------------------------------------------

    def select(self, project_id: str, object_id: str, points: list[tuple[int, int, int]], box: tuple[int, int, int, int] | None) -> dict:
        """`points` is the full list so far for this object: (x, y, label)."""
        if self._render_lock.locked():
            raise EngineBusy("a render is running; selection is disabled until it finishes")
        p = self._project(project_id)
        self._ensure_embeddings(p)
        state = self.objects.setdefault((project_id, object_id), ObjectState())
        state.points = [[int(x), int(y)] for x, y, _ in points]
        state.labels = [int(l) for _, _, l in points]
        state.box = [int(v) for v in box] if box is not None else None
        cands = list(self.segmenter.select(state))
        if state.box is not None and not state.points:
            # the drawn box itself: what "Generate here" needs on empty sky, sand or a wall, where SAM finds no object
            H, W = p.shape
            x0, y0, x1, y1 = state.box
            rect = np.zeros((H, W), bool)
            rect[max(0, min(y0, y1)):min(H, max(y0, y1)), max(0, min(x0, x1)):min(W, max(x0, x1))] = True
            if rect.any():
                cands.append(Candidate(rect, 0.5, None))
        cdir = p.root / "candidates"
        cdir.mkdir(exist_ok=True)
        out = []
        tag = new_id("s")[2:8]  # unique per call: a box after a click has the same number of points, and ids must not repeat
        for i, c in enumerate(cands):
            cid = f"c-{object_id}-{len(state.points)}-{tag}-{i}"
            save_mask_png(c.mask, cdir / f"{cid}.png")
            self._remember(cid, c)
            out.append({"candidate_id": cid, "score": round(c.score, 3), "png_path": str(cdir / f"{cid}.png"),
                        "share": round(float(c.mask.mean()), 4)})
        if cands:
            state.logits = cands[0].logits
        return {"candidates": out}

    def prefetch(self, project_id: str) -> None:
        """Image features for clicks and words, computed before they are asked for."""
        if self._render_lock.locked():
            return
        p = self._project(project_id)
        self._ensure_embeddings(p)
        if self.grounder is not None:
            self._ensure_grounder(p)

    def prepare_text(self, text: str) -> None:
        """Let the idle worker encode the edit sentence now, so Render skips that step."""
        if text and not self._render_lock.locked():
            self.client.prepare(text)

    # ---- selection from words ---------------------------------------

    def _ensure_grounder(self, p: Project) -> None:
        if self.grounder is None:
            raise RuntimeError("text selection is not available: no grounder is configured (start with --grounder dino)")
        key = (p.record.project_id, p.record.head)
        if self._grounded_for != key:
            self.grounder.set_image(p.base(p.record.head))
            self._grounded_for = key

    MAX_CANDIDATES_IN_MEMORY = 48  # a 24 MP mask is 24 MB; older candidates are read back from their PNG

    def _remember(self, cid: str, c: Candidate) -> None:
        self.candidates[cid] = c
        self.candidates.move_to_end(cid)
        while len(self.candidates) > self.MAX_CANDIDATES_IN_MEMORY:
            self.candidates.popitem(last=False)

    def _candidate_mask(self, p: Project, cid: str) -> np.ndarray:
        """A candidate from memory, or from the PNG it was saved as (after a restart, or once it
        has left the in-memory set)."""
        if cid in self.candidates:
            self.candidates.move_to_end(cid)
            return self.candidates[cid].mask
        path = p.root / "candidates" / f"{Path(cid).name}.png"
        if not path.exists():
            raise KeyError(cid)
        return load_mask_png(path, p.shape)

    def _store_candidate(self, p: Project, cid: str, mask: np.ndarray, score: float, meta: dict) -> dict:
        cdir = p.root / "candidates"
        cdir.mkdir(exist_ok=True)
        save_mask_png(mask, cdir / f"{cid}.png")
        self._remember(cid, Candidate(mask, float(score)))
        self.candidate_meta[cid] = meta
        return {"candidate_id": cid, "score": round(float(score), 3), "png_path": str(cdir / f"{cid}.png"), **meta}

    def _find(self, p: Project, phrase: Phrase) -> list[dict]:
        self._ensure_embeddings(p)
        self._ensure_grounder(p)
        items = find_phrase(self.grounder, self.segmenter, p.base(p.record.head), phrase)
        if self.grounder is not None and hasattr(self.grounder, "_torch"):
            self.grounder._torch.cuda.empty_cache()
        return items

    def ground(self, project_id: str, object_id: str, text: str) -> dict:
        """Candidates for a typed phrase ("the red car", "faces"): each instance, best first, and
        when there are several, all of them together. Kept with commit_mask like click candidates."""
        if self._render_lock.locked():
            raise EngineBusy("a render is running; selection is disabled until it finishes")
        p = self._project(project_id)
        phrase = make_phrase(text or "")
        if phrase is None:
            return {"candidates": [], "phrase": None}
        items = self._find(p, phrase)
        out = []
        grounder = getattr(self.grounder, "name", "none")
        order = items
        if phrase.plural and len(items) > 1:
            picked = default_pick(items, True)
            union = np.logical_or.reduce([it["mask"] for it in picked])
            out.append(self._store_candidate(p, f"t-{object_id}-all", union, max(it["score"] for it in picked),
                                             {"text": phrase.label(), "grounder": grounder, "box": None, "label": f"all {len(picked)} {phrase.text}", "confident": True}))
        for i, it in enumerate(order):
            out.append(self._store_candidate(p, f"t-{object_id}-{i}", it["mask"], it["score"],
                                             {"text": phrase.label(), "grounder": grounder, "box": it["found"].box, "confident": bool(it["confident"]),
                                              "label": f"{singular(phrase.text)} {i + 1}" + ("" if it["confident"] else " (weak match, check it)")}))
        if not phrase.plural and len(items) > 1:
            confident = [it for it in items if it["confident"]]
            if len(confident) > 1:
                union = np.logical_or.reduce([it["mask"] for it in confident])
                out.append(self._store_candidate(p, f"t-{object_id}-all", union, max(it["score"] for it in confident),
                                                 {"text": phrase.label(), "grounder": grounder, "box": None, "label": f"all {len(confident)} together", "confident": True}))
        return {"candidates": out, "phrase": phrase.label()}

    def propose(self, project_id: str, instruction: str, target_text: str | None = None, protect_text: str | None = None) -> dict:
        """Read the instruction, find what it names, and return proposals to show before anything
        is kept. `target_text` and `protect_text` (comma separated) override the parse when the
        user has edited them. Nothing is committed here."""
        if self._render_lock.locked():
            raise EngineBusy("a render is running; selection is disabled until it finishes")
        p = self._project(project_id)
        parsed = parse_instruction(instruction or "")
        if target_text is not None:
            tt = target_text.strip().lower()
            parsed.background = tt in {"background", "everything else", "the background"}
            parsed.whole = tt in {"whole photo", "the whole photo", "whole image", "the whole image", "everything", "entire photo", "the photo", "photo"}
            parsed.targets = [] if parsed.background or parsed.whole else [ph for ph in (make_phrase(x) for x in _split_list(tt)) if ph] if tt else []
        if protect_text is not None:
            parsed.protect = [ph for ph in (make_phrase(x) for x in re.split(r"\s*,\s*", protect_text.strip())) if ph] if protect_text.strip() else []
        H, W = p.shape
        stamp = f"{int(time.time() * 1000)}"
        protect_out, prot_by_text = [], {}
        prot = np.zeros((H, W), bool)
        for i, ph in enumerate(parsed.protect):
            items = self._find(p, ph)
            used = default_pick(items, ph.plural)
            entry = {"phrase": ph.label(), "found": len(items), "used": len(used), "weak": bool(used and not used[0]["confident"]), "candidate_id": None}
            if used:
                m = np.logical_or.reduce([it["mask"] for it in used])
                prot |= m
                prot_by_text[ph.text] = prot_by_text.get(ph.text, np.zeros((H, W), bool)) | m
                entry.update(self._store_candidate(p, f"p-{stamp}-{i}", m, max(it["score"] for it in used),
                                                   {"text": ph.label(), "grounder": self.grounder.name, "box": used[0]["found"].box if len(used) == 1 else None,
                                                    "label": ph.label(), "confident": not entry["weak"]}))
            protect_out.append(entry)
        auto_subject = parsed.background and any("main subject is taken" in n for n in parsed.notes)
        if auto_subject and not prot.any():  # no person: the subject may be an animal, a vehicle, food on a plate
            for word in ("animal", "vehicle", "food"):
                ph = make_phrase(word)
                items = [it for it in self._find(p, ph) if it["confident"]]
                share = sum(float(it["mask"].mean()) for it in items)
                if items and 0.01 <= share <= 0.8:
                    prot = np.logical_or.reduce([it["mask"] for it in items])
                    entry = {"phrase": word, "found": len(items), "used": len(items), "weak": False, "candidate_id": None}
                    entry.update(self._store_candidate(p, f"p-{stamp}-auto", prot, max(it["score"] for it in items),
                                                       {"text": word, "grounder": self.grounder.name, "box": None, "label": word, "confident": True}))
                    protect_out = [e for e in protect_out if e["used"]] + [entry]
                    parsed.notes = [n.replace("taken as 'person'", f"taken as '{word}'") for n in parsed.notes]
                    break
            if not prot.any():  # nothing to keep: the model changes the background of the whole photo by itself
                parsed.background, parsed.whole = False, True
                parsed.notes = [n for n in parsed.notes if "main subject is taken" not in n]
                parsed.notes.append("no main subject was found to keep, so the whole photo is the edit area; click the subject and mark it Keep to keep it exact")
        target = {"phrases": [ph.label() for ph in parsed.targets], "background": parsed.background, "whole": parsed.whole,
                  "candidate_id": None, "weak": False, "found": []}
        tmask = np.zeros((H, W), bool)
        if parsed.background:
            if prot.any():
                tmask = ~prot
        if parsed.whole:  # "make it look like winter": the whole photo, except what is kept
            tmask = ~prot
        for ph in parsed.targets:
            items = self._find(p, ph)
            # Skip protected candidates only for other phrases, so "edit the face, keep the face" is a conflict, not another face.
            others = [m for t, m in prot_by_text.items() if t != ph.text]
            used = default_pick(items, ph.plural, np.logical_or.reduce(others) if others else None)
            target["found"].append({"phrase": ph.label(), "found": len(items), "used": len(used),
                                    "ref_missing": bool(ph.ref is not None and items and not items[0].get("ref_found"))})
            if ph.ref is not None and items and not items[0].get("ref_found"):
                parsed.notes.append(f"could not find {ph.ref.text!r}, so {ph.text!r} was picked without it; check the selection")
            if used:
                tmask |= np.logical_or.reduce([it["mask"] for it in used])
                target["weak"] = target["weak"] or not used[0]["confident"]
        if tmask.any() and parsed.adds:
            # Something new needs room: the place's box, grown a quarter of its size (on, in, to), or more than half
            # of it when the new thing goes beside, behind, in front of or near it. Kept areas stay out.
            ys, xs = np.nonzero(tmask)
            bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
            k = 0.6 if parsed.add_where in {"beside", "next to", "near", "behind", "in front of", "above", "over", "under", "beneath", "below"} else 0.25
            gy, gx = int(k * bh), int(k * bw)
            room = np.zeros_like(tmask)
            room[max(0, ys.min() - gy):min(H, ys.max() + gy + 1), max(0, xs.min() - gx):min(W, xs.max() + gx + 1)] = True
            tmask = room & ~prot
        if not tmask.any() and parsed.targets:
            # nothing named was found: the whole photo, so the sentence still works (the model edits what it names)
            missing = " and ".join(ph.label() for ph in parsed.targets)
            parsed.targets, parsed.whole = [], True
            target.update({"whole": True, "phrases": []})
            tmask = ~prot
            parsed.notes = [n for n in parsed.notes if "the selection is where" not in n]
            parsed.notes.append(f"could not find {missing!r}, so the whole photo is the edit area and the model chooses the place; click it or draw a box to choose"
                                if parsed.adds else f"could not find {missing!r}, so the whole photo is the edit area; click it to edit only that part")
        if tmask.any():
            name = "background" if parsed.background else "whole photo" if parsed.whole else " and ".join(ph.label() for ph in parsed.targets)
            target.update(self._store_candidate(p, f"tg-{stamp}", tmask, 1.0, {"text": name, "grounder": self.grounder.name, "box": None,
                                                                                 "label": name, "confident": not target["weak"],
                                                                                 "background": parsed.background}))
        return {"parsed": parsed.to_json(), "edit": parsed.edit, "target": target, "protect": protect_out, "notes": parsed.notes}

    def commit_mask(self, project_id: str, object_id: str, candidate_id: str | None, kind: str, name: str,
                    brush_add_png: str | None = None, brush_subtract_png: str | None = None, invert: bool = False) -> dict:
        """`invert` keeps everything except the selection: the way to target a background."""
        p = self._project(project_id)
        mask = self._candidate_mask(p, candidate_id).copy() if candidate_id else np.zeros(p.shape, dtype=bool)
        if invert:
            mask = ~mask
        mask = self._apply_brush(p, mask, brush_add_png, brush_subtract_png)
        state = self.objects.pop((project_id, object_id), ObjectState())
        prompts = state.prompts()
        prompts.update({"candidate_id": candidate_id, "brush_add": brush_add_png, "brush_subtract": brush_subtract_png, "inverted": invert})
        if candidate_id in self.candidate_meta:  # selected from words: record the phrase and the grounder
            prompts["text"] = {k: v for k, v in self.candidate_meta[candidate_id].items() if k in ("text", "grounder", "box", "confident", "background")}
        rec = p.add_mask(mask, kind, name, prompts)
        return {"mask_id": rec.mask_id}

    def update_mask(self, project_id: str, mask_id: str, brush_add_png: str | None = None, brush_subtract_png: str | None = None) -> dict:
        p = self._project(project_id)
        mask = self._apply_brush(p, p.mask(mask_id), brush_add_png, brush_subtract_png)
        p.update_mask(mask_id, mask, {"brush_updates": (p.mask_record(mask_id).prompts.get("brush_updates", 0) + 1)})
        return {"mask_id": mask_id}

    @staticmethod
    def _apply_brush(p: Project, mask: np.ndarray, add: str | None, sub: str | None) -> np.ndarray:
        if add:
            mask = mask | load_mask_png(Path(add), p.shape)
        if sub:
            mask = mask & ~load_mask_png(Path(sub), p.shape)
        return mask

    # ---- render --------------------------------------------------------------------

    def render(self, plan: dict, cancel: threading.Event | None = None) -> Iterator[dict]:
        """Yield {"type": "progress", ...} messages, then one {"type": "result", **RenderResult}."""
        if not self._render_lock.acquire(blocking=False):
            raise EngineBusy("a render is already running")
        try:
            yield from self._render(plan, cancel)
        finally:
            self._render_lock.release()

    # ---- new pictures from words ---------------------------------------------------------

    GEN_SIZES = {"square": (1024, 1024), "landscape": (1216, 832), "portrait": (832, 1216), "wide": (1344, 768), "tall": (768, 1344)}

    @staticmethod
    def fit_1mp(w: int, h: int) -> tuple[int, int]:
        """The largest multiple-of-16 size with this aspect ratio and at most 1024 x 1024 pixels."""
        s = math.sqrt(MAX_AREA / float(w * h))
        return max(16, int(w * s) // 16 * 16), max(16, int(h * s) // 16 * 16)

    def generate(self, prompt: str, size: str, seed: int, out_path: Path, cancel: threading.Event | None = None,
                 steps: int = 4, refs: list[Path] | None = None) -> Iterator[dict]:
        """A new picture from words (Klein text to image, about 1 MP), optionally drawn from
        reference photos: the open photo for a new pose or scene of the same person, two photos to
        put someone from one into the other. Yields progress, then one {"type": "result", ...}.
        The picture is written to `out_path`; it becomes a project only when the user opens it."""
        if size not in self.GEN_SIZES and size != "photo":
            raise ValueError(f"unknown size {size!r}")
        if not self._render_lock.acquire(blocking=False):
            raise EngineBusy("a render is already running")
        try:
            ref_paths = []
            for k, r in enumerate(refs or []):  # references at most 1 MP, multiples of 16 (the pipeline would crop otherwise)
                im = Image.open(r).convert("RGB")
                rw, rh = self.fit_1mp(*im.size) if im.width * im.height > MAX_AREA else (im.width // 16 * 16, im.height // 16 * 16)
                dst = out_path.parent / f"{out_path.stem}_ref{k}.png"
                im.resize((rw, rh), Image.LANCZOS).save(dst)
                ref_paths.append(str(dst))
            if size == "photo":
                if not ref_paths:
                    raise ValueError("'photo' size needs a reference photo")
                w, h = self.fit_1mp(*Image.open(ref_paths[0]).size)
            else:
                w, h = self.GEN_SIZES[size]
            job = {"type": "job", "id": new_id("g"), "strategy": "generate", "image_path": None, "mask_path": None, "ref_paths": ref_paths,
                   "instruction": prompt, "seed": int(seed), "steps": int(steps), "width": w, "height": h, "out_path": str(out_path)}
            final = None
            t0 = time.time()
            self._park(True)
            try:
                for msg in self.client.render(job, cancel):
                    if msg["type"] == "progress":
                        yield {"type": "progress", "step": msg["step"], "of": msg["of"]}
                    else:
                        final = msg
            except WorkerUnavailable as e:
                final = {"type": "failed", "cause": "worker_unavailable", "detail": str(e)}
            finally:
                self._park(False)
            if final and final["type"] == "done":
                yield {"type": "result", "status": "ok", "path": str(out_path), "width": w, "height": h, "seed": int(seed),
                       "timing_ms": {**final.get("timing_ms", {}), "all": int((time.time() - t0) * 1000)}}
            else:
                cause = (final or {}).get("cause", "error")
                yield {"type": "result", "status": "cancelled" if cause == "cancelled" else "failed",
                       "message": (final or {}).get("detail", "no answer from the worker")}
        finally:
            self._render_lock.release()

    # ---- extending a photo past its edges ---------------------------------------------------

    @staticmethod
    def extend_sides(h: int, w: int, shape: str) -> tuple[int, int, int, int]:
        """(left, right, top, bottom) pixels to add for a named shape, centred."""
        if shape == "wider":
            a = round(w * 0.25); return a, a, 0, 0
        if shape == "taller":
            a = round(h * 0.25); return 0, 0, a, a
        if shape == "all":
            a, b = round(w * 0.15), round(h * 0.15); return a, a, b, b
        ratio = {"square": 1.0, "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4}.get(shape)
        if ratio is None:
            raise ValueError(f"unknown shape {shape!r}")
        if w / h < ratio:  # too narrow: add width
            add = round(h * ratio) - w; return add // 2, add - add // 2, 0, 0
        add = round(w / ratio) - h
        return 0, 0, add // 2, add - add // 2

    def extend(self, project_id: str, shape: str, hint: str, seed: int, out_path: Path,
               cancel: threading.Event | None = None, steps: int = 4, protected_ids: list[str] | None = None) -> Iterator[dict]:
        """The image on screen with new picture added past its edges (Klein edit on a canvas whose
        new border holds the stretched, blurred edge; measured best of four routes,
        )The original's pixels are pasted back exactly, except a band of
        BLEND px inside each new edge where the two are cross-faded so no seam shows."""
        from PIL import ImageFilter

        p = self._project(project_id)
        src = p.base(p.record.head)
        H, W = src.shape[:2]
        L, R, T, B = self.extend_sides(H, W, shape)
        if L + R + T + B <= 0:
            yield {"type": "result", "status": "rejected", "message": "The photo already has that shape."}
            return
        NW, NH = W + L + R, H + T + B
        if NW * NH > 60_000_000:
            yield {"type": "result", "status": "rejected", "message": "The extended photo would be over 60 megapixels. Crop or use a smaller photo first."}
            return
        if not self._render_lock.acquire(blocking=False):
            raise EngineBusy("a render is already running")
        try:
            s = min(1.0, math.sqrt(MAX_AREA / float(NW * NH)))
            ww, wh = max(16, int(NW * s) // 16 * 16), max(16, int(NH * s) // 16 * 16)
            fx, fy = ww / NW, wh / NH
            sl, st = round(L * fx), round(T * fy)
            sw, sh = max(1, round(W * fx)), max(1, round(H * fy))
            small = np.asarray(Image.fromarray(src).resize((sw, sh), Image.LANCZOS))
            canvas = np.pad(small, ((st, max(0, wh - sh - st)), (sl, max(0, ww - sw - sl)), (0, 0)), mode="edge")[:wh, :ww]
            canvas = np.asarray(Image.fromarray(canvas).filter(ImageFilter.GaussianBlur(max(4, round(0.024 * max(ww, wh))))))
            canvas = canvas.copy()
            h2, w2 = min(sh, wh - st), min(sw, ww - sl)
            canvas[st:st + h2, sl:sl + w2] = small[:h2, :w2]
            tmp = out_path.parent / (out_path.stem + "_in.png")
            Image.fromarray(canvas).save(tmp)
            where = [n for n, v in (("left", L), ("right", R), ("top", T), ("bottom", B)) if v]
            sides = " and ".join(where) if len(where) < 4 else "on every side"
            prompt = (f"Extend the photo {'to the ' if len(where) < 4 else ''}{sides}: replace the blurred areas with a sharp, natural "
                      f"continuation of the same scene, same lighting, perspective and style.")
            if hint.strip():
                prompt += " " + hint.strip().rstrip(".") + "."
            job = {"type": "job", "id": new_id("x"), "strategy": "edit", "image_path": str(tmp), "mask_path": None,
                   "instruction": prompt, "seed": int(seed), "steps": int(steps), "width": ww, "height": wh, "out_path": str(out_path)}
            final = None
            t0 = time.time()
            self._park(True)
            try:
                for msg in self.client.render(job, cancel):
                    if msg["type"] == "progress":
                        yield {"type": "progress", "step": msg["step"], "of": msg["of"]}
                    else:
                        final = msg
            except WorkerUnavailable as e:
                final = {"type": "failed", "cause": "worker_unavailable", "detail": str(e)}
            finally:
                self._park(False)
            tmp.unlink(missing_ok=True)
            if not final or final["type"] != "done":
                cause = (final or {}).get("cause", "error")
                yield {"type": "result", "status": "cancelled" if cause == "cancelled" else "failed", "message": (final or {}).get("detail", "no answer from the worker")}
                return
            # On wide shapes from square photos the edit sometimes keeps the blurred border. Then the
            # new border alone is drawn again by the inpaint pipeline from the first result.
            first = np.asarray(Image.open(out_path).convert("RGB").resize((ww, wh), Image.LANCZOS))
            border = np.ones((wh, ww), bool)
            border[st:st + h2, sl:sl + w2] = False
            if border.any() and self._detail(first, border) < 0.35 * self._detail(first, ~border):
                retry_in = out_path.parent / (out_path.stem + "_retry_in.png")
                retry_mask = out_path.parent / (out_path.stem + "_retry_mask.png")
                Image.fromarray(first).save(retry_in)
                from scipy import ndimage as _nd
                Image.fromarray(np.where(_nd.binary_dilation(border, iterations=6), 255, 0).astype(np.uint8), mode="L").save(retry_mask)
                job2 = {**job, "id": new_id("x"), "strategy": "inpaint", "image_path": str(retry_in), "mask_path": str(retry_mask),
                        "seed": int(seed) + 1, "strength": 1.0}
                again = None
                self._park(True)
                try:
                    for msg in self.client.render(job2, cancel):
                        if msg["type"] == "progress":
                            yield {"type": "progress", "step": msg["step"], "of": msg["of"]}
                        else:
                            again = msg
                except WorkerUnavailable:
                    again = None
                finally:
                    self._park(False)
                if not (again and again["type"] == "done"):  # keep the first result rather than fail
                    Image.fromarray(first).save(out_path)
                retry_in.unlink(missing_ok=True)
                retry_mask.unlink(missing_ok=True)
            gen = np.asarray(Image.open(out_path).convert("RGB").resize((NW, NH), Image.LANCZOS)).astype(np.float32)
            # Cross-fade only inside the original, along the edges that got new picture.
            band = max(8, round(0.02 * max(NW, NH)))
            a = np.ones((H, W), np.float32)
            ramp = lambda n: np.clip((np.arange(n, dtype=np.float32) + 0.5) / band, 0.0, 1.0)  # noqa: E731
            if L: a = np.minimum(a, ramp(W)[None, :])
            if R: a = np.minimum(a, ramp(W)[::-1][None, :])
            if T: a = np.minimum(a, ramp(H)[:, None])
            if B: a = np.minimum(a, ramp(H)[::-1][:, None])
            known = {m.mask_id for m in p.record.masks}
            kept = [m for m in (protected_ids or []) if m in known]
            if kept:  # kept areas are never faded, even where they touch a new edge
                a[p.union(kept)] = 1.0
            region = gen[T:T + H, L:L + W]
            gen[T:T + H, L:L + W] = a[..., None] * src.astype(np.float32) + (1.0 - a[..., None]) * region
            out = np.clip(np.floor(gen + 0.5), 0, 255).astype(np.uint8)
            exact = a >= 1.0
            assert np.array_equal(out[T:T + H, L:L + W][exact], src[exact])  # the original, exact away from the blend band
            save_rgb_png(out, out_path)
            yield {"type": "result", "status": "ok", "path": str(out_path), "width": NW, "height": NH, "seed": int(seed),
                   "sides": [L, R, T, B], "band": band, "timing_ms": {**final.get("timing_ms", {}), "all": int((time.time() - t0) * 1000)}}
        finally:
            self._render_lock.release()

    @staticmethod
    def _detail(img: np.ndarray, where: np.ndarray) -> float:
        """Mean absolute Laplacian of the brightness over `where`: low on a blurred area."""
        from scipy import ndimage

        if not where.any():
            return 0.0
        lum = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        return float(np.abs(ndimage.laplace(lum))[where].mean())

    def _park(self, away: bool) -> None:
        """Move the selection models to system memory (away=True) or back to the GPU. About
        2.6 GB of VRAM for SAM 2.1 plus SAM 3; the cached image features are dropped and
        recomputed at the next selection."""
        if not self.cfg.park_selection:
            return
        moved = False
        for holder in (self.segmenter, self.grounder):
            model = getattr(holder, "model", None)
            if model is None or not hasattr(model, "to"):
                continue
            model.to("cpu" if away else holder.device)
            for attr in ("embeddings", "vision"):
                if hasattr(holder, attr):
                    setattr(holder, attr, None)
            moved = True
        if moved:
            self._embedded_for = None
            self._grounded_for = None
            import torch

            torch.cuda.empty_cache()

    def _result(self, status: str, message: str = "", **extra) -> dict:
        base = {"type": "result", "status": status, "revision_id": None, "image_path": None, "strategy_used": None,
                "working_region": None, "protected_check": {"status": "not_applicable", "mismatched_pixels": 0}, "drift": 0.0,
                "expansion_hint_mask_path": None, "message": message, "timing_ms": {}, "backend": {"profile_id": self.cfg.profile_id},
                "outside_check": {"status": "not_applicable", "mismatched_pixels": 0}, "edit_check": {"status": "not_checked"}, "change_map_path": None}
        base.update(extra)
        return base

    def _render(self, plan: dict, cancel: threading.Event | None) -> Iterator[dict]:
        p = self._project(plan["project_id"])
        t_all = time.time()
        res = resolve(plan, p, self.manifest)
        if not res.ok:
            yield self._result("rejected", res.message)
            return
        op = plan["operation"]
        base = p.base(p.record.head)
        target, protected = res.target, res.protected
        shadow = None
        if op.get("kind") == "move" and op.get("shadow", True):  # find its shadow: the model removes both, both are moved
            t0 = time.time()
            try:
                if self.grounder is not None:
                    self._ensure_grounder(p)
                shadow, _how = find_shadow(base, target & ~protected, self.grounder)
                shadow &= ~protected
            except Exception:
                shadow = None
            if shadow is not None and shadow.any():
                target = target | shadow
                op = {**op, "instruction": re.sub(r"^remove the (.+?) and fill in", r"remove the \1 and its shadow and fill in", op["instruction"])}
            else:
                shadow = None
            timing_shadow = int((time.time() - t0) * 1000)
        has_protected = bool(protected.any())
        band_px = int(op.get("band_px", 8))
        feather = C.feather_px(target, band_px) if res.blend == "tight" else max(C.feather_px(target, band_px), int(0.06 * max(base.shape[:2])))
        grow = C.grow_px(feather)
        t_core = target & ~protected
        repaint = C.repaint_region(target, protected, feather, grow)
        region, box = working_region(base.shape[:2], t_core, feather, grow)
        x0, y0, w16, h16 = box
        scale = min(1.0, math.sqrt(MAX_AREA / float(w16 * h16)))

        rev_id = new_id("r")
        run_dir = p.root / "runs" / rev_id
        run_dir.mkdir(parents=True)
        base_box = crop_with_edge_pad(base, box)
        mask_box = crop_with_edge_pad(repaint, box)
        timing: dict = {"shadow": timing_shadow} if op.get("kind") == "move" and op.get("shadow", True) else {}
        final: dict | None = None
        notes: list[str] = [res.message] if res.message else []

        timing["prepare"] = int((time.time() - t_all) * 1000)
        dims = [w16, h16]

        def work(seed: int, instruction: str):
            """Run the worker (one retry at a smaller size if the GPU runs out of memory); returns its last message."""
            nonlocal scale
            final = None
            t0 = time.time()
            self._park(True)  # small GPUs: the editing model gets the memory the SAM models held
            timing["park"] = int((time.time() - t0) * 1000)
            t_w = time.time()
            try:
                for attempt in range(2):
                    ww, wh = worker_dims(w16, h16, scale)
                    dims[:] = [ww, wh]
                    w_img = Image.fromarray(base_box, mode="RGB")
                    w_mask = Image.fromarray(np.where(mask_box, 255, 0).astype(np.uint8), mode="L")
                    if (ww, wh) != (w16, h16):
                        w_img = w_img.resize((ww, wh), Image.LANCZOS)
                        w_mask = w_mask.resize((ww, wh), Image.NEAREST)
                    w_img.save(run_dir / "W.png")
                    w_mask.save(run_dir / "W_mask.png")
                    job = {"type": "job", "id": rev_id, "strategy": res.strategy, "image_path": str(run_dir / "W.png"),
                           "mask_path": str(run_dir / "W_mask.png") if res.strategy == "inpaint" else None,
                           "instruction": instruction, "seed": seed, "steps": int(plan.get("steps", 4)),
                           "strength": float(plan.get("strength", 0.8)), "width": ww, "height": wh, "out_path": str(run_dir / "candidate.png")}
                    try:
                        for msg in self.client.render(job, cancel):
                            if msg["type"] == "progress":
                                yield {"type": "progress", "step": msg["step"], "of": msg["of"], "attempt": attempt}
                            else:
                                final = msg
                    except WorkerUnavailable as e:
                        final = {"type": "failed", "cause": "worker_unavailable", "detail": str(e)}
                    if final and final["type"] == "failed" and final.get("cause") == "out_of_memory" and attempt == 0:
                        scale *= OOM_SHRINK
                        notes.append(f"The GPU ran out of memory; retried at {scale:.0%} of the working region.")
                        yield {"type": "progress", "step": 0, "of": int(plan.get("steps", 4)), "attempt": 1, "note": notes[-1]}
                        continue
                    break
            finally:
                timing["worker_wall"] = timing.get("worker_wall", 0) + int((time.time() - t_w) * 1000)
                t0 = time.time()
                self._park(False)
                timing["unpark"] = int((time.time() - t0) * 1000)
            return final

        final = yield from work(int(plan.get("seed", 0)), op["instruction"])
        ww, wh = dims

        if not final or final["type"] != "done":
            cause = (final or {}).get("cause", "error")
            status = "cancelled" if cause == "cancelled" else "failed"
            if status == "cancelled":  # nothing to inspect: free the disk
                shutil.rmtree(run_dir, ignore_errors=True)
            else:
                self._fail_run(p, run_dir)
            yield self._result(status, " ".join([*notes, (final or {}).get("detail", "no answer from the worker")]).strip(),
                               backend={"profile_id": self.cfg.profile_id, "cause": cause})
            return
        timing.update(final.get("timing_ms", {}))

        # Candidate back to canonical scale, trimmed to the in-image part of the box.
        def composed():
            cand = decode_canonical(run_dir / "candidate.png")
            if cand.shape[:2] != (h16, w16):
                cand = np.asarray(Image.fromarray(cand).resize((w16, h16), Image.LANCZOS), dtype=np.uint8)
            cand = cand[: region.h, : region.w]
            reg = C.WorkingRegion(region.x, region.y, region.w, region.h, scale)
            return reg, C.compose(base, cand, reg, target, protected, C.CompositeParams(band_px=band_px, color_match=self.cfg.color_match, blend=res.blend))

        t0 = time.time()
        region, comp = composed()
        timing["composite_prepare"] = int((time.time() - t0) * 1000)
        removing = op.get("kind") == "move" or (bool(REMOVAL_PATTERN.search(op["instruction"])) and not (res.target & ~protected).all())
        if removing and still_there(base, comp.out, res.target & ~protected):
            # the model left the object where it was (it happens on some seeds): one more try (a move would duplicate it)
            first = run_dir / "candidate_first_try.png"
            shutil.copy2(run_dir / "candidate.png", first)
            again = yield from work(int(plan.get("seed", 0)) + 1, op["instruction"].rstrip(". ") + ". Remove it completely and show only what was behind it.")
            if again and again["type"] == "done":
                region2, comp2 = composed()
                if not still_there(base, comp2.out, res.target & ~protected):
                    region, comp = region2, comp2
                    final = again
                    notes.append("The first try left it in its old place; a second seed removed it.")
                else:
                    shutil.copy2(first, run_dir / "candidate.png")
                    notes.append("Two tries may have left it in its old place, so it may appear twice. If so, try another seed, or paint its old place and remove it."
                                 if op.get("kind") == "move" else "It may still be there after two tries. If so, try another seed, or select it more tightly.")
            else:
                shutil.copy2(first, run_dir / "candidate.png")
        moved_note = None
        if op.get("kind") == "move":  # the model filled the old place; the object itself is copied, not redrawn
            moved_note = self._place_moved(base, comp, res.target, protected, op, shadow)
            notes.append(moved_note)
        timing["composite_compose"] = int((time.time() - t0) * 1000) - timing["composite_prepare"]
        save_rgb_png(comp.out, run_dir / "out.png")
        save_mask_png(comp.alpha > 0, run_dir / "allowed.png")  # the only pixels this edit may change
        hint_path = None
        if comp.expansion_hint is not None:
            hint_path = run_dir / "expansion_hint.png"
            save_mask_png(comp.expansion_hint, hint_path)
        timing["composite"] = int((time.time() - t0) * 1000)

        # The check is on files, not arrays.
        t0 = time.time()
        protected_ids = list(plan.get("protected_mask_ids") or [])
        if has_protected:  # the protected area as it was at this render: later mask edits must not change this version's proof
            save_mask_png(protected, run_dir / "protected.png")
        check = self._file_check(p, run_dir / "out.png", p.base_path(p.record.head), protected_ids, run_dir / "protected.png")
        outside = self._outside_check(p, run_dir / "out.png", p.base_path(p.record.head), run_dir / "allowed.png")
        timing["check"] = int((time.time() - t0) * 1000)
        if check["status"] == "fail" or outside["status"] == "fail":
            self._fail_run(p, run_dir)
            what = (f"{check['mismatched_pixels']} protected pixels" if check["status"] == "fail" else
                    f"{outside['mismatched_pixels']} pixels outside the edit area")
            yield self._result("failed", f"Render discarded: {what} changed. This is a bug; the files are kept under runs/failed for inspection.",
                               protected_check=check, outside_check=outside)
            return
        t0 = time.time()
        self._change_map(base, comp.out, comp.alpha > 0, protected, run_dir / "change_map.png")  # a picture for the eye; the checks above read the file
        timing["change_map"] = int((time.time() - t0) * 1000)

        # Did the requested change happen? A report, never a gate.
        t0 = time.time()
        try:
            if moved_note:
                edit_check = {"status": "not_checked", "kind": "move",
                              "message": "Moved: the object's own pixels (and its shadow, when one was found) were placed at the new spot and its old place was filled by the model. Judge the old place by eye."}
            else:
                tmeta = p.mask_record(op["target_mask_id"]).prompts.get("text") or {}
                edit_check = verify_edit(base, comp.out, target, protected, op["instruction"], tmeta.get("text"), self.grounder, self.segmenter)
        except Exception as e:  # the check must never break a good render
            edit_check = {"status": "not_checked", "kind": None, "message": f"edit check failed: {type(e).__name__}"}
        self._grounded_for = None
        self._embedded_for = None
        timing["edit_check"] = int((time.time() - t0) * 1000)

        t_fin = time.time()
        profile = self.manifest.get("profiles", {}).get(self.cfg.profile_id, {})
        rev = Revision.new(
            parent=p.record.head,
            operation={"kind": op["kind"], "strategy_requested": op.get("strategy", "auto"), "strategy_used": res.strategy,
                       "blend_requested": op.get("blend", "auto"), "blend_used": res.blend,
                       "instruction": op["instruction"], "target_mask_id": op["target_mask_id"], "protected_mask_ids": protected_ids,
                       "band_px": band_px, "feather": comp.feather, "grow": comp.grow, "seed": int(plan.get("seed", 0)),
                       "steps": int(plan.get("steps", 4)), "strength": float(plan.get("strength", 0.8)),
                       **({"offset": [int(v) for v in op["offset"]], "shadow": {"asked": bool(op.get("shadow", True)), "moved": shadow is not None,
                                                                              "pixels": int(shadow.sum()) if shadow is not None else 0}}
                          if op.get("kind") == "move" else {})},
            working_region={"x": region.x, "y": region.y, "w": region.w, "h": region.h, "scale": round(scale, 4), "worker_size": [ww, wh]},
            backend={"profile_id": self.cfg.profile_id, "manifest_sha256": {k: v.get("sha256") for k, v in profile.get("files", {}).items() if k.endswith(".safetensors")},
                     "last_verified": self.manifest.get("last_verified")},
            candidate_sha256=sha256_file(run_dir / "candidate.png"), output_sha256=sha256_file(run_dir / "out.png"),
            protected_check=check, drift=round(comp.drift, 3), timing_ms={**timing, "total": int((time.time() - t_all) * 1000)},
            outside_check=outside, edit_check=edit_check,
        )
        rev.revision_id = rev_id
        History(p).add(rev)
        rev.timing_ms["record"] = int((time.time() - t_fin) * 1000)
        (run_dir / "meta.json").write_text(json.dumps(rev.__dict__, indent=2, default=str))
        self._embedded_for = None  # the base changed; embeddings are recomputed on the next select
        yield self._result("ok", " ".join(notes).strip(), revision_id=rev_id, image_path=str(run_dir / "out.png"), strategy_used=res.strategy, blend_used=res.blend,
                           working_region=rev.working_region, protected_check=check, outside_check=outside, edit_check=edit_check,
                           change_map_path=str(run_dir / "change_map.png"), drift=rev.drift,
                           expansion_hint_mask_path=str(hint_path) if hint_path else None, timing_ms=rev.timing_ms, backend=rev.backend)

    @staticmethod
    def _fail_run(p: Project, run_dir: Path) -> None:
        dst = p.root / "runs" / "failed" / run_dir.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(dst))

    @staticmethod
    def _outside_check(p: Project, out_path: Path, base_path: Path, allowed_path: Path) -> dict:
        """Every pixel that differs between the written output and the base must lie in the
        compositor's allowed region (alpha above 0). Checked on the files."""
        if not Path(allowed_path).exists():
            return {"status": "not_applicable", "mismatched_pixels": 0}
        out = decode_canonical(out_path)
        base = decode_canonical(base_path)
        allowed = load_mask_png(Path(allowed_path), p.shape)
        changed = np.any(out != base, axis=-1)
        n_out = int(np.count_nonzero(changed & ~allowed))
        return {"status": "pass" if n_out == 0 else "fail", "mismatched_pixels": n_out, "changed_pixels": int(changed.sum()),
                "allowed_pixels": int(allowed.sum()), "changed_fraction": round(float(changed.mean()), 4)}

    @staticmethod
    def _place_moved(base: np.ndarray, comp, target: np.ndarray, protected: np.ndarray, op: dict, shadow: np.ndarray | None = None) -> str:
        """Put the object's own pixels (from the base, exact inside, a 1 to 2 px soft edge) at
        the offset, over the model's removal result. Kept areas stay in front and exact. The
        allowed area grows by where the object lands, so the outside check still holds.

        With a shadow: its dimming (base / the model's clean ground, lis.shadow.shadow_factor)
        is carried to the offset and multiplies the ground there, under the object."""
        from scipy import ndimage

        H, W = target.shape
        dx, dy = (int(v) for v in op["offset"])
        src = target & ~protected
        soft = np.clip(ndimage.gaussian_filter(src.astype(np.float32), 1.0) * 1.6, 0.0, 1.0)
        soft[ndimage.binary_erosion(src, iterations=2)] = 1.0
        soft[~ndimage.binary_dilation(src, iterations=3)] = 0.0

        def shifted(a: np.ndarray, fill: float = 0) -> np.ndarray:
            out = np.full_like(a, fill)
            ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
            xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
            if abs(dx) < W and abs(dy) < H:
                out[yd, xd] = a[ys, xs]
            return out

        a = shifted(soft)
        a[protected] = 0.0
        obj = shifted(base)
        out = comp.out.copy()
        shadow_note = ""
        if shadow is not None:
            got = shadow_factor(base, comp.out, shadow, src)
            if got is None:
                shadow_note = " Its shadow was found but the model left it in place, so it was not copied."
            else:
                f, (sy, sx) = got
                F = np.ones((H, W, 3), np.float32)
                F[sy, sx] = f
                Fs = shifted(F, 1.0)
                Fs[protected] = 1.0
                dim = (Fs < 1.0).any(-1)
                out[dim] = np.clip(np.floor(out[dim].astype(np.float32) * Fs[dim] + 0.5), 0, 255).astype(np.uint8)
                comp.alpha = np.maximum(comp.alpha, dim.astype(comp.alpha.dtype))
                shadow_note = " Its shadow moved with it."
        band = a > 0
        av = a[band][:, None]
        out[band] = np.clip(np.floor(av * obj[band].astype(np.float32) + (1.0 - av) * out[band].astype(np.float32) + 0.5), 0, 255).astype(np.uint8)
        out[protected] = base[protected]
        comp.out = out
        comp.alpha = np.maximum(comp.alpha, a)
        comp.expansion_hint = None
        kept = int(src.sum())
        landed = int((shifted(src) & ~protected).sum())
        msg = f"Moved {abs(dx)} px {'right' if dx >= 0 else 'left'} and {abs(dy)} px {'down' if dy >= 0 else 'up'}."
        if landed < kept:
            msg += f" {100 - round(100 * landed / max(1, kept))}% of it is now outside the photo or behind a kept area."
        if shadow is None and op.get("shadow", True):
            shadow_note = " No shadow was found next to it."
        return msg + shadow_note

    @staticmethod
    def _change_map(base: np.ndarray, out: np.ndarray, allowed: np.ndarray, protected: np.ndarray, path: Path) -> None:
        """Dimmed photo; changed pixels in orange, protected in blue, the allowed area outlined.

        A picture for the eye, made at most CHANGE_MAP_MAX pixels on the long side: each cell of
        f x f pixels is marked changed if any pixel in it changed (so no change is hidden by the
        reduction). Full resolution took 9.6 s on a 13.5 MP photo; this takes well under 1 s."""
        from scipy import ndimage

        H, W = base.shape[:2]
        f = max(1, -(-max(H, W) // CHANGE_MAP_MAX))
        h, w = -(-H // f), -(-W // f)

        def pool(a: np.ndarray, how: str) -> np.ndarray:
            if f == 1:
                return a
            pad = [(0, h * f - H), (0, w * f - W)] + [(0, 0)] * (a.ndim - 2)
            a = np.pad(a, pad, mode="edge")
            a = a.reshape(h, f, w, f, *a.shape[2:])
            return a.max(axis=(1, 3)) if how == "max" else a.mean(axis=(1, 3))

        diff = np.abs(out.astype(np.int16) - base.astype(np.int16)).max(-1).astype(np.uint8)
        dmax = pool(diff, "max").astype(np.float32)
        changed = dmax > 0
        img = pool(base, "mean").astype(np.float32) * 0.35
        prot = pool(protected, "max")
        img[prot] = img[prot] * 0.5 + np.array([0, 200, 255], np.float32) * 0.5
        strength = np.clip(dmax / 40.0, 0.25, 1.0)
        img[changed] = img[changed] * (1 - strength[changed, None]) + np.array([255, 140, 0], np.float32) * strength[changed, None]
        al = pool(allowed, "max")
        edge = al & ~ndimage.binary_erosion(al, iterations=max(1, max(h, w) // 500))
        img[edge] = (255, 255, 255)
        save_rgb_png(img.clip(0, 255).astype(np.uint8), path)

    @staticmethod
    def _file_check(p: Project, out_path: Path, base_path: Path, protected_ids: list[str], snapshot: Path | None = None) -> dict:
        if not protected_ids:
            return {"status": "not_applicable", "mismatched_pixels": 0}
        out = decode_canonical(out_path)
        base = decode_canonical(base_path)
        if snapshot is not None and snapshot.exists():
            prot = load_mask_png(snapshot, p.shape)
        else:  # versions made before snapshots were kept
            prot = np.zeros(p.shape, dtype=bool)
            for mid in protected_ids:
                prot |= load_mask_png(p.root / "masks" / f"{mid}.png", p.shape)
        if not prot.any():
            return {"status": "not_applicable", "mismatched_pixels": 0}
        n = C.count_mismatched(out, base, prot)
        return {"status": "pass" if n == 0 else "fail", "mismatched_pixels": n}

    # ---- history and export ---------------------------------------------------------

    def history(self, project_id: str) -> dict:
        h = History(self._project(project_id))
        return {"revisions": h.summaries(), "head": h.head}

    def set_head(self, project_id: str, revision_id: str | None) -> dict:
        p = self._project(project_id)
        History(p).set_head(revision_id)
        self._embedded_for = None
        return {"head": p.record.head}

    def export(self, project_id: str, revision_id: str | None, path: str | Path, fmt: str = "png") -> dict:
        p = self._project(project_id)
        src = p.base_path(revision_id)  # a revision's out.png, or source.png for None
        check = {"status": "not_applicable", "mismatched_pixels": 0}
        if revision_id is not None:
            rev = History(p).get(revision_id)
            check = self._file_check(p, src, p.base_path(rev.parent), rev.operation.get("protected_mask_ids", []),
                                     p.root / "runs" / revision_id / "protected.png")
            outside = self._outside_check(p, src, p.base_path(rev.parent), p.root / "runs" / revision_id / "allowed.png")
            if outside["status"] == "fail":
                check = {**check, "status": "fail", "outside_mismatched_pixels": outside["mismatched_pixels"]}
        img = decode_canonical(src)
        path = Path(path)
        if fmt == "png":
            save_rgb_png(img, path)
        elif fmt == "jpeg":
            Image.fromarray(img, mode="RGB").save(path, format="JPEG", quality=95)
        else:
            raise ValueError("format must be png or jpeg")
        return {"ok": check["status"] != "fail", "bytes": path.stat().st_size, "protected_check": check, "format": fmt}

    def close(self) -> None:
        self.client.close()
