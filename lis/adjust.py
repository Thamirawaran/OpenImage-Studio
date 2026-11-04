"""Edits without a model: Adjust (brightness, contrast, saturation, warmth) and the geometry
operations (rotate, flip, crop).

Adjust makes an ordinary revision with the same proof as a render: protected pixels are copied
from the base exactly, `allowed.png` records the only pixels the adjustment may change, and the
engine's file checks (`_file_check` against the `protected.png` snapshot, `_outside_check`
against `allowed.png`) run on the files written to disk. A failed check discards the result.

Geometry changes every pixel's position, which the project's masks and history cannot follow,
so rotate, flip and crop make a new project from the image on screen (a lossless PNG) and leave
the old project untouched.

The adjustment formula. The page's live preview (lis/static/app.js, `adjustPixels`) computes the
same steps, so what the user sees while dragging is what Apply writes. Values are integers from
-100 to 100; 0 leaves that step out. Channels are r, g, b in 0..1.

    1. brightness, per channel:  x = x ** (2 ** (-brightness / 100))
       a gamma curve: black and white stay where they are, +100 lifts mid grey from 0.50 to 0.71
    2. contrast, per channel:    x = (x - 0.5) * (1 + contrast / 100) + 0.5
       -100 is flat grey, +100 doubles the distance from mid grey
    3. saturation:               L = 0.299 r + 0.587 g + 0.114 b
                                 x = L + (x - L) * (1 + saturation / 100)
       -100 is black and white, +100 doubles the colour
    4. warmth:                   r = r * (1 + 0.2 * warmth / 100), b = b * (1 - 0.2 * warmth / 100)
       a warmer or cooler white balance; black stays black
    5. out = clip(floor(x * 255 + 0.5), 0, 255)

Steps 1 and 2 act on each channel alone and are one 256-entry table. Where the adjustment applies
to a selection, its edge gets a soft ring of `edge_px` (a few pixels) outside the selection:
alpha = min(1, 2 * gaussian(T_core, sigma = edge_px / 2)), forced to 1 on T_core and to 0 on the
protected union P, and out = floor(alpha * adjusted + (1 - alpha) * base + 0.5), then
out[P] = base[P]. Over the whole photo alpha is 1 everywhere except P, so the edge of a kept
object is exact.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from lis.history import History, Revision
from lis.project import Project, decode_canonical, new_id, save_mask_png, save_rgb_png, sha256_file

KEYS = ("brightness", "contrast", "saturation", "warmth")
LUMA = (0.299, 0.587, 0.114)
WARMTH_GAIN = 0.2
ROWS_PER_CHUNK = 512  # bounds the float32 working memory on large photos


class AdjustError(ValueError):
    """A request the user can fix, with a message for them."""


def clean_values(raw: dict) -> dict:
    """The four slider values as integers in -100..100. Raises AdjustError otherwise."""
    out = {}
    for k in KEYS:
        v = raw.get(k, 0)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise AdjustError(f"{k} must be a number from -100 to 100")
        if not np.isfinite(fv) or fv != int(fv) or not -100 <= fv <= 100:
            raise AdjustError(f"{k} must be a whole number from -100 to 100")
        out[k] = int(fv)
    return out


def describe(values: dict) -> str:
    """'Adjust: brightness +20, warmth +10', the history row's text."""
    parts = [f"{k} {values[k]:+d}" for k in KEYS if values.get(k)]
    return "Adjust: " + ", ".join(parts) if parts else "Adjust: no change"


def tone_table(values: dict) -> np.ndarray:
    """Steps 1 and 2 as a 256-entry float32 table, indexed by the uint8 value."""
    x = np.arange(256, dtype=np.float64) / 255.0
    b, c = values.get("brightness", 0), values.get("contrast", 0)
    if b:
        x = x ** (2.0 ** (-b / 100.0))
    if c:
        x = (x - 0.5) * (1.0 + c / 100.0) + 0.5
    return x.astype(np.float32)


def adjusted(img: np.ndarray, values: dict, lut: np.ndarray | None = None) -> np.ndarray:
    """Steps 1 to 4 on a uint8 (h, w, 3) array; returns float32 in 0..1, not yet clipped."""
    lut = tone_table(values) if lut is None else lut
    x = lut[img]  # (h, w, 3) float32
    s, w = values.get("saturation", 0), values.get("warmth", 0)
    if s:
        L = x[..., 0] * LUMA[0] + x[..., 1] * LUMA[1] + x[..., 2] * LUMA[2]
        k = np.float32(1.0 + s / 100.0)
        x = L[..., None] + (x - L[..., None]) * k
    if w:
        x[..., 0] *= np.float32(1.0 + WARMTH_GAIN * w / 100.0)
        x[..., 2] *= np.float32(1.0 - WARMTH_GAIN * w / 100.0)
    return x


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(x * 255.0 + 0.5), 0, 255).astype(np.uint8)


def edge_px(shape: tuple[int, int]) -> int:
    """The soft ring outside a selection: a few pixels, 2 on small photos, 6 at 6000 px."""
    return int(min(6, max(2, round(max(shape) / 1000.0))))


def alpha_for(shape: tuple[int, int], target: np.ndarray | None, protected: np.ndarray, edge: int) -> tuple[np.ndarray, tuple[slice, slice]]:
    """Alpha over the bounding box it covers, and that box. `target` None means the whole photo."""
    H, W = shape
    if target is None:
        return (~protected).astype(np.float32), (slice(0, H), slice(0, W))
    t_core = target & ~protected
    ys, xs = np.nonzero(t_core)
    reach = 2 * edge + 2  # the blur is truncated at 3 sigma = 1.5 edge, plus a margin
    y0, y1 = max(0, int(ys.min()) - reach), min(H, int(ys.max()) + 1 + reach)
    x0, x1 = max(0, int(xs.min()) - reach), min(W, int(xs.max()) + 1 + reach)
    box = (slice(y0, y1), slice(x0, x1))
    core = t_core[box]
    a = ndimage.gaussian_filter(core.astype(np.float32), sigma=edge / 2.0, mode="nearest", truncate=3.0)
    a = np.minimum(1.0, 2.0 * a)
    a[a < 1e-3] = 0.0  # no invisible tail: allowed.png is exactly where the adjustment can show
    a[core] = 1.0
    a[protected[box]] = 0.0
    return a, box


def compose(base: np.ndarray, values: dict, target: np.ndarray | None, protected: np.ndarray, edge: int) -> tuple[np.ndarray, np.ndarray]:
    """(out uint8 (H, W, 3), allowed bool (H, W)). Pure numpy; protected pixels are copied from base."""
    H, W = base.shape[:2]
    alpha, (ys, xs) = alpha_for((H, W), target, protected, edge)
    out = base.copy()
    allowed = np.zeros((H, W), bool)
    allowed[ys, xs] = alpha > 0
    lut = tone_table(values)
    lut8 = to_uint8(lut) if not values.get("saturation") and not values.get("warmth") else None  # tone only: one uint8 table, same result
    y0 = ys.start
    for r0 in range(0, alpha.shape[0], ROWS_PER_CHUNK):
        r1 = min(alpha.shape[0], r0 + ROWS_PER_CHUNK)
        a = alpha[r0:r1]
        if not a.any():
            continue
        b = base[y0 + r0:y0 + r1, xs]
        adj = lut8[b] if lut8 is not None else to_uint8(adjusted(b, values, lut))
        full = a >= 1.0
        part = (a > 0) & ~full
        blk = np.where(full[..., None], adj, b)
        if part.any():
            ap = a[part][:, None]
            blk[part] = np.floor(ap * adj[part].astype(np.float32) + (1.0 - ap) * b[part].astype(np.float32) + 0.5).clip(0, 255).astype(np.uint8)
        out[y0 + r0:y0 + r1, xs] = blk
    out[protected] = base[protected]  # copy-back, exact
    return out, allowed


# ---- the revision -----------------------------------------------------------------------


def adjust(eng, project_id: str, values: dict, target_ids: list[str], protected_ids: list[str], parent: str | None = "head") -> dict:
    """Apply an adjustment to the image on screen and record it as a revision. Raises AdjustError
    for a request that would change nothing; returns {"status": "ok" | "failed", ...}."""
    from lis.engine import EngineBusy

    if eng._render_lock.locked():
        raise EngineBusy("a render is running; adjust when it finishes")
    t_all = time.time()
    values = clean_values(values)
    if not any(values.values()):
        raise AdjustError("All four sliders are at 0, so nothing would change. Move a slider first.")
    p: Project = eng._project(project_id)
    if parent != "head" and parent != p.record.head:
        raise AdjustError("The image changed since the sliders were moved. Look at the current image and apply again.")
    timing: dict = {}
    t0 = time.time()
    base = p.base(p.record.head)
    protected = p.union(protected_ids)
    target = p.union(target_ids) if target_ids else None
    if target is not None and not (target & ~protected).any():
        raise AdjustError("The edit area is entirely inside a kept area, so nothing can change. Unkeep it or pick another area.")
    if not (~protected).any():
        raise AdjustError("The whole photo is kept exactly, so nothing can change.")
    edge = edge_px(p.shape)
    timing["load"] = int((time.time() - t0) * 1000)
    t0 = time.time()
    out, allowed = compose(base, values, target, protected, edge)
    timing["adjust"] = int((time.time() - t0) * 1000)

    rev_id = new_id("r")
    run_dir = p.root / "runs" / rev_id
    run_dir.mkdir(parents=True)
    t0 = time.time()
    save_rgb_png(out, run_dir / "out.png")
    save_mask_png(allowed, run_dir / "allowed.png")
    if protected.any():  # the protected area as it was now: later mask edits must not change this version's proof
        save_mask_png(protected, run_dir / "protected.png")
    timing["write"] = int((time.time() - t0) * 1000)

    # The checks are on the files, exactly as for a render.
    t0 = time.time()
    check = eng._file_check(p, run_dir / "out.png", p.base_path(p.record.head), list(protected_ids), run_dir / "protected.png")
    outside = eng._outside_check(p, run_dir / "out.png", p.base_path(p.record.head), run_dir / "allowed.png")
    timing["check"] = int((time.time() - t0) * 1000)
    if check["status"] == "fail" or outside["status"] == "fail":
        eng._fail_run(p, run_dir)
        what = (f"{check['mismatched_pixels']} protected pixels" if check["status"] == "fail" else
                f"{outside['mismatched_pixels']} pixels outside the adjusted area")
        return {"status": "failed", "message": f"Adjustment discarded: {what} changed. This is a bug; the files are kept under runs/failed for inspection.",
                "protected_check": check, "outside_check": outside}
    if outside.get("changed_pixels", 1) == 0:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise AdjustError("These values change no pixels in the adjusted area. Move a slider further.")
    _change_map_later(eng, base, out, allowed, protected, run_dir)

    area = "edit area" if target is not None else "whole photo"
    if target is not None:
        ys, xs = np.nonzero(allowed)
        region = {"x": int(xs.min()), "y": int(ys.min()), "w": int(xs.max() - xs.min() + 1), "h": int(ys.max() - ys.min() + 1), "scale": 1.0}
    else:
        region = {"x": 0, "y": 0, "w": p.record.width, "h": p.record.height, "scale": 1.0}
    rev = Revision.new(
        parent=p.record.head,
        operation={"kind": "adjust", "values": values, "instruction": describe(values), "area": area,
                   "target_mask_id": None, "target_mask_ids": list(target_ids), "protected_mask_ids": list(protected_ids),
                   "edge_px": edge if target is not None else 0, "model": None},
        working_region=region,
        backend={"profile_id": None, "model": "none: a fixed formula computed with numpy"},
        candidate_sha256="", output_sha256=sha256_file(run_dir / "out.png"),
        protected_check=check, drift=0.0, timing_ms={**timing, "total": int((time.time() - t_all) * 1000)},
        outside_check=outside,
        edit_check={"status": "not_checked", "kind": "adjust", "message": "No model was used, so there is nothing to verify: the adjustment is a fixed formula."},
    )
    rev.revision_id = rev_id
    History(p).add(rev)
    (run_dir / "meta.json").write_text(json.dumps(rev.__dict__, indent=2, default=str))
    eng._embedded_for = None  # the base changed; embeddings are recomputed on the next select
    eng._grounded_for = None
    return {"status": "ok", "revision_id": rev_id, "protected_check": check, "outside_check": outside, "timing_ms": rev.timing_ms}


def _change_map_later(eng, base, out, allowed, protected, run_dir: Path) -> None:
    """The What changed picture, drawn after the answer: on a 12 MP photo it took longer than the
    adjustment, the write and both checks together. Until it exists the page shows the photo."""
    def run():
        tmp = run_dir / "change_map.tmp.png"
        try:
            eng._change_map(base, out, allowed, protected, tmp)
            tmp.replace(run_dir / "change_map.png")
        except Exception as e:  # noqa: BLE001  a missing map never affects the result or its proof
            print(f"change map failed: {e!r}", file=sys.stderr)
            tmp.unlink(missing_ok=True)

    threading.Thread(target=run, daemon=True).start()


# ---- geometry: a new project ------------------------------------------------------------

GEOMETRY = {"rotate_left": "Rotated left", "rotate_right": "Rotated right", "flip_horizontal": "Flipped horizontally", "crop": "Cropped"}
MIN_CROP = 16


def transformed(img: np.ndarray, op: str, box: list | tuple | None = None) -> np.ndarray:
    """The image rotated a quarter turn, mirrored left to right, or cropped to (x, y, w, h).
    Exact pixel moves: no resampling."""
    if op == "rotate_left":
        return np.ascontiguousarray(np.rot90(img, 1))
    if op == "rotate_right":
        return np.ascontiguousarray(np.rot90(img, -1))
    if op == "flip_horizontal":
        return np.ascontiguousarray(img[:, ::-1])
    if op == "crop":
        x, y, w, h = check_crop(box, img.shape[:2])
        return np.ascontiguousarray(img[y:y + h, x:x + w])
    raise AdjustError(f"Unknown operation {op!r}. Use rotate_left, rotate_right, flip_horizontal or crop.")


def check_crop(box, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    H, W = shape
    try:
        x, y, w, h = (int(v) for v in box)
        if any(float(v) != int(v) for v in box):
            raise ValueError
    except (TypeError, ValueError):
        raise AdjustError("The crop needs a box of four whole numbers: x, y, width, height.")
    if w < MIN_CROP or h < MIN_CROP:
        raise AdjustError(f"The crop is {w}x{h}; it must be at least {MIN_CROP} pixels on each side.")
    if x < 0 or y < 0 or x + w > W or y + h > H:
        raise AdjustError(f"The crop {w}x{h} at {x},{y} goes outside the photo ({W}x{H}).")
    if (x, y, w, h) == (0, 0, W, H):
        raise AdjustError("The crop covers the whole photo, so nothing would change.")
    return x, y, w, h


def note_for(op: str, name: str, box=None) -> str:
    if op == "crop":
        x, y, w, h = (int(v) for v in box)
        return f"Cropped from {name}, {w}x{h} at {x},{y}"
    return f"{GEOMETRY[op]} from {name}"


def transform(eng, project_id: str, op: str, box=None, work_dir: Path | None = None) -> dict:
    """Make a new project from the image on screen, rotated, flipped or cropped. The old project
    is not changed. Returns eng.open's answer for the new project, plus the note."""
    if op not in GEOMETRY:
        raise AdjustError(f"Unknown operation {op!r}. Use rotate_left, rotate_right, flip_horizontal or crop.")
    p: Project = eng._project(project_id)
    img = p.base(p.record.head)
    if op == "crop":
        box = check_crop(box, img.shape[:2])
    new = transformed(img, op, box)
    name = p.record.original_name
    stem = Path(name).stem or "photo"
    suffix = {"rotate_left": "rotated-left", "rotate_right": "rotated-right", "flip_horizontal": "flipped", "crop": "cropped"}[op]
    work_dir = Path(work_dir or (Path(eng.cfg.projects_dir) / ".uploads" / new_id("g")))
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp = work_dir / f"{stem}-{suffix}.png"
    try:
        save_rgb_png(new, tmp)
        o = eng.open(tmp, embed=False)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    q = eng._project(o["project_id"])
    note = note_for(op, name, box)
    if p.record.head is not None:
        note += f" (version {p.record.head})"
    q.record.import_notes = [note, *(q.record.import_notes or [])]
    q.save()
    return {**o, "note": note}
