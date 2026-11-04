"""Localisation and the preservation guarantee.

Pure numpy and scipy. Places the candidate on the working region, builds the repaint
region D and the gaussian alpha (grow = 4 + feather // 2, sigma = feather / 4), forces alpha
to 0 on protected pixels after the blur, blends over the base, copies protected pixels back
from the base. The file-level equality check is engine.py's; `count_mismatched` is the
shared primitive. Testable on a laptop without a GPU.

Conventions: images are uint8 arrays of shape (H, W, 3); masks are bool arrays of shape
(H, W); everything is in canonical pixel space.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

# ---- exact parallel versions of the slow filters ------------------------------------------
# SciPy's filters release the GIL, so strips of rows run on several cores. Each strip carries
# a margin of rows at least as wide as the filter's reach, so every output value is computed
# from exactly the same inputs in the same order as on the whole array: the results are
# bit-identical (tests/test_composite.py checks against SciPy directly). A 13.5 MP edit spent
# 9 s here on one core.
_THREADS = max(1, min(8, (os.cpu_count() or 2)))
_POOL = ThreadPoolExecutor(_THREADS) if _THREADS > 1 else None
_PAR_MIN = 1_000_000  # pixels; below this one call is faster than the strips


def _strips(n: int, reach: int) -> list[tuple[int, int, int, int]]:
    """(out0, out1, in0, in1) row ranges: output rows and the input rows they need."""
    k = min(_THREADS, max(1, n // max(64, 2 * reach)))
    edges = np.linspace(0, n, k + 1).astype(int)
    return [(int(a), int(b), max(0, int(a) - reach), min(n, int(b) + reach)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def gauss(x: np.ndarray, sigma, mode: str = "reflect") -> np.ndarray:
    """ndimage.gaussian_filter(x, sigma, mode=mode) with the default truncate of 4, in parallel."""
    s0 = float(sigma[0] if isinstance(sigma, (tuple, list)) else sigma)
    reach = int(4.0 * s0 + 0.5) + 1
    if _POOL is None or x.shape[0] * (x.shape[1] if x.ndim > 1 else 1) < _PAR_MIN or s0 <= 0:
        return ndimage.gaussian_filter(x, sigma, mode=mode)
    parts = _strips(x.shape[0], reach)
    if len(parts) < 2:
        return ndimage.gaussian_filter(x, sigma, mode=mode)
    out = np.empty(x.shape, dtype=x.dtype)  # SciPy returns the input dtype

    def run(t):
        a, b, i0, i1 = t
        if i0 == 0 and i1 == x.shape[0]:
            out[a:b] = ndimage.gaussian_filter(x, sigma, mode=mode)[a:b]
            return
        # Margins only on the inside: at the array's own edges the boundary mode applies as before.
        out[a:b] = ndimage.gaussian_filter(x[i0:i1], sigma, mode=mode)[a - i0:b - i0]

    list(_POOL.map(run, parts))
    return out


def within(background: np.ndarray, radius: float) -> np.ndarray:
    """ndimage.distance_transform_edt(background) <= radius, in parallel: the zero (feature)
    pixel nearest to any pixel within the radius lies inside its strip's margin, so the answer
    is the same; pixels with no feature pixel in reach are False either way."""
    reach = int(np.ceil(radius)) + 1
    if background.all():  # no feature pixel: nothing is within reach (SciPy measures from outside the array here)
        return np.zeros(background.shape, dtype=bool)
    if _POOL is None or background.size < _PAR_MIN:
        return ndimage.distance_transform_edt(background) <= radius
    parts = _strips(background.shape[0], reach)
    if len(parts) < 2:
        return ndimage.distance_transform_edt(background) <= radius
    out = np.empty(background.shape, dtype=bool)

    def run(t):
        a, b, i0, i1 = t
        tile = background[i0:i1]
        if tile.all():  # no feature pixel within reach of these rows
            out[a:b] = False
            return
        out[a:b] = (ndimage.distance_transform_edt(tile) <= radius)[a - i0:b - i0]

    list(_POOL.map(run, parts))
    return out


@dataclass(frozen=True)
class WorkingRegion:
    """Where the model looked, in canonical pixels. scale <= 1 maps worker pixels to canonical."""

    x: int
    y: int
    w: int
    h: int
    scale: float = 1.0


@dataclass(frozen=True)
class CompositeParams:
    band_px: int = 8  # floor of the feather width; the edge also scales with image size (feather_px)
    grow_offset: int = 4  # Krita's binary grow before the blur
    color_match: bool = True
    hint_threshold: int = 24  # max-channel |C - B| that counts as "the model wanted to change this"
    hint_min_fraction: float = 0.001  # of the working region's area, below which no hint is offered
    blend: str = "tight"  # tight: narrow edge on the mask. context: accept connected spill, wide edge
    detail: bool = True  # restore the source's fine detail where a downscaled edit kept the structure
    spill_threshold: int = 20  # max-channel change that counts as spill in context mode
    spill_reach: float = 0.15  # of the image long side: how far spill may extend from the target


@dataclass
class CompositeResult:
    out: np.ndarray  # uint8 (H, W, 3)
    alpha: np.ndarray  # float32 (H, W), 0 outside the working region
    repaint: np.ndarray  # bool (H, W): the region D handed to the inpaint pipeline
    feather: int
    grow: int
    drift: float  # mean |C - B| over W outside D and P, in 0..255 units
    expansion_hint: np.ndarray | None  # bool (H, W) or None
    color_matched: bool


EDGE_FRACTION = 0.004  # soft edge width as a fraction of the image's long side (7 px at 1800, 16 at 4000)


def feather_px(target: np.ndarray, band_px: int) -> int:
    """max(band_px, 0.4 percent of the image's long side).

    A SAM mask edge is a real object edge, so the transition must be narrow: a wide band
    (Krita's 10 percent of the selection diagonal, meant for inpainting seams) smeared the
    edited colour into neighbouring parts in the first real run (review 2026-11-03)."""
    return int(max(band_px, round(EDGE_FRACTION * max(target.shape))))


def grow_px(feather: int, grow_offset: int = 4) -> int:
    return int(grow_offset + feather // 2)


def place_candidate(candidate: np.ndarray, region: WorkingRegion) -> np.ndarray:
    """Return the candidate at the working region's canonical size (LANCZOS if it was scaled)."""
    if candidate.ndim != 3 or candidate.shape[2] != 3 or candidate.dtype != np.uint8:
        raise ValueError("candidate must be uint8 (h, w, 3)")
    if candidate.shape[0] == region.h and candidate.shape[1] == region.w:
        return candidate
    img = Image.fromarray(candidate, mode="RGB").resize((region.w, region.h), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def count_mismatched(out: np.ndarray, base: np.ndarray, protected: np.ndarray | None) -> int:
    """Number of protected pixels where any channel of `out` differs from `base`."""
    if protected is None or not protected.any():
        return 0
    return int(np.count_nonzero(np.any(out[protected] != base[protected], axis=-1)))


def repaint_region(target: np.ndarray, protected: np.ndarray | None, feather: int, grow: int) -> np.ndarray:
    """D = dilate(T_core, disc(grow)) & ~P, the region handed to the inpaint pipeline as its mask.
    Computed over the target's bounding box plus margin; identical to what compose() builds."""
    H, W = target.shape
    if protected is None:
        protected = np.zeros((H, W), dtype=bool)
    t_core = target & ~protected
    out = np.zeros((H, W), dtype=bool)
    ys, xs = np.nonzero(t_core)
    if ys.size == 0:
        return out
    m = grow + 2
    y0, y1 = max(0, int(ys.min()) - m), min(H, int(ys.max()) + 1 + m)
    x0, x1 = max(0, int(xs.min()) - m), min(W, int(xs.max()) + 1 + m)
    out[y0:y1, x0:x1] = within(~t_core[y0:y1, x0:x1], grow) & ~protected[y0:y1, x0:x1]
    return out


def context_feather_px(region_mask: np.ndarray, band_px: int) -> int:
    """Wide edge for context blends: 5 percent of the changed region's diagonal, at least 1
    percent and at most 6 percent of the image long side."""
    long_side = max(region_mask.shape)
    ys, xs = np.nonzero(region_mask)
    diag = float(np.hypot(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)) if ys.size else 0.0
    return int(max(band_px, round(0.01 * long_side), min(round(0.06 * long_side), round(0.05 * diag))))


def _connected_spill(base: np.ndarray, cand_w: np.ndarray, region: WorkingRegion, t_core: np.ndarray,
                     protected: np.ndarray, params: CompositeParams) -> np.ndarray:
    """Pixels next to the target that the model changed as a consequence (shadows, reflections,
    background behind a removed object): changed by more than spill_threshold, within
    spill_reach of the target, connected to it, never protected."""
    H, W = base.shape[:2]
    wy, wx = slice(region.y, region.y + region.h), slice(region.x, region.x + region.w)
    diff = np.abs(cand_w.astype(np.int16) - base[wy, wx].astype(np.int16)).max(axis=-1).astype(np.float32)
    diff = gauss(diff, 2.0)
    reach = params.spill_reach * max(H, W)
    near = within(~t_core[wy, wx], reach)
    changed = (diff > params.spill_threshold) & near & ~protected[wy, wx]
    changed = ndimage.binary_closing(changed, structure=np.ones((5, 5), bool))
    lab, n = ndimage.label(changed | t_core[wy, wx], structure=np.ones((3, 3), bool))
    keep = np.unique(lab[t_core[wy, wx]])
    keep = keep[keep > 0]
    spill_w = np.isin(lab, keep) & changed & ~t_core[wy, wx]
    spill_w = ndimage.binary_fill_holes(spill_w | t_core[wy, wx]) & ~t_core[wy, wx] & ~protected[wy, wx]
    out = np.zeros((H, W), dtype=bool)
    out[wy, wx] = spill_w
    return out


def restore_detail(base_w: np.ndarray, cand_w: np.ndarray, scale: float, min_sigma: float = 0.0,
                   crop: tuple[slice, slice] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Add back the source's own fine detail that a downscaled edit lost, only where the edit kept
    the structure.

    The detail is what the round trip removed: B minus B resampled down by `scale` and back up.
    The weight is the local normalised cross-correlation between the candidate and that low-pass
    base: high where edges line up (recolours, material changes), low where the content changed
    (removed or added objects), so invented texture is never added and real texture is never
    pasted onto new content. Detail is transferred in luminance so colours stay the candidate's.
    `crop` limits the work to part of W (the low-pass base is still made from all of W, so the
    values inside the crop are the same as without it, away from the crop's own edges).
    Returns (candidate float32, weight), for the crop when one is given."""
    crop = crop or (slice(None), slice(None))
    c = cand_w[crop].astype(np.float32)
    if scale >= 0.999 and min_sigma <= 0:
        return c, np.zeros(c.shape[:2], np.float32)
    h, w = base_w.shape[:2]
    if scale < 0.999:
        small = Image.fromarray(base_w).resize((max(16, round(w * scale)), max(16, round(h * scale))), Image.LANCZOS)
        b_low = np.asarray(small.resize((w, h), Image.LANCZOS), dtype=np.uint8)[crop].astype(np.float32)
        b = base_w[crop].astype(np.float32)
    else:
        b = base_w[crop].astype(np.float32)
        b_low = gauss(b, (min_sigma, min_sigma, 0))
    luma = lambda x: x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
    hf = luma(b) - luma(b_low)
    lc, lb = luma(c), luma(b_low)
    win = max(3.0, 2.5 / max(scale, 0.05))
    mc, mb = gauss(lc, win), gauss(lb, win)
    cov = gauss(lc * lb, win) - mc * mb
    vc = np.maximum(gauss(lc * lc, win) - mc * mc, 1e-3)
    vb = np.maximum(gauss(lb * lb, win) - mb * mb, 1e-3)
    ncc = cov / np.sqrt(vc * vb)
    weight = np.clip((ncc - 0.3) / 0.5, 0.0, 1.0).astype(np.float32)  # 0 below 0.3, 1 above 0.8
    # Scale the detail to the candidate's local contrast so a darkened surface gets darker detail.
    gain = np.clip(np.sqrt(vc / vb), 0.25, 2.0)
    c = c + (weight * gain * hf)[..., None]
    return np.clip(c, 0.0, 255.0), weight


def _round_half_up_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.floor(x + 0.5), 0, 255).astype(np.uint8)


def compose(
    base: np.ndarray,
    candidate: np.ndarray,
    region: WorkingRegion,
    target: np.ndarray,
    protected: np.ndarray | None,
    params: CompositeParams = CompositeParams(),
) -> CompositeResult:
    """Blend the candidate into the base inside the target plus a soft band, keep protected
    pixels exactly, and report drift and an expansion hint. Work is confined to a region of
    interest around the target; outside it the output is the base and alpha is zero."""
    if base.ndim != 3 or base.shape[2] != 3 or base.dtype != np.uint8:
        raise ValueError("base must be uint8 (H, W, 3)")
    H, W = base.shape[:2]
    if target.shape != (H, W) or target.dtype != bool:
        raise ValueError("target must be a bool (H, W) mask")
    if protected is None:
        protected = np.zeros((H, W), dtype=bool)
    if protected.shape != (H, W) or protected.dtype != bool:
        raise ValueError("protected must be a bool (H, W) mask")
    if not (0 <= region.x and 0 <= region.y and region.x + region.w <= W and region.y + region.h <= H):
        raise ValueError("working region lies outside the base")

    t_core = target & ~protected
    wy, wx = slice(region.y, region.y + region.h), slice(region.x, region.x + region.w)
    cand_w = place_candidate(candidate, region)  # (h, w, 3) at canonical scale, covers W
    if params.detail and region.scale < 0.999:  # real source detail, gated by structure
        # Only near the target: everything that can reach the output (spill, edge, blur) plus the
        # correlation window. Large photos otherwise spent most of the composite here.
        ty, tx = np.nonzero(t_core[wy, wx])
        if ty.size:
            long_side = max(H, W)
            win = max(3.0, 2.5 / max(region.scale, 0.05))
            reach = (params.spill_reach + 0.07) if params.blend == "context" else 0.02
            pad = int(reach * long_side) + int(np.ceil(4 * win)) + 64
            crop = (slice(max(0, int(ty.min()) - pad), min(region.h, int(ty.max()) + 1 + pad)),
                    slice(max(0, int(tx.min()) - pad), min(region.w, int(tx.max()) + 1 + pad)))
            restored, _ = restore_detail(base[wy, wx], cand_w, region.scale, crop=crop)
            cand_w = cand_w.copy()
            cand_w[crop] = _round_half_up_u8(restored)

    spill_full = np.zeros((H, W), dtype=bool)
    if params.blend == "context" and t_core.any():
        spill_full = _connected_spill(base, cand_w, region, t_core, protected, params)
        feather = context_feather_px(t_core | spill_full, params.band_px)
    else:
        feather = feather_px(target, params.band_px)
    grow = grow_px(feather, params.grow_offset)
    sigma = feather / 4.0

    out = base.copy()
    alpha_full = np.zeros((H, W), dtype=np.float32)
    repaint_full = np.zeros((H, W), dtype=bool)

    ys, xs = np.nonzero(t_core | spill_full)
    if ys.size == 0:
        # Nothing to repaint (target wholly protected or empty). Policy should have rejected this.
        return CompositeResult(out, alpha_full, repaint_full, feather, grow, 0.0, None, False)

    # Region of interest: target bbox grown by everything that can be non-zero, clipped to W.
    margin = grow + int(np.ceil(4 * sigma)) + 2
    y0 = max(region.y, int(ys.min()) - margin)
    y1 = min(region.y + region.h, int(ys.max()) + 1 + margin)
    x0 = max(region.x, int(xs.min()) - margin)
    x1 = min(region.x + region.w, int(xs.max()) + 1 + margin)
    roi = (slice(y0, y1), slice(x0, x1))
    roi_in_w = (slice(y0 - region.y, y1 - region.y), slice(x0 - region.x, x1 - region.x))

    tc = t_core[roi]
    sp = spill_full[roi]
    pr = protected[roi]
    c_u8 = cand_w[roi_in_w]

    # D: dilate T_core by a disc of radius grow. Distance transform of the complement.
    d = within(~(tc | sp), grow) & ~pr  # protected pixels are never part of the repaint region
    repaint_full[roi] = d

    # Alpha: blur D, then force the core to 1 and protected to 0 (after the blur).
    alpha = gauss(d.astype(np.float32), sigma, mode="nearest")
    np.clip(alpha, 0.0, 1.0, out=alpha)
    alpha[tc] = 1.0
    alpha[pr] = 0.0

    # Ring just outside D, used for colour matching and the expansion hint.
    ring = ~d & within(~d, feather) & ~pr

    # Robust colour match: only pixels the model left essentially unchanged tell us about a
    # global tone shift. Pixels it changed on purpose (spill onto a neighbour) are excluded,
    # and the correction is bounded so it can never repaint the edit itself.
    # Per-channel |C - B| over W, computed once in integers: used by the colour match, the drift
    # and the expansion hint (whole-W float copies made large photos slow).
    b_w_u8 = base[wy, wx]
    dch = [np.abs(cand_w[..., k].astype(np.int16) - b_w_u8[..., k].astype(np.int16)) for k in range(3)]
    diff_w = np.maximum(np.maximum(dch[0], dch[1]), dch[2])  # max over channels, (h, w)
    color_matched = False
    if params.color_match:
        quiet = ~repaint_full[wy, wx] & ~protected[wy, wx] & (diff_w <= params.hint_threshold)
        nq = int(np.count_nonzero(quiet))
        if nq >= 1024:
            offset = np.array([(b_w_u8[..., k].sum(where=quiet, dtype=np.float64) - cand_w[..., k].sum(where=quiet, dtype=np.float64)) / nq
                               for k in range(3)], dtype=np.float32)
            offset = np.clip(offset, -12.0, 12.0)
            if float(np.abs(offset).max()) >= 1.0:
                color_matched = True

    # alpha * C + (1 - alpha) * B, computed only where alpha > 0: where it is 0 the result is B
    # and where it is 1 it is C, exactly, so this equals the whole-array blend bit for bit.
    out_roi = base[roi].copy()
    band = alpha > 0.0
    a = alpha[band][:, None]
    c = c_u8[band].astype(np.float32)
    if color_matched:
        c = np.clip(c + offset, 0.0, 255.0)
    out_roi[band] = _round_half_up_u8(a * c + (1.0 - a) * base[roi][band].astype(np.float32))
    out_roi[pr] = base[roi][pr]  # copy-back, exact
    out[roi] = out_roi
    alpha_full[roi] = alpha

    # Diagnostics over the whole working region.
    outside = ~repaint_full[wy, wx] & ~protected[wy, wx]
    n_out = int(np.count_nonzero(outside))
    drift = float(sum(dk.sum(where=outside, dtype=np.float64) for dk in dch) / (3 * n_out)) if n_out else 0.0

    # Expansion hint: solid regions just outside D that the model changed, touching D. Speckle
    # from re-rendered texture (grass, fabric weave) is not a hint: it is dropped by an opening
    # and a minimum component size (measured 3 Nov: most raw hints were hundreds of specks).
    hint_full: np.ndarray | None = None
    raw = ring & (diff_w[roi_in_w] > params.hint_threshold)
    raw = ndimage.binary_opening(raw, structure=np.ones((3, 3), bool))
    if raw.any():
        lab, n = ndimage.label(raw, structure=np.ones((3, 3), bool))
        touching = set(np.unique(lab[ndimage.binary_dilation(d, iterations=2) & raw]).tolist()) - {0}
        min_comp = max(200, int(0.0005 * region.w * region.h))
        sizes = ndimage.sum(raw, lab, index=range(1, n + 1))
        keep = [i + 1 for i, sz in enumerate(sizes) if (i + 1) in touching and sz >= min_comp]
        hint_roi = np.isin(lab, keep)
        if keep and int(hint_roi.sum()) >= params.hint_min_fraction * region.w * region.h:
            hint_full = np.zeros((H, W), dtype=bool)
            hint_full[roi] = hint_roi

    if params.blend == "context":
        hint_full = None  # spill was accepted, there is nothing left to offer
    return CompositeResult(out, alpha_full, repaint_full, feather, grow, drift, hint_full, color_matched)
