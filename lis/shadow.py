"""The shadow an object casts, so a move can take it along.

find_shadow: SAM 3 is asked for "shadow"; an instance is the object's shadow when it touches the
object, is mostly outside it, and is darker than the ground around it. Without SAM 3 (or when it
finds none) a fallback looks for a dark patch joined to the object: pixels clearly darker than
the ground just around them, with the same colour direction (a shadow dims a surface, it does not
recolour it).

shadow_factor: how much the shadow dims each pixel, as base / clean, where clean is the model's
result with the object and shadow removed. Moving the shadow is then multiplying the ground at
the new place by that factor, which keeps the new ground's own texture."""
from __future__ import annotations

import numpy as np

MIN_FRACTION = 0.03  # a shadow smaller than 3% of the object is not worth moving
MAX_FRACTION = 2.0  # nor one over twice the object's size (a building's shadow, a dark floor, another object)


def _lum(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _box(mask: np.ndarray, pad: int, shape) -> tuple[slice, slice] | None:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    H, W = shape
    return slice(max(0, ys.min() - pad), min(H, ys.max() + pad + 1)), slice(max(0, xs.min() - pad), min(W, xs.max() + pad + 1))


def _versus_around(image: np.ndarray, lum: np.ndarray, m: np.ndarray, obj: np.ndarray, reach: int) -> tuple[float, float]:
    """(brightness of m / brightness of the ground in a ring around it, cosine between their mean
    colours). A shadow is darker than the ground around it with about the same colour; a dark
    object (a car, a bag) usually has another colour."""
    from scipy import ndimage

    ring = ndimage.binary_dilation(m, iterations=reach) & ~m & ~obj
    if ring.sum() < 20 or m.sum() < 1:
        return 1.0, 0.0
    a = image[m].astype(np.float64).mean(0) + 1.0
    b = image[ring].astype(np.float64).mean(0) + 1.0
    return float(lum[m].mean() / max(1.0, lum[ring].mean())), float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_shadow(image: np.ndarray, obj: np.ndarray, grounder=None) -> tuple[np.ndarray, str]:
    """(shadow mask, how): how is "sam3", "dark" or "none". The mask never overlaps obj."""
    from scipy import ndimage

    H, W = obj.shape
    none = np.zeros_like(obj)
    area = int(obj.sum())
    if area < 50:
        return none, "none"
    size = max(4, int(area ** 0.5))
    lum = _lum(image)
    touch = ndimage.binary_dilation(obj, iterations=max(3, size // 25))

    reach = ndimage.binary_dilation(obj, iterations=max(8, int(1.2 * size)))  # a shadow starts at the object

    def acceptable(m: np.ndarray) -> bool:
        a = int(m.sum())
        if not (MIN_FRACTION * area <= a <= MAX_FRACTION * area and bool((m & touch).any())):
            return False
        dark, cos = _versus_around(image, lum, m, obj, max(4, size // 10))
        return dark < 0.85 and cos > 0.985

    if grounder is not None and getattr(grounder, "name", "") == "sam3":
        got = none.copy()
        try:
            found = grounder.find("shadow")
        except Exception:
            found = []
        for f in found:
            if f.mask is None or f.score < 0.2:
                continue
            fm = f.mask.astype(bool)
            if fm.sum() == 0 or (fm & obj).sum() > 0.6 * fm.sum():
                continue
            m = fm & ~obj & reach
            lab, n = ndimage.label(m)  # only the parts joined to the object
            if n > 1:
                m = np.isin(lab, [i for i in range(1, n + 1) if ((lab == i) & touch).any()])
            if acceptable(m):
                got |= m
        if got.any():
            return got, "sam3"

    # Fallback: a dark patch joined to the object, within about one object size of it.
    near = ndimage.binary_dilation(obj, iterations=size) & ~obj
    bb = _box(near, 2 * size, (H, W))
    if bb is None:
        return none, "none"
    L = ndimage.gaussian_filter(lum[bb], 1.0)
    o, nr = obj[bb], near[bb]
    ground = ~o
    # the ground's brightness without the shadow: a wide average of the ground, its darkest third left out
    lit = ground & (L >= np.percentile(L[ground], 35)) if ground.any() else ground
    w = ndimage.gaussian_filter(lit.astype(np.float32), size / 2)
    ref = ndimage.gaussian_filter(np.where(lit, L, 0.0).astype(np.float32), size / 2) / np.maximum(w, 1e-3)
    img = image[bb].astype(np.float32)
    rgb_ref = np.stack([ndimage.gaussian_filter(np.where(lit, img[..., c], 0.0), size / 2) / np.maximum(w, 1e-3) for c in range(3)], -1)
    # same colour direction: the pixel is roughly a dimmed copy of the lit ground
    cos = (img * rgb_ref).sum(-1) / np.maximum(1e-3, np.linalg.norm(img, axis=-1) * np.linalg.norm(rgb_ref, axis=-1))
    dark = nr & (L < 0.72 * ref) & (cos > 0.97) & (ref > 40)
    dark = ndimage.binary_opening(dark, iterations=1)
    lab, n = ndimage.label(dark)
    if n == 0:
        return none, "none"
    t = touch[bb]
    keep = np.zeros_like(dark)
    for i in range(1, n + 1):
        c = lab == i
        if (c & t).any():
            keep |= c
    keep = ndimage.binary_closing(keep, iterations=2) & ~o
    out = none.copy()
    out[bb] = keep
    if not acceptable(out):
        return none, "none"
    return out, "dark"


def shadow_factor(base: np.ndarray, clean: np.ndarray, shadow: np.ndarray, obj: np.ndarray | None = None) -> tuple[np.ndarray, tuple[slice, slice]] | None:
    """Per-pixel, per-channel dimming in [0.08, 1] over the shadow's box (1 outside the shadow),
    with a soft 1 to 3 px edge. None when the shadow is empty or the removal left it in place.
    The thin gap between the object and its shadow (where the object's outline and the shadow's
    meet) takes the darkest dimming next to it, or the moved ground shows there as a bright line."""
    from scipy import ndimage

    bb = _box(shadow, 6, shadow.shape)
    if bb is None:
        return None
    b = base[bb].astype(np.float32)
    c = clean[bb].astype(np.float32)
    b = np.stack([ndimage.gaussian_filter(b[..., k], 1.2) for k in range(3)], -1)
    c = np.stack([ndimage.gaussian_filter(c[..., k], 1.2) for k in range(3)], -1)
    r = np.clip((b + 2.0) / (c + 2.0), 0.08, 1.0)
    s = shadow[bb]
    soft = np.clip(ndimage.gaussian_filter(s.astype(np.float32), 1.5) * 1.4, 0.0, 1.0)
    soft[ndimage.binary_erosion(s, iterations=2)] = 1.0
    f = 1.0 - soft[..., None] * (1.0 - r)
    if float((1.0 - f[s]).mean()) < 0.04:  # the removal did not lift it: nothing to carry
        return None
    if obj is not None:
        o = obj[bb]
        gap = ndimage.binary_dilation(o, iterations=5) & ndimage.binary_dilation(s, iterations=6) & ~ndimage.binary_erosion(s, iterations=2)
        if gap.any():
            inner = np.where(ndimage.binary_erosion(s, iterations=2)[..., None], f, 1.0)
            darkest = np.stack([ndimage.minimum_filter(inner[..., k], size=13) for k in range(3)], -1)
            f[gap] = np.minimum(f[gap], darkest[gap])
    return f, bb
