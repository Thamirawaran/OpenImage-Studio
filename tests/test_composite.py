import numpy as np
import pytest
from scipy import ndimage

from lis.composite import (
    CompositeParams,
    WorkingRegion,
    compose,
    count_mismatched,
    feather_px,
    grow_px,
)


def disc(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def rect(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def rng_image(h, w, seed):
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_protected_pixels_identical_and_core_is_candidate():
    H, W = 400, 600
    base = rng_image(H, W, 1)
    candidate = (255 - base).astype(np.uint8)  # differs everywhere
    target = disc(H, W, 200, 300, 90)
    protected = disc(H, W, 200, 360, 50)  # overlaps the target
    res = compose(base, candidate, WorkingRegion(0, 0, W, H), target, protected,
                  CompositeParams(color_match=False))
    assert count_mismatched(res.out, base, protected) == 0
    assert np.array_equal(res.out[protected], base[protected])
    core = target & ~protected
    assert np.array_equal(res.out[core], candidate[core])
    assert res.alpha[protected].max() == 0.0
    assert res.repaint[core].all() and not res.repaint[protected].any()


def test_far_outside_is_base_exactly():
    H, W = 500, 500
    base = rng_image(H, W, 2)
    candidate = (255 - base).astype(np.uint8)
    target = disc(H, W, 250, 250, 60)
    res = compose(base, candidate, WorkingRegion(0, 0, W, H), target, None,
                  CompositeParams(color_match=False))
    dist = ndimage.distance_transform_edt(~res.repaint)
    far = dist >= res.feather
    assert np.array_equal(res.out[far], base[far])


@pytest.mark.parametrize("band_px,shape", [(32, "disc"), (300, "disc"), (32, "rect"), (300, "rect")])
def test_alpha_step_at_selection_edge_is_small(band_px, shape):
    H, W = 1400, 1400
    base = np.zeros((H, W, 3), dtype=np.uint8)
    candidate = np.full((H, W, 3), 255, dtype=np.uint8)
    target = disc(H, W, 700, 700, 300) if shape == "disc" else rect(H, W, 400, 1000, 400, 1000)
    res = compose(base, candidate, WorkingRegion(0, 0, W, H), target, None,
                  CompositeParams(band_px=band_px, color_match=False))
    assert res.feather >= band_px
    dist = ndimage.distance_transform_edt(~target)
    just_outside = (dist > 0) & (dist <= 1.0)
    step = 1.0 - float(res.alpha[just_outside].min())
    limit = 0.03 if shape == "disc" else 0.06  # convex corners are blurred from two sides
    assert step <= limit, f"alpha step {step:.3f} at feather {res.feather}"


def test_candidate_placed_at_offset_and_scale():
    H, W = 300, 400
    base = rng_image(H, W, 3)
    region = WorkingRegion(x=96, y=48, w=192, h=160, scale=0.5)
    # Worker returned a half-size candidate for the region; a flat colour survives LANCZOS exactly.
    candidate_small = np.full((80, 96, 3), (10, 200, 30), dtype=np.uint8)
    target = rect(H, W, 100, 160, 150, 230)  # well inside the region
    res = compose(base, candidate_small, region, target, None, CompositeParams(color_match=False))
    assert (res.out[target] == np.array([10, 200, 30], dtype=np.uint8)).all()
    outside_region = np.ones((H, W), dtype=bool)
    outside_region[region.y:region.y + region.h, region.x:region.x + region.w] = False
    assert np.array_equal(res.out[outside_region], base[outside_region])
    assert res.alpha[outside_region].max() == 0.0


def test_expansion_hint_when_model_changed_the_ring():
    H, W = 600, 600
    base = np.full((H, W, 3), 100, dtype=np.uint8)
    target = disc(H, W, 300, 300, 80)
    candidate = base.copy()
    changed = disc(H, W, 300, 300, 80 + 40)  # the model altered a 40 px ring around the target
    candidate[changed] = 220
    res = compose(base, candidate, WorkingRegion(0, 0, W, H), target, None,
                  CompositeParams(color_match=False))
    assert res.expansion_hint is not None
    assert res.expansion_hint.any()
    quiet = compose(base, base.copy(), WorkingRegion(0, 0, W, H), target, None,
                    CompositeParams(color_match=False))
    assert quiet.expansion_hint is None
    assert quiet.drift == 0.0


def test_color_match_removes_global_shift():
    H, W = 400, 400
    base = rng_image(H, W, 4)
    target = disc(H, W, 200, 200, 60)
    candidate = np.clip(base.astype(np.int16) + 30, 0, 255).astype(np.uint8)
    candidate = np.clip(base.astype(np.int16) + 10, 0, 255).astype(np.uint8)  # small global shift: "quiet" pixels
    without = compose(base, candidate, WorkingRegion(0, 0, W, H), target, None,
                      CompositeParams(color_match=False))
    withm = compose(base, candidate, WorkingRegion(0, 0, W, H), target, None,
                    CompositeParams(color_match=True))
    assert without.drift > 5  # drift is measured on the raw candidate, same for both
    assert withm.color_matched
    # In the band, the matched output is closer to the base than the unmatched one.
    dist = ndimage.distance_transform_edt(~withm.repaint)
    band = (dist > 0) & (dist <= withm.feather)
    err_with = np.abs(withm.out[band].astype(int) - base[band].astype(int)).mean()
    err_without = np.abs(without.out[band].astype(int) - base[band].astype(int)).mean()
    assert err_with < err_without


def test_feather_and_grow_rules():
    assert feather_px(rect(1000, 1000, 100, 400, 100, 500), 8) == 8
    assert feather_px(rect(3000, 4000, 10, 20, 10, 20), 8) == 16  # 0.4 percent of 4000
    assert feather_px(rect(100, 100, 10, 20, 10, 20), 32) == 32
    assert grow_px(8) == 8 and grow_px(16) == 12


def test_invalid_inputs_raise():
    base = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compose(base, base, WorkingRegion(0, 0, 12, 10), np.zeros((10, 10), bool), None)
    with pytest.raises(ValueError):
        compose(base, base, WorkingRegion(0, 0, 10, 10), np.zeros((10, 10), np.uint8), None)


def test_color_match_ignores_deliberate_spill():
    H, W = 400, 400
    base = np.full((H, W, 3), 230, dtype=np.uint8)  # white truck
    target = rect(H, W, 100, 300, 100, 200)  # the door
    candidate = base.copy()
    candidate[80:320, 80:320] = (200, 110, 50)  # model painted door AND fender copper
    res = compose(base, candidate, WorkingRegion(0, 0, W, H), target, None, CompositeParams(color_match=True))
    core = target
    assert np.abs(res.out[core].astype(int) - np.array([200, 110, 50])).max() <= 1  # copper not pulled toward white


def test_context_blend_accepts_connected_spill_but_not_protected_or_far_changes():
    H, W = 500, 500
    base = np.full((H, W, 3), 120, dtype=np.uint8)
    target = rect(H, W, 150, 300, 150, 300)
    protected = rect(H, W, 300, 340, 150, 200)  # part of the shadow area, locked
    candidate = base.copy()
    candidate[150:300, 150:300] = 200  # object removed
    candidate[300:340, 140:320] = 60   # its shadow changed, connected to the target
    candidate[20:40, 20:40] = 250      # unrelated far change (within reach but not connected)
    tight = compose(base, candidate, WorkingRegion(0, 0, W, H), target, protected, CompositeParams(color_match=False, blend="tight"))
    ctx = compose(base, candidate, WorkingRegion(0, 0, W, H), target, protected, CompositeParams(color_match=False, blend="context"))
    shadow = rect(H, W, 322, 338, 230, 290)  # beyond the tight edge band
    assert (ctx.out[shadow] == 60).all()          # spill accepted
    assert (tight.out[shadow] == 120).all()       # tight keeps the base there
    assert np.array_equal(ctx.out[protected], base[protected])
    assert (ctx.out[22:38, 22:38] == 120).all()   # unconnected change rejected


def test_restore_detail_only_where_structure_is_kept():
    from PIL import Image as _I
    from lis.composite import restore_detail
    rng = np.random.default_rng(5)
    H, W = 256, 256
    base = np.clip(128 + 60 * rng.standard_normal((H, W, 3)), 0, 255).astype(np.uint8)
    base = ndimage.gaussian_filter(base.astype(np.float32), (1.0, 1.0, 0)).astype(np.uint8)  # textured
    scale = 0.5
    low = np.asarray(_I.fromarray(base).resize((128, 128), _I.LANCZOS).resize((256, 256), _I.LANCZOS)).astype(np.int16)
    recolour = np.clip(low + np.array([40, -20, -30]), 0, 255).astype(np.uint8)  # same structure, new colour
    flat = np.full((H, W, 3), 90, np.uint8)  # structure removed
    out1, w1 = restore_detail(base, recolour, scale)
    out2, w2 = restore_detail(base, flat, scale)
    lap = lambda a: ndimage.laplace(a.astype(np.float32).mean(axis=-1)).var()
    assert lap(out1) > 1.5 * lap(recolour)  # detail came back
    assert w1.mean() > 0.6 and w2.mean() < 0.05  # and only where the structure matched
    assert abs(out2.mean() - 90) < 1.0
    assert abs((out1.astype(float) - recolour).mean(axis=(0, 1))).max() < 3  # colour stays the candidate's


def test_expansion_hint_ignores_texture_speckle():
    H, W = 600, 600
    rng = np.random.default_rng(9)
    base = np.full((H, W, 3), 100, dtype=np.uint8)
    target = disc(H, W, 300, 300, 80)
    candidate = base.copy()
    speck = rng.random((H, W)) < 0.02  # isolated re-rendered pixels everywhere
    candidate[speck] = 200
    res = compose(base, candidate, WorkingRegion(0, 0, W, H), target, None, CompositeParams(color_match=False))
    assert res.expansion_hint is None


def test_parallel_filters_are_bit_identical(monkeypatch):
    """The strip-parallel blur and distance threshold give SciPy's exact values."""
    from scipy import ndimage

    from lis import composite as C

    monkeypatch.setattr(C, "_PAR_MIN", 1000)
    rng = np.random.default_rng(5)
    x = rng.random((700, 300)).astype(np.float32)
    for sg in (1.5, 6.0, 40.0):
        for mode in ("reflect", "nearest"):
            assert np.array_equal(C.gauss(x, sg, mode), ndimage.gaussian_filter(x, sg, mode=mode))
    x3 = rng.random((500, 200, 3)).astype(np.float32)
    assert np.array_equal(C.gauss(x3, (3.0, 3.0, 0)), ndimage.gaussian_filter(x3, (3.0, 3.0, 0)))
    m = rng.random((700, 300)) > 0.002
    for r in (3, 25, 120):
        assert np.array_equal(C.within(m, r), ndimage.distance_transform_edt(m) <= r)
    assert not C.within(np.ones((700, 300), bool), 10).any()


def test_within_without_any_feature_pixel_is_all_false(monkeypatch):
    from lis import composite as C

    for par_min in (10**9, 1000):
        monkeypatch.setattr(C, "_PAR_MIN", par_min)
        assert not C.within(np.ones((700, 300), bool), 10).any()
