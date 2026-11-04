"""Shadows: found next to the object, carried by a move as a dimming of the new ground."""
import types

import numpy as np

from lis.engine import Engine
from lis.shadow import find_shadow, shadow_factor


def scene(seed=0):
    rng = np.random.default_rng(seed)
    H, W = 240, 320
    floor = np.clip(np.full((H, W, 3), (200, 190, 170), np.float32) + rng.normal(0, 4, (H, W, 1)), 0, 255)
    obj = np.zeros((H, W), bool); obj[80:150, 100:150] = True
    sh = np.zeros((H, W), bool); sh[130:150, 150:215] = True  # cast to the right along the ground
    img = floor.copy(); img[sh] *= 0.5
    img[obj] = (180, 30, 30)
    return img.astype(np.uint8), floor.astype(np.uint8), obj, sh


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def test_dark_patch_next_to_the_object_is_its_shadow():
    img, _, obj, sh = scene()
    got, how = find_shadow(img, obj, None)
    assert how == "dark" and iou(got, sh) > 0.8 and not (got & obj).any()


def test_no_shadow_on_an_even_floor():
    _, floor, obj, _ = scene()
    img = floor.copy(); img[obj] = (180, 30, 30)
    got, how = find_shadow(img, obj, None)
    assert how == "none" and not got.any()


def test_a_dark_object_far_away_is_not_a_shadow():
    img, floor, obj, _ = scene()
    img = floor.copy(); img[obj] = (180, 30, 30)
    img[20:50, 250:300] = (img[20:50, 250:300] * 0.4).astype(np.uint8)
    got, how = find_shadow(img, obj, None)
    assert how == "none"


def test_move_carries_the_shadow():
    img, floor, obj, sh = scene()
    comp = types.SimpleNamespace(out=floor.copy(), alpha=(obj | sh).astype(np.float32), expansion_hint=None)
    protected = np.zeros_like(obj)
    msg = Engine._place_moved(img, comp, obj, protected, {"offset": [0, 60], "shadow": True}, sh)
    assert "shadow moved" in msg
    out = comp.out.astype(np.float32)
    new_sh = np.roll(sh, 60, axis=0)
    inner = np.zeros_like(sh); inner[192:208, 154:211] = True  # inside the moved shadow, away from its soft edge
    ratio = out[inner] / floor[inner].astype(np.float32)
    assert abs(float(ratio.mean()) - 0.5) < 0.05
    from scipy import ndimage
    old = sh & ~ndimage.binary_dilation(np.roll(obj, 60, axis=0), iterations=4)  # away from the moved object's soft edge
    assert np.array_equal(comp.out[old], floor[old])  # the old shadow is gone (the model's clean ground)
    assert ((comp.alpha > 0) | ~new_sh).all()  # where the shadow lands is inside the allowed area
    core = ndimage.binary_erosion(obj, iterations=3)
    assert np.array_equal(comp.out[np.roll(core, 60, axis=0)], img[core])  # the object itself, exact inside its soft edge


def test_kept_area_is_not_darkened():
    img, floor, obj, sh = scene()
    comp = types.SimpleNamespace(out=floor.copy(), alpha=(obj | sh).astype(np.float32), expansion_hint=None)
    protected = np.zeros_like(obj); protected[185:215, 140:230] = True
    Engine._place_moved(img, comp, obj, protected, {"offset": [0, 60]}, sh)
    assert np.array_equal(comp.out[protected], img[protected])


def test_factor_none_when_the_removal_left_the_shadow():
    img, _, obj, sh = scene()
    assert shadow_factor(img, img, sh) is None


def test_removal_that_left_the_object_is_noticed():
    from lis.engine import still_there

    img, floor, obj, _ = scene()
    assert still_there(img, img, obj)
    assert not still_there(img, floor, obj)
