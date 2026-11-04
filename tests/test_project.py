import json
import numpy as np
import pytest
from PIL import Image

from lis.project import Project, decode_canonical, sha256_file


def make_jpeg_with_rotation_exif(path, w=120, h=80):
    """A w x h image stored with EXIF orientation 6 (rotate 90 CW on display)."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] = (255, 0, 0)  # left half red, right half blue
    arr[:, w // 2 :] = (0, 0, 255)
    im = Image.fromarray(arr)
    exif = im.getexif()
    exif[0x0112] = 6
    im.save(path, format="JPEG", quality=100, exif=exif.tobytes())


def test_create_applies_exif_orientation_once_and_records_sha(tmp_path):
    src = tmp_path / "photo.jpg"
    make_jpeg_with_rotation_exif(src)
    p = Project.create(tmp_path / "projects", src)
    # Orientation 6 turns a 120x80 landscape into an 80x120 portrait.
    assert p.shape == (120, 80)
    s = p.source()
    assert s.shape == (120, 80, 3) and s.dtype == np.uint8
    assert not s.flags.writeable
    assert p.record.source_sha256 == sha256_file(p.root / "source.png")
    assert (p.root / "original.jpg").exists()
    # Re-decoding source.png gives the same pixels: no second rotation.
    again = decode_canonical(p.root / "source.png")
    assert np.array_equal(again, s)
    # Red (originally left) is now at the top after a clockwise rotation.
    assert tuple(s[5, 40]) == (254, 0, 0) or tuple(s[5, 40]) == (255, 0, 0)


def test_alpha_is_flattened_on_white_and_png_source_is_lossless(tmp_path):
    rgba = np.random.default_rng(0).integers(0, 256, size=(30, 40, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[:5, :5, 3] = 0  # a transparent corner shows as white, as in an image viewer
    src = tmp_path / "in.png"
    Image.fromarray(rgba).save(src)
    p = Project.create(tmp_path / "projects", src)
    assert np.array_equal(p.source()[5:, 5:], rgba[5:, 5:, :3])
    assert (p.source()[:5, :5] == 255).all()


def test_masks_round_trip_and_union(tmp_path):
    src = tmp_path / "in.png"
    Image.fromarray(np.zeros((50, 60, 3), dtype=np.uint8)).save(src)
    p = Project.create(tmp_path / "projects", src)
    a = np.zeros((50, 60), bool); a[10:20, 10:20] = True
    b = np.zeros((50, 60), bool); b[15:25, 15:25] = True
    ra = p.add_mask(a, "target", "jacket", {"clicks": [[12, 12, 1]]})
    rb = p.add_mask(b, "protected", "face")
    assert np.array_equal(p.mask(ra.mask_id), a)
    assert np.array_equal(p.mask(rb.mask_id), b)
    assert np.array_equal(p.union([ra.mask_id, rb.mask_id]), a | b)
    reopened = Project(p.root)
    assert [m.kind for m in reopened.record.masks] == ["target", "protected"]
    assert reopened.mask_record(ra.mask_id).prompts == {"clicks": [[12, 12, 1]]}
    with pytest.raises(KeyError):
        p.mask("m-nope")
    with pytest.raises(ValueError):
        p.add_mask(np.zeros((10, 10), bool), "target", "bad shape")
    with pytest.raises(ValueError):
        p.add_mask(a, "locked", "bad kind")


def test_base_is_source_when_no_head(tmp_path):
    src = tmp_path / "in.png"
    img = np.random.default_rng(1).integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    Image.fromarray(img).save(src)
    p = Project.create(tmp_path / "projects", src)
    assert p.base_path(None) == p.root / "source.png"
    assert np.array_equal(p.base(None), img)
    assert p.base_path("r-abc") == p.root / "runs" / "r-abc" / "out.png"
    data = json.loads((p.root / "project.json").read_text())
    assert data["head"] is None and data["schema_version"] == 1
