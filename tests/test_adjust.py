"""Adjust (no model) and the geometry operations, through the editor's HTTP API with the fake
engine, plus the formula itself."""
import io
import time

import numpy as np
from PIL import Image

from lis import adjust as A
from tests.test_web import make_client, png_bytes


def decode(content: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(content)).convert("RGB"))


def open_photo(c, w=1000, h=600):
    data, img = png_bytes(w, h)
    p = c.post("/api/projects?name=street.png", content=data).json()
    return p, img


def keep_and_edit(c, pid, keep=True, edit=False):
    """A kept 'sign' (700..900 x 100..400) and an edit 'car' (100..500 x 100..400) from the fake grounder."""
    pr = c.post(f"/api/projects/{pid}/propose", json={"instruction": "make the car blue, keep the sign unchanged"}).json()
    items = []
    if keep:
        items.append({"candidate_id": pr["protect"][0]["candidate_id"], "kind": "protected", "name": "sign"})
    if edit:
        items.append({"candidate_id": pr["target"]["candidate_id"], "kind": "target", "name": "car"})
    return c.post(f"/api/projects/{pid}/objects", json={"items": items, "replace": True}).json()


def test_formula_matches_the_docstring():
    v = {"brightness": 40, "contrast": -20, "saturation": 30, "warmth": 50}
    px = np.array([[[200, 100, 50], [0, 0, 0], [255, 255, 255]]], np.uint8)
    out = A.to_uint8(A.adjusted(px, v))
    # by hand, step by step, in float64
    x = px.astype(np.float64) / 255
    x = x ** (2 ** (-0.4))
    x = (x - 0.5) * 0.8 + 0.5
    L = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    x = L[..., None] + (x - L[..., None]) * 1.3
    x[..., 0] *= 1.1
    x[..., 2] *= 0.9
    ref = np.clip(np.floor(x * 255 + 0.5), 0, 255).astype(np.uint8)
    assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1
    assert A.describe(v) == "Adjust: brightness +40, contrast -20, saturation +30, warmth +50"
    assert A.describe({"brightness": 0, "contrast": 0, "saturation": 0, "warmth": -10}) == "Adjust: warmth -10"
    # brightness alone keeps black and white where they are
    bw = A.to_uint8(A.adjusted(np.array([[[0, 0, 0], [255, 255, 255]]], np.uint8), {"brightness": 100}))
    assert bw.tolist() == [[[0, 0, 0], [255, 255, 255]]]


def test_adjust_whole_photo_keeps_protected_exact(tmp_path):
    c, studio = make_client(tmp_path)
    p, img = open_photo(c)
    pid = p["project_id"]
    p = keep_and_edit(c, pid, keep=True, edit=False)
    sign = studio.engine._project(pid).mask(p["objects"][0]["mask_id"])
    r = c.post(f"/api/projects/{pid}/adjust", json={"brightness": 20, "warmth": 10, "parent": "source"})
    assert r.status_code == 200, r.text
    r = r.json()
    rev = r["revision"]
    assert rev["instruction"] == "Adjust: brightness +20, warmth +10" and rev["kind"] == "adjust" and rev["parent"] is None
    rows = {x["key"]: x for x in rev["checks"]["rows"]}
    assert rows["protected"]["status"] == "pass" and rows["outside"]["status"] == "pass"
    assert rows["edit"]["status"] == "info" and "No model" in rows["edit"]["title"]  # no edit check is claimed
    assert rev["checks"]["overall"] == "pass"
    assert r["project"]["head"] == rev["revision_id"] and len(r["project"]["revisions"]) == 1
    out = decode(c.get(rev["png_url"]).content)
    assert np.array_equal(out[sign], img[sign])  # kept pixels byte-identical, on the file served
    changed = np.any(out != img, axis=-1)
    assert changed[~sign].mean() > 0.9  # everything else changed
    # the revision records what was done, and the proof files exist
    pj = studio.engine._project(pid)
    meta = next(x for x in pj.record.revisions if x["revision_id"] == rev["revision_id"])
    assert meta["operation"]["kind"] == "adjust" and meta["operation"]["values"] == {"brightness": 20, "contrast": 0, "saturation": 0, "warmth": 10}
    assert meta["operation"]["area"] == "whole photo" and meta["edit_check"]["status"] == "not_checked"
    run = pj.root / "runs" / rev["revision_id"]
    assert (run / "protected.png").exists() and (run / "allowed.png").exists()
    for _ in range(100):  # the change map is drawn after the answer
        if (run / "change_map.png").exists():
            break
        time.sleep(0.05)
    assert c.get(rev["change_map_url"]).status_code == 200
    # export proof passes on the file it hands over; undo and redo work like any version
    e = c.get(f"/api/projects/{pid}/export?fmt=png")
    assert e.status_code == 200 and e.headers["x-check"] == "pass" and np.array_equal(decode(e.content)[sign], img[sign])
    assert c.post(f"/api/projects/{pid}/undo").json()["head"] is None
    assert c.post(f"/api/projects/{pid}/redo").json()["head"] == rev["revision_id"]
    # a second adjustment builds on the first
    r2 = c.post(f"/api/projects/{pid}/adjust", json={"contrast": 15}).json()
    assert r2["revision"]["parent"] == rev["revision_id"]
    out2 = decode(c.get(r2["revision"]["png_url"]).content)
    assert np.array_equal(out2[sign], img[sign])
    studio.engine.close()


def test_adjust_selection_only(tmp_path):
    c, studio = make_client(tmp_path)
    p, img = open_photo(c)
    pid = p["project_id"]
    p = keep_and_edit(c, pid, keep=True, edit=True)
    pj = studio.engine._project(pid)
    sign = pj.mask(next(o["mask_id"] for o in p["objects"] if o["kind"] == "protected"))
    car = pj.mask(next(o["mask_id"] for o in p["objects"] if o["kind"] == "target"))
    r = c.post(f"/api/projects/{pid}/adjust", json={"saturation": -100})
    assert r.status_code == 200, r.text
    rev = r.json()["revision"]
    out = decode(c.get(rev["png_url"]).content)
    assert np.array_equal(out[sign], img[sign])
    changed = np.any(out != img, axis=-1)
    assert changed[car].mean() > 0.95  # the selection is black and white
    o = out[car].astype(int)
    assert np.abs(o[:, 0] - o[:, 1]).max() <= 1 and np.abs(o[:, 1] - o[:, 2]).max() <= 1
    # only a few pixels of soft edge outside the selection changed, nothing further away
    from scipy import ndimage
    near = ndimage.binary_dilation(car, iterations=A.edge_px(pj.shape) * 2 + 2)
    assert not changed[~near].any() and changed[near & ~car].any()
    meta = next(x for x in pj.record.revisions if x["revision_id"] == rev["revision_id"])
    assert meta["operation"]["area"] == "edit area" and meta["operation"]["target_mask_ids"]
    assert c.get(f"/api/projects/{pid}/export?fmt=png").headers["x-check"] == "pass"
    studio.engine.close()


def test_adjust_refusals(tmp_path):
    c, studio = make_client(tmp_path)
    p, _ = open_photo(c)
    pid = p["project_id"]
    r = c.post(f"/api/projects/{pid}/adjust", json={"brightness": 0, "contrast": 0, "saturation": 0, "warmth": 0})
    assert r.status_code == 400 and "nothing would change" in r.json()["detail"]
    assert c.post(f"/api/projects/{pid}/adjust", json={"brightness": 101}).status_code == 400
    assert c.post(f"/api/projects/{pid}/adjust", json={"brightness": "a lot"}).status_code == 400
    assert c.post(f"/api/projects/{pid}/adjust", json={"brightness": 10, "parent": "r-elsewhere"}).status_code == 400  # not the image on screen
    assert c.get(f"/api/projects/{pid}").json()["revisions"] == []
    # refused while an edit runs on this photo
    from lis.web import Job
    job = Job(pid)
    studio.jobs[job.id] = job
    assert c.post(f"/api/projects/{pid}/adjust", json={"brightness": 10}).status_code == 409
    assert c.post(f"/api/projects/{pid}/transform", json={"op": "rotate_left"}).status_code == 409
    job.done = True
    assert c.post(f"/api/projects/{pid}/adjust", json={"brightness": 10}).status_code == 200
    studio.engine.close()


def test_rotate_flip_crop_make_new_projects(tmp_path):
    c, studio = make_client(tmp_path)
    p, img = open_photo(c, 1000, 600)
    pid = p["project_id"]
    a = c.post(f"/api/projects/{pid}/adjust", json={"brightness": 30}).json()  # geometry uses the image on screen
    shown = decode(c.get(a["revision"]["png_url"]).content)

    r = c.post(f"/api/projects/{pid}/transform", json={"op": "rotate_left"}).json()
    assert r["project_id"] != pid and (r["width"], r["height"]) == (600, 1000)
    assert r["import_notes"][0].startswith("Rotated left from street.png") and r["note"] == r["import_notes"][0]
    assert r["revisions"] == [] and r["objects"] == [] and r["head"] is None
    assert np.array_equal(decode(c.get(r["png_url"]).content), np.rot90(shown, 1))

    r = c.post(f"/api/projects/{pid}/transform", json={"op": "rotate_right"}).json()
    assert (r["width"], r["height"]) == (600, 1000) and r["note"].startswith("Rotated right from street.png")
    assert np.array_equal(decode(c.get(r["png_url"]).content), np.rot90(shown, -1))

    r = c.post(f"/api/projects/{pid}/transform", json={"op": "flip_horizontal"}).json()
    assert (r["width"], r["height"]) == (1000, 600) and r["note"].startswith("Flipped horizontally from street.png")
    assert np.array_equal(decode(c.get(r["png_url"]).content), shown[:, ::-1])

    r = c.post(f"/api/projects/{pid}/transform", json={"op": "crop", "box": [300, 120, 400, 300]}).json()
    assert (r["width"], r["height"]) == (400, 300)
    assert r["note"].startswith("Cropped from street.png, 400x300 at 300,120")
    assert np.array_equal(decode(c.get(r["png_url"]).content), shown[120:420, 300:700])

    # the old project is untouched
    old = c.get(f"/api/projects/{pid}").json()
    assert (old["width"], old["height"]) == (1000, 600) and len(old["revisions"]) == 1 and old["head"] == a["revision"]["revision_id"]
    assert len(c.get("/api/projects").json()) == 5
    studio.engine.close()


def test_crop_bounds_validated(tmp_path):
    c, studio = make_client(tmp_path)
    p, _ = open_photo(c, 1000, 600)
    pid = p["project_id"]
    before = len(c.get("/api/projects").json())
    for box in ([900, 0, 200, 100], [-1, 0, 100, 100], [0, 0, 10, 100], [0, 550, 100, 100], [0, 0, 1000, 600], None, [1, 2, 3], ["a", 0, 50, 50], [0.5, 0, 50, 50]):
        r = c.post(f"/api/projects/{pid}/transform", json={"op": "crop", "box": box})
        assert r.status_code == 400, (box, r.text)
    assert c.post(f"/api/projects/{pid}/transform", json={"op": "shear"}).status_code == 400
    assert len(c.get("/api/projects").json()) == before  # nothing was created
    assert not any((tmp_path / "projects" / ".uploads").glob("*/*.png"))  # and nothing was left behind
    ok = c.post(f"/api/projects/{pid}/transform", json={"op": "crop", "box": [0, 0, 1000, 599]})
    assert ok.status_code == 200 and ok.json()["height"] == 599
    studio.engine.close()
