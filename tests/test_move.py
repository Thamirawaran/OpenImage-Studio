"""Moving an object: the model fills its old place, its own pixels land at the offset, and the
usual proofs hold on the saved files."""
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from test_web import make_client, png_bytes, wait_job  # noqa: E402


def _setup(c):
    data, img = png_bytes()
    p = c.post("/api/projects?name=m.png", content=data).json()
    pid = p["project_id"]
    pr = c.post(f"/api/projects/{pid}/propose", json={"instruction": "make the car blue, keep the sign unchanged"}).json()
    items = [{"candidate_id": pr["target"]["candidate_id"], "kind": "target", "name": "car"},
             {"candidate_id": pr["protect"][0]["candidate_id"], "kind": "protected", "name": "sign"}]
    c.post(f"/api/projects/{pid}/objects", json={"items": items, "replace": True})
    return pid, img


def test_move_object(tmp_path):
    c, studio = make_client(tmp_path)
    pid, img = _setup(c)
    assert c.post(f"/api/projects/{pid}/render", json={"move": {"dx": 0, "dy": 0}}).status_code == 400
    assert c.post(f"/api/projects/{pid}/render", json={"move": {"dx": "a"}}).status_code == 400
    j = wait_job(c, c.post(f"/api/projects/{pid}/render", json={"move": {"dx": 60, "dy": -30}}).json()["job_id"])
    assert j["status"] == "ok", j
    rev = j["revision"]
    assert rev["kind"] == "move" and rev["instruction"] == "Move 60 px right, 30 px up"
    rows = {r["key"]: r for r in rev["checks"]["rows"]}
    assert rows["protected"]["status"] == "pass" and rows["outside"]["status"] == "pass"
    assert "Moved" in rows["edit"]["detail"]
    out = np.asarray(Image.open(io.BytesIO(c.get(rev["png_url"]).content)).convert("RGB"))
    p = studio.engine._project(pid)
    car = p.mask(p.record.masks[0].mask_id) if p.record.masks[0].kind == "target" else p.mask(p.record.masks[1].mask_id)
    sign = p.union([m.mask_id for m in p.record.masks if m.kind == "protected"])
    # the object's own pixels, eroded away from its soft edge, are at the offset (where not kept)
    from scipy import ndimage
    core = ndimage.binary_erosion(car, iterations=3)
    ys, xs = np.nonzero(core)
    ny, nx = ys - 30, xs + 60
    ok = (ny >= 0) & (nx < img.shape[1]) & ~sign[ny.clip(0), nx.clip(0, img.shape[1] - 1)]
    assert ok.sum() > 1000
    assert np.array_equal(out[ny[ok], nx[ok]], img[ys[ok], xs[ok]])
    assert np.array_equal(out[sign], img[sign])
    # another seed of a move moves by the same offset
    j2 = wait_job(c, c.post(f"/api/projects/{pid}/render", json={"retry_of": rev["revision_id"], "seed": 9}).json()["job_id"])
    assert j2["status"] == "ok" and j2["revision"]["instruction"] == "Move 60 px right, 30 px up"
    assert c.get(f"/api/projects/{pid}/export?fmt=png").status_code == 200


def test_move_plan_needs_an_offset(tmp_path):
    from lis.policy import resolve

    c, studio = make_client(tmp_path)
    pid, img = _setup(c)
    p = studio.engine._project(pid)
    tid = next(m.mask_id for m in p.record.masks if m.kind == "target")
    base = {"schema_version": 2, "project_id": pid, "parent_revision": p.record.head, "protected_mask_ids": [],
            "backend_profile_id": "fake", "network_policy": "offline"}
    for off in (None, [0, 0], [1.5, 2], [1], "3,4"):
        op = {"kind": "move", "target_mask_id": tid, "instruction": "remove the car", **({"offset": off} if off is not None else {})}
        assert not resolve({**base, "operation": op}, p, studio.engine.manifest).ok, off
    assert resolve({**base, "operation": {"kind": "move", "target_mask_id": tid, "instruction": "remove the car", "offset": [5, -3]}},
                   p, studio.engine.manifest).ok
