"""New pictures from words: made, shown, opened as a project, then edited like a photo."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from test_web import make_client, wait_job  # noqa: E402


def test_make_pictures_and_edit_one(tmp_path):
    c, studio = make_client(tmp_path)
    assert c.post("/api/generate", json={"prompt": "  "}).status_code == 400
    assert c.post("/api/generate", json={"prompt": "a fox", "size": "huge"}).status_code == 400
    r = c.post("/api/generate", json={"prompt": "a red fox in the snow", "size": "landscape", "count": 2, "seed": 7}).json()
    assert r["count"] == 2 and r["seed"] == 7
    assert c.post("/api/generate", json={"prompt": "another"}).status_code in (200, 409)  # busy while the first runs
    j = wait_job(c, r["job_id"])
    assert j["status"] == "ok" and [im["seed"] for im in j["images"]] == [7, 8]
    assert all((im["width"], im["height"]) == (1216, 832) for im in j["images"])
    assert c.get(j["images"][1]["url"]).headers["content-type"] == "image/jpeg"
    a = c.get(j["images"][0]["url"]).content
    b = c.get(j["images"][1]["url"]).content
    assert a != b  # one seed each

    p = c.post(f"/api/generated/{r['job_id']}/1/open").json()
    assert (p["width"], p["height"]) == (1216, 832) and p["head"] is None
    assert p["name"].startswith("a-red-fox-in-the-snow") and "seed 8" in p["import_notes"][0]
    # edited like any photo
    pid = p["project_id"]
    pr = c.post(f"/api/projects/{pid}/propose", json={"instruction": "make the car blue"}).json()
    items = [{"candidate_id": pr["target"]["candidate_id"], "kind": "target", "name": "car"}]
    c.post(f"/api/projects/{pid}/objects", json={"items": items, "replace": True})
    jr = wait_job(c, c.post(f"/api/projects/{pid}/render", json={"instruction": "make the car blue"}).json()["job_id"])
    assert jr["status"] == "ok"
    assert c.post(f"/api/generated/{r['job_id']}/9/open").status_code == 404


def test_cancel_making(tmp_path):
    c, studio = make_client(tmp_path)
    r = c.post("/api/generate", json={"prompt": "a lighthouse", "count": 4}).json()
    c.post(f"/api/jobs/{r['job_id']}/cancel")
    j = wait_job(c, r["job_id"])
    assert j["status"] in ("cancelled", "ok") and len(j["images"]) < 4


def test_extend_photo(tmp_path):
    import io

    import numpy as np
    from PIL import Image

    from test_web import png_bytes

    c, studio = make_client(tmp_path)
    data, img = png_bytes(400, 300)
    p = c.post("/api/projects?name=room.png", content=data).json()
    pid = p["project_id"]
    assert c.post(f"/api/projects/{pid}/extend", json={"shape": "huge"}).status_code == 400
    assert c.post(f"/api/projects/{pid}/extend", json={"shape": "4:3"}).status_code == 400  # already 4:3
    r = c.post(f"/api/projects/{pid}/extend", json={"shape": "wider", "count": 1, "seed": 3}).json()
    assert r["size"] == [600, 300]
    j = wait_job(c, r["job_id"])
    assert j["status"] == "ok" and (j["images"][0]["width"], j["images"][0]["height"]) == (600, 300)
    q = c.post(f"/api/generated/{r['job_id']}/0/open").json()
    assert (q["width"], q["height"]) == (600, 300) and q["name"] == "room-extended-1.png"
    assert "extended from room.png (100 px left, 100 px right)" in q["import_notes"][0]
    out = np.asarray(Image.open(io.BytesIO(c.get(q["png_url"]).content)).convert("RGB"))
    band = max(8, round(0.02 * 600))
    assert np.array_equal(out[:, 100 + band:500 - band], img[:, band:400 - band])  # the original, exact away from the fade
    assert not np.array_equal(out[:, :100], np.repeat(img[:, :1], 100, 1))  # something was made there
    sq = c.post(f"/api/projects/{pid}/extend", json={"shape": "square", "count": 1}).json()
    assert sq["size"] == [400, 400]
    wait_job(c, sq["job_id"])


def test_extend_never_fades_a_kept_area(tmp_path):
    import io

    import numpy as np
    from PIL import Image

    from test_web import png_bytes

    c, studio = make_client(tmp_path)
    data, img = png_bytes(400, 300)
    pid = c.post("/api/projects?name=k.png", content=data).json()["project_id"]
    cand = c.post(f"/api/projects/{pid}/select", json={"object_id": "k1", "box": [0, 0, 60, 300]}).json()["candidates"][0]
    c.post(f"/api/projects/{pid}/objects", json={"items": [{"object_id": "k1", "candidate_id": cand["candidate_id"], "kind": "protected", "name": "edge"}]})
    p = studio.engine._project(pid)
    kept = p.union([m.mask_id for m in p.record.masks if m.kind == "protected"])
    assert kept[:, :5].any()  # the kept area touches the left edge, inside the fade band
    r = c.post(f"/api/projects/{pid}/extend", json={"shape": "wider", "count": 1}).json()
    assert wait_job(c, r["job_id"])["status"] == "ok"
    q = c.post(f"/api/generated/{r['job_id']}/0/open").json()
    out = np.asarray(Image.open(io.BytesIO(c.get(q["png_url"]).content)).convert("RGB"))
    inner = out[:, 100:500]
    bad = np.any(inner != img, axis=-1) & kept
    assert not bad.any(), (int(bad.sum()), np.argwhere(bad)[:5].tolist(), studio.ui_state(pid)["objects"])


def test_pictures_from_this_photo_and_another(tmp_path):
    import io

    import numpy as np
    from PIL import Image

    from test_web import png_bytes

    c, studio = make_client(tmp_path)
    data, img = png_bytes(640, 960)
    pid = c.post("/api/projects?name=her.png", content=data).json()["project_id"]
    assert c.post("/api/generate", json={"prompt": "x", "size": "photo"}).status_code == 400  # same shape needs a photo
    buf = io.BytesIO(); Image.new("RGB", (500, 300), (10, 200, 30)).save(buf, format="PNG")
    ref = c.post("/api/refs?name=other.png", content=buf.getvalue()).json()
    assert (ref["width"], ref["height"]) == (500, 300)
    r = c.post("/api/generate", json={"prompt": "the same woman sitting on a bench", "size": "photo", "count": 1, "ref_project": pid,
                                      "ref_uploads": [ref["ref_id"]]}).json()
    j = wait_job(c, r["job_id"])
    assert j["status"] == "ok"
    w, h = j["images"][0]["width"], j["images"][0]["height"]
    assert w % 16 == 0 and h % 16 == 0 and abs(w / h - 640 / 960) < 0.03 and w * h <= 1024 * 1024
    q = c.post(f"/api/generated/{r['job_id']}/0/open").json()
    assert "from two reference photos" in q["import_notes"][0]
    assert c.post("/api/generate", json={"prompt": "x", "ref_uploads": ["nope"]}).status_code == 404


def test_pose_sentences_redraw_the_whole_picture():
    from lis.ground import parse_instruction as P

    for t in ["make her sit on a cafe chair", "change her pose", "she should be sitting on the steps", "make him wave at the camera",
              "keep the car unchanged and make her stand with her arms crossed"]:
        r = P(t)
        assert r.whole and r.redraw and not r.targets, t
    assert [p.text for p in P("keep the car unchanged and make her stand with her arms crossed").protect] == ["car"]
    for t in ["remove the walking man", "make her dress red", "the standing desk should be white"]:
        assert not P(t).redraw, t
    for t, who, what in [("give her sunglasses", "woman", "sunglasses"), ("make him wear a black suit", "man", "a black suit"),
                         ("she should wear a red scarf", "woman", "a red scarf"), ("dress her in a leather jacket", "woman", "a leather jacket")]:
        r = P(t)
        assert [p.text for p in r.targets] == [who] and r.adds == what, t
