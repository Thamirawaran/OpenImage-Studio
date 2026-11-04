"""The editor's HTTP API, driven end to end with the fake segmenter, grounder and worker."""
import base64
import io
import json
import sys
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from lis.engine import Engine, EngineConfig
from lis.render import WorkerConfig
from lis.web import Studio, create_app


def make_client(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"profiles": {"fake": {"files": {}}}}))
    projects = tmp_path / "projects"
    table = {"car": [[100, 100, 500, 400, 0.9]], "sign": [[700, 100, 900, 400, 0.8]]}

    def factory():
        return Engine(EngineConfig(projects_dir=projects, manifest_path=manifest, worker=WorkerConfig(module="lis.worker_fake", python=sys.executable),
                                   segmenter="fake", profile_id="fake", color_match=False, grounder="fake", grounder_table=table))

    studio = Studio(factory, projects)
    studio.start(background=False)
    return TestClient(create_app(studio)), studio


def png_bytes(w=1000, h=600):
    img = np.random.default_rng(1).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue(), img


def wait_job(c, jid):
    for _ in range(300):
        r = c.get(f"/api/jobs/{jid}").json()
        if r["done"]:
            return r
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_full_flow(tmp_path):
    c, studio = make_client(tmp_path)
    assert c.get("/api/status").json()["state"] == "ready"
    assert "OpenImage Studio" in c.get("/").text
    data, img = png_bytes()
    p = c.post("/api/projects?name=my photo.png", content=data).json()
    pid = p["project_id"]
    assert p["name"] == "my photo.png" and p["width"] == 1000 and p["objects"] == [] and p["head"] is None
    assert c.get(p["image_url"]).status_code == 200 and c.get(p["png_url"]).content[:4] == b"\x89PNG"
    assert c.get(f"/api/projects/{pid}/thumb/source.jpg").status_code == 200

    # words: target and keep proposed, nothing kept yet
    pr = c.post(f"/api/projects/{pid}/propose", json={"instruction": "make the car blue, keep the sign unchanged"}).json()
    assert pr["target"] and pr["target"]["name"] == "car" and pr["protect"][0]["candidate_id"]
    assert pr["fields"] == {"target": "car", "protect": "sign"} and "sign" not in pr["edit"]
    assert c.get(f"/api/projects/{pid}").json()["objects"] == []
    items = [{"candidate_id": pr["protect"][0]["candidate_id"], "kind": "protected", "name": "sign"},
             {"candidate_id": pr["target"]["candidate_id"], "kind": "target", "name": "car"}]
    p = c.post(f"/api/projects/{pid}/objects", json={"items": items, "replace": True}).json()
    assert [(o["name"], o["kind"]) for o in p["objects"]] == [("sign", "protected"), ("car", "target")]
    assert c.get(p["objects"][0]["url"]).status_code == 200

    # render through a job
    j = c.post(f"/api/projects/{pid}/render", json={"instruction": "make the car blue, keep the sign unchanged", "seed": 7, "steps": 2}).json()
    r = wait_job(c, j["job_id"])
    assert r["status"] == "ok", r
    rev = r["revision"]
    assert rev["checks"]["rows"][0]["status"] == "pass" and rev["checks"]["rows"][1]["key"] == "outside"
    assert r["project"]["head"] == rev["revision_id"]
    out = np.asarray(Image.open(io.BytesIO(c.get(rev["png_url"]).content)).convert("RGB"))
    sign = studio.engine._project(pid).mask(p["objects"][0]["mask_id"])
    assert np.array_equal(out[sign], img[sign])
    assert c.get(rev["change_map_url"]).status_code == 200
    # the proof is on the protected area as it was at this render: growing the keep afterwards does not break export
    c.post(f"/api/projects/{pid}/objects/{p['objects'][0]['mask_id']}/grow", json={"px": 260})  # now reaches into the edited car
    assert c.get(f"/api/projects/{pid}/export?fmt=png").headers["x-check"] == "pass"
    eng, pj = studio.engine, studio.engine._project(pid)
    stale = eng._file_check(pj, pj.base_path(rev["revision_id"]), pj.base_path(None), [p["objects"][0]["mask_id"]])
    assert stale["status"] == "fail"  # what export did before the snapshot: the grown mask made a good version unexportable

    # another seed replaces the step from the same parent
    j2 = c.post(f"/api/projects/{pid}/render", json={"instruction": "make the car blue", "seed": 8, "steps": 2, "retry_of": rev["revision_id"]}).json()
    r2 = wait_job(c, j2["job_id"])
    assert r2["revision"]["parent"] is None

    # several versions in one job, all from the same parent; the first is shown
    j3 = c.post(f"/api/projects/{pid}/render", json={"instruction": "make the car blue", "steps": 2, "retry_of": r2["revision"]["revision_id"],
                                                     "seeds": [11, 12, 13]}).json()
    r3 = wait_job(c, j3["job_id"])
    assert r3["status"] == "ok" and len(r3["variants"]) == 3
    assert {v["parent"] for v in r3["variants"]} == {None} and [v["seed"] for v in r3["variants"]] == [11, 12, 13]
    assert r3["project"]["head"] == r3["variants"][0]["revision_id"]
    assert all(v["checks"]["rows"][0]["status"] == "pass" for v in r3["variants"])
    assert c.post(f"/api/projects/{pid}/head", json={"revision_id": r2["revision"]["revision_id"]}).json()["head"] == r2["revision"]["revision_id"]

    # export is checked, undo and redo move the head
    e = c.get(f"/api/projects/{pid}/export?fmt=png")
    assert e.status_code == 200 and e.headers["x-check"] == "pass" and e.content[:4] == b"\x89PNG"
    assert c.post(f"/api/projects/{pid}/undo").json()["head"] is None
    assert c.post(f"/api/projects/{pid}/redo").json()["head"] == r2["revision"]["revision_id"]

    # samples open with their sentence
    sm = c.get("/api/samples").json()
    assert sm and c.get(sm[0]["thumb_url"]).status_code == 200
    ps = c.post(f"/api/samples/{sm[0]['id']}").json()
    assert ps["instruction"] == sm[0]["say"] and ps["project_id"] != pid
    assert c.post(f"/api/samples/{sm[0]['id']}").json()["project_id"] == ps["project_id"]  # an untouched copy is reused
    c.delete(f"/api/projects/{ps['project_id']}")

    # selection from words survives a reload; recent lists the project; delete removes it
    assert c.get(f"/api/projects/{pid}").json()["instruction"].startswith("make the car blue")
    assert c.get("/api/projects").json()[0]["project_id"] == pid
    assert c.delete(f"/api/projects/{pid}").json()["ok"]
    assert c.get("/api/projects").json() == []
    studio.engine.close()


def test_clicks_brush_switch_and_errors(tmp_path):
    c, studio = make_client(tmp_path)
    data, img = png_bytes()
    pid = c.post("/api/projects?name=a.png", content=data).json()["project_id"]
    assert c.post("/api/projects?name=a.txt", content=b"hello").status_code == 415
    assert c.post(f"/api/projects/{pid}/render", json={"instruction": "x"}).status_code == 400  # nothing selected
    s = c.post(f"/api/projects/{pid}/select", json={"object_id": "o1", "points": [[300, 300, 1]], "box": None}).json()
    assert s["candidates"]
    p = c.post(f"/api/projects/{pid}/objects", json={"items": [{"object_id": "o1", "candidate_id": s["candidates"][0]["candidate_id"], "kind": "target", "name": "jacket"}]}).json()
    mid = p["objects"][0]["mask_id"]
    a0 = p["objects"][0]["area"]
    stroke = np.zeros((600, 1000, 4), np.uint8)
    stroke[10:60, 10:60] = (255, 138, 31, 255)
    buf = io.BytesIO()
    Image.fromarray(stroke).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    p = c.post(f"/api/projects/{pid}/objects/{mid}/brush", json={"op": "add", "png": url}).json()
    assert p["objects"][0]["area"] == a0 + 2500
    p = c.patch(f"/api/projects/{pid}/objects/{mid}", json={"kind": "protected"}).json()
    assert p["objects"][0]["kind"] == "protected" and p["objects"][0]["mask_id"] != mid
    p = c.patch(f"/api/projects/{pid}/objects/{p['objects'][0]['mask_id']}", json={"name": "the jacket"}).json()
    assert p["objects"][0]["name"] == "the jacket"
    gm = p["objects"][0]["mask_id"]
    a1 = p["objects"][0]["area"]
    p = c.post(f"/api/projects/{pid}/objects/{gm}/grow", json={"px": 5}).json()
    assert p["objects"][0]["area"] > a1
    p = c.post(f"/api/projects/{pid}/objects/{gm}/grow", json={"px": -10}).json()
    assert p["objects"][0]["area"] < a1
    # a painted area from nothing
    p = c.post(f"/api/projects/{pid}/objects", json={"items": [{"candidate_id": None, "kind": "target", "name": "painted", "brush_png": url}]}).json()
    assert p["objects"][-1]["area"] == 2500
    f = c.post(f"/api/projects/{pid}/find", json={"object_id": "o2", "text": "car"}).json()
    assert f["candidates"] and f["candidates"][0]["box"] == [100, 100, 500, 400]
    assert c.get("/api/projects/../etc/image/source.png").status_code in (400, 404)
    assert c.get(f"/api/projects/{pid}/mask/masks/..%2Fproject.png").status_code in (400, 404)
    # Using a proposal replaces the selection it was made over, not a keep added while it was shown.
    before = [o["mask_id"] for o in c.get(f"/api/projects/{pid}").json()["objects"]]
    p = c.post(f"/api/projects/{pid}/objects", json={"items": [{"candidate_id": None, "kind": "protected", "name": "added keep", "brush_png": url}]}).json()
    p = c.post(f"/api/projects/{pid}/objects", json={"items": [{"candidate_id": None, "kind": "target", "name": "proposed", "brush_png": url}],
                                                    "replace_ids": before}).json()
    assert [o["name"] for o in p["objects"]] == ["added keep", "proposed"]
    # A candidate no longer in memory (a restart, a long session) is read back from its PNG.
    f = c.post(f"/api/projects/{pid}/find", json={"object_id": "o3", "text": "car"}).json()
    studio.engine.candidates.clear()
    p = c.post(f"/api/projects/{pid}/objects", json={"items": [{"candidate_id": f["candidates"][0]["candidate_id"], "kind": "target", "name": "car again"}]}).json()
    assert p["objects"][-1]["name"] == "car again" and p["objects"][-1]["area"] > 0
    assert c.delete(f"/api/projects/{pid}/objects").json()["objects"] == []
    studio.engine.close()


def test_real_world_photos(tmp_path):
    """What people actually open: phone photos turned by EXIF, iPhone HEIC in Display P3,
    transparent PNGs, 16-bit TIFFs, huge and tiny images, files that are not images."""
    from PIL import ImageCms

    c, studio = make_client(tmp_path)

    def up(data, name):
        return c.post(f"/api/projects?name={name}", content=data)

    # EXIF orientation 6 (camera held upright): opened 600 wide, 1000 tall
    im = Image.fromarray(np.full((600, 1000, 3), 120, np.uint8))
    exif = Image.Exif(); exif[0x0112] = 6
    buf = io.BytesIO(); im.save(buf, format="JPEG", exif=exif.tobytes())
    p = up(buf.getvalue(), "phone.jpg").json()
    assert (p["width"], p["height"]) == (600, 1000)
    # transparent PNG: the clear part becomes white, not black
    a = np.zeros((200, 300, 4), np.uint8); a[..., 0] = 200; a[:, :150, 3] = 255
    buf = io.BytesIO(); Image.fromarray(a, "RGBA").save(buf, format="PNG")
    p = up(buf.getvalue(), "logo.png").json()
    src = np.asarray(Image.open(studio.engine._project(p["project_id"]).root / "source.png"))
    assert tuple(src[10, 10]) == (200, 0, 0) and tuple(src[10, 250]) == (255, 255, 255)
    # 16-bit grey TIFF: scaled, not clipped to white
    g = (np.linspace(0, 65535, 300 * 200).reshape(200, 300)).astype(np.uint16)
    buf = io.BytesIO(); Image.fromarray(g).save(buf, format="TIFF")
    p = up(buf.getvalue(), "scan.tif").json()
    src = np.asarray(Image.open(studio.engine._project(p["project_id"]).root / "source.png"))
    assert src[0, 0, 0] == 0 and src[-1, -1, 0] == 255 and 100 < src[100, 150, 0] < 160
    # a colour profile other than sRGB is converted, and the user is told
    rgb_icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    buf = io.BytesIO(); Image.fromarray(np.full((100, 100, 3), 90, np.uint8)).save(buf, format="PNG", icc_profile=rgb_icc)
    assert up(buf.getvalue(), "srgb.png").json()["import_notes"] == []
    # HEIC, when pillow-heif is installed
    try:
        import pillow_heif  # noqa: F401
        buf = io.BytesIO(); Image.fromarray(np.full((128, 160, 3), 70, np.uint8)).save(buf, format="HEIF")
        p = up(buf.getvalue(), "IMG_0001.HEIC").json()
        assert (p["width"], p["height"]) == (160, 128)
    except ImportError:
        pass
    # tiny, broken, unsupported
    buf = io.BytesIO(); Image.fromarray(np.zeros((12, 12, 3), np.uint8)).save(buf, format="PNG")
    r = up(buf.getvalue(), "icon.png"); assert r.status_code == 415 and "16" in r.json()["detail"]
    r = up(b"not an image at all", "notes.jpg"); assert r.status_code == 415
    r = up(b"x", "doc.pdf"); assert r.status_code == 415
    studio.engine.close()


def test_huge_photo_scaled(tmp_path, monkeypatch):
    import lis.project as proj

    monkeypatch.setattr(proj, "MAX_PIXELS", 200_000)
    c, studio = make_client(tmp_path)
    data, _ = png_bytes(1000, 600)
    p = c.post("/api/projects?name=big.png", content=data).json()
    assert p["width"] * p["height"] <= 200_000 and p["import_notes"] and "opened at" in p["import_notes"][0]
    studio.engine.close()


def test_only_this_machine_or_the_key(tmp_path):
    """A page on another site (DNS rebinding) must not read the photos; with --lan a device
    needs the key once."""
    c, studio = make_client(tmp_path)
    assert c.get("/api/projects", headers={"host": "evil.example:7860"}).status_code == 403
    assert c.get("/api/projects", headers={"host": "127.0.0.1:9000"}).status_code == 200
    assert c.get("/api/projects", headers={"host": "localhost:7860"}).status_code == 200
    lan = TestClient(create_app(studio, "k3y"))
    assert lan.get("/api/projects", headers={"host": "192.168.1.5:7860"}).status_code == 403
    r = lan.get("/?key=k3y", headers={"host": "192.168.1.5:7860"})
    assert r.status_code == 200 and "lis_key" in r.headers.get("set-cookie", "")
    assert lan.get("/api/projects", headers={"host": "192.168.1.5:7860"}).status_code == 200  # the cookie
    assert TestClient(create_app(studio, "k3y")).get("/?key=wrong", headers={"host": "192.168.1.5:7860"}).status_code == 403
    studio.engine.close()


def test_click_waits_for_background_prefetch(tmp_path):
    """A slow background feature computation must not turn the first click into a 409."""
    c, studio = make_client(tmp_path)
    seg = studio.engine.segmenter
    orig = seg.set_image

    def slow(img):
        time.sleep(1.5)
        return orig(img)

    seg.set_image = slow
    data, _ = png_bytes()
    pid = c.post("/api/projects?name=a.png", content=data).json()["project_id"]  # starts the prefetch
    time.sleep(0.1)
    r = c.post(f"/api/projects/{pid}/select", json={"object_id": "o1", "points": [[300, 250, 1]]})
    assert r.status_code == 200 and r.json()["candidates"]


def test_find_falls_back_to_the_whole_photo_and_adds_get_room(tmp_path):
    """A named thing that is not found, or a sentence that names no part, used to give
    nothing to edit. Now the whole photo is proposed (kept areas still excluded), with a note."""
    c, studio = make_client(tmp_path)
    data, img = png_bytes()
    pid = c.post("/api/projects?name=w.png", content=data).json()["project_id"]
    pr = c.post(f"/api/projects/{pid}/propose", json={"instruction": "make the zebra blue, keep the sign unchanged"}).json()
    assert pr["whole"] and pr["target"]["candidate_id"] and any("could not find" in n for n in pr["notes"])
    m = np.asarray(Image.open(io.BytesIO(c.get(pr["target"]["url"]).content)).convert("L")) > 127
    assert m.mean() > 0.8 and not m[150:350, 750:850].any()  # everything but the kept sign
    pr = c.post(f"/api/projects/{pid}/propose", json={"instruction": "I want to see this in a winter scene"}).json()
    assert pr["whole"] and pr["target"]["candidate_id"]
    pr = c.post(f"/api/projects/{pid}/propose", json={"instruction": "add a dog next to the car"}).json()
    m = np.asarray(Image.open(io.BytesIO(c.get(pr["target"]["url"]).content)).convert("L")) > 127
    ys, xs = np.nonzero(m)
    assert xs.min() < 50 and xs.max() > 600  # room beside the car (x 100 to 500), not just the car
