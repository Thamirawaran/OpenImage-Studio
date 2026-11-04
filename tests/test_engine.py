import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lis.engine import Engine, EngineConfig, working_region, worker_dims, crop_with_edge_pad
from lis.render import WorkerConfig
from lis import composite as C


def make_engine(tmp_path, color_match=False, **worker_extra):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"profiles": {"klein-4b-bf16": {"files": {"transformer/x.safetensors": {"sha256": "sha256:abc"}}}}, "last_verified": "t"}))
    cfg = EngineConfig(projects_dir=tmp_path / "projects", manifest_path=manifest,
                       worker=WorkerConfig(module="lis.worker_fake", weights="", python=sys.executable, **worker_extra),
                       segmenter="fake", color_match=color_match)
    return Engine(cfg)


def photo(tmp_path, h=480, w=640, seed=0):
    img = np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    p = tmp_path / "photo.png"
    Image.fromarray(img).save(p)
    return p, img


def plan(pid, head, target, protected=(), **over):
    d = {"schema_version": 2, "project_id": pid, "parent_revision": head,
         "operation": {"kind": "native_instruction_edit", "target_mask_id": target, "instruction": "make the jacket copper", "band_px": 32, "strategy": "auto"},
         "protected_mask_ids": list(protected), "seed": 7, "steps": 3, "strength": 0.8, "backend_profile_id": "klein-4b-bf16", "network_policy": "offline"}
    for k, v in over.items():
        (d["operation"] if k in d["operation"] else d)[k] = v
    return d


def run(gen):
    msgs = list(gen)
    assert msgs[-1]["type"] == "result"
    return [m for m in msgs if m["type"] == "progress"], msgs[-1]


def test_end_to_end_select_protect_render_export(tmp_path):
    eng = make_engine(tmp_path)
    src, img = photo(tmp_path)
    o = eng.open(src)
    pid = o["project_id"]
    sel = eng.select(pid, "obj1", [(200, 240, 1)], None)
    assert len(sel["candidates"]) == 2 and sel["candidates"][0]["score"] >= sel["candidates"][1]["score"]
    t = eng.commit_mask(pid, "obj1", sel["candidates"][0]["candidate_id"], "target", "jacket")["mask_id"]
    sel2 = eng.select(pid, "obj2", [(330, 240, 1)], None)  # overlaps the target disc
    pr = eng.commit_mask(pid, "obj2", sel2["candidates"][0]["candidate_id"], "protected", "face")["mask_id"]

    progress, result = run(eng.render(plan(pid, None, t, [pr])))
    assert result["status"] == "ok", result
    assert len(progress) == 3 and result["strategy_used"] == "edit"
    assert result["protected_check"] == {"status": "pass", "mismatched_pixels": 0}
    assert result["working_region"]["scale"] == 1.0 and "protected and will not change" in result["message"]
    out = np.asarray(Image.open(result["image_path"]).convert("RGB"))
    eng_p = eng._project(pid)
    prot = eng_p.mask(pr)
    assert np.array_equal(out[prot], img[prot])
    core = eng_p.mask(t) & ~prot
    assert not np.array_equal(out[core], img[core])  # the fake worker tinted it
    h = eng.history(pid)
    assert h["head"] == result["revision_id"] and h["revisions"][0]["is_head"]
    assert (eng_p.root / "runs" / result["revision_id"] / "meta.json").exists()

    # Second edit on top: base is now revision 1; protect the (tinted) jacket and edit elsewhere.
    sel3 = eng.select(pid, "obj3", [(100, 100, 1)], None)
    t2 = eng.commit_mask(pid, "obj3", sel3["candidates"][0]["candidate_id"], "target", "corner")["mask_id"]
    _, r2 = run(eng.render(plan(pid, h["head"], t2, [t], instruction="remove the stain")))
    assert r2["status"] == "ok" and r2["strategy_used"] == "edit"
    out2 = np.asarray(Image.open(r2["image_path"]).convert("RGB"))
    jacket = eng_p.mask(t)
    assert np.array_equal(out2[jacket], out[jacket])  # protected against the base, not the original

    # Stale plan is rejected; undo moves head; export re-checks.
    _, r3 = run(eng.render(plan(pid, None, t)))
    assert r3["status"] == "rejected" and "changed since" in r3["message"]
    assert eng.set_head(pid, result["revision_id"])["head"] == result["revision_id"]
    ex = eng.export(pid, result["revision_id"], tmp_path / "out.png", "png")
    assert ex["ok"] and ex["protected_check"]["status"] == "pass"
    exj = eng.export(pid, r2["revision_id"], tmp_path / "out.jpg", "jpeg")
    assert exj["ok"] and exj["format"] == "jpeg"
    eng.close()


def test_rejections_and_failed_runs_leave_no_revision(tmp_path):
    eng = make_engine(tmp_path)
    src, _ = photo(tmp_path)
    pid = eng.open(src)["project_id"]
    sel = eng.select(pid, "a", [(200, 240, 1)], None)
    t = eng.commit_mask(pid, "a", sel["candidates"][0]["candidate_id"], "target", "x")["mask_id"]
    big = eng.commit_mask(pid, "b", None, "protected", "all", brush_add_png=None)["mask_id"]  # empty protected mask
    _, r = run(eng.render(plan(pid, None, t, [big], instruction="")))
    assert r["status"] == "rejected" and "empty" in r["message"].lower()
    # Cancel mid-render: no revision, and the cancelled run is deleted (only real failures are kept for inspection).
    ev = threading.Event()
    gen = eng.render(plan(pid, None, t, steps=30), cancel=ev)
    first = next(gen)
    ev.set()
    rest = list(gen)
    assert rest[-1]["status"] == "cancelled"
    p = eng._project(pid)
    assert eng.history(pid)["head"] is None and not any(d.name != "failed" for d in (p.root / "runs").iterdir())
    eng.close()


def test_large_photo_uses_scaled_working_region(tmp_path):
    eng = make_engine(tmp_path)
    src, img = photo(tmp_path, h=1500, w=2000, seed=3)
    pid = eng.open(src)["project_id"]
    sel = eng.select(pid, "a", [(1000, 750, 1)], None)
    t = eng.commit_mask(pid, "a", sel["candidates"][0]["candidate_id"], "target", "x")["mask_id"]
    _, r = run(eng.render(plan(pid, None, t)))
    assert r["status"] == "ok"
    wr = r["working_region"]
    assert wr["worker_size"][0] % 16 == 0 and wr["worker_size"][1] % 16 == 0
    assert wr["worker_size"][0] * wr["worker_size"][1] <= 1024 * 1024
    assert wr["w"] >= 512 and wr["h"] >= 512
    assert wr["scale"] <= 1.0
    out = np.asarray(Image.open(r["image_path"]).convert("RGB"))
    outside = np.ones(img.shape[:2], bool)
    outside[wr["y"]:wr["y"] + wr["h"], wr["x"]:wr["x"] + wr["w"]] = False
    assert np.array_equal(out[outside], img[outside])
    eng.close()


def test_working_region_geometry_rules():
    H, W = 600, 800
    t = np.zeros((H, W), bool); t[280:320, 380:420] = True  # small centre target
    region, box = working_region((H, W), t, feather=32, grow=20)
    assert box[2] % 16 == 0 and box[3] % 16 == 0 and box[2] >= 512 and box[3] >= 512
    assert region.x >= 0 and region.y >= 0 and region.x + region.w <= W and region.y + region.h <= H
    # Target at the corner of an image whose size is not a multiple of 16: box may overflow, crop pads.
    H2, W2 = 300, 410
    t2 = np.zeros((H2, W2), bool); t2[0:30, 380:410] = True
    region2, box2 = working_region((H2, W2), t2, 32, 20)
    assert box2[0] == 0 and box2[1] == 0 and box2[2] >= W2 and box2[3] >= H2
    padded = crop_with_edge_pad(np.zeros((H2, W2, 3), np.uint8), box2)
    assert padded.shape == (box2[3], box2[2], 3)
    assert worker_dims(2048, 1536, 0.5773) == (1168, 880)


def test_commit_inverted_mask_targets_everything_else(tmp_path):
    eng = make_engine(tmp_path)
    src, _ = photo(tmp_path)
    pid = eng.open(src)["project_id"]
    sel = eng.select(pid, "person", [(320, 240, 1)], None)
    cid = sel["candidates"][0]["candidate_id"]
    m = eng.candidates[cid].mask
    bg = eng.commit_mask(pid, "person", cid, "target", "background", invert=True)["mask_id"]
    assert np.array_equal(eng._project(pid).mask(bg), ~m)
    eng.close()
