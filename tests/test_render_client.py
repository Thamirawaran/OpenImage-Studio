import os
import sys
import threading
import time

import numpy as np
import pytest
from PIL import Image

from lis import render
from lis.render import RenderClient, WorkerConfig, WorkerUnavailable


@pytest.fixture
def img_paths(tmp_path):
    img = np.random.default_rng(0).integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    p = tmp_path / "W.png"
    Image.fromarray(img).save(p)
    return p, tmp_path / "candidate.png"


def job(img_paths, jid="r-1", steps=4, strategy="edit"):
    src, out = img_paths
    return {"type": "job", "id": jid, "strategy": strategy, "image_path": str(src), "mask_path": None,
            "instruction": "x", "seed": 1, "steps": steps, "strength": 0.8, "width": 96, "height": 64, "out_path": str(out)}


def client(**extra):
    c = RenderClient(WorkerConfig(module="lis.worker_fake", weights="", python=sys.executable, **extra))
    return c


def test_ready_progress_done(img_paths):
    c = client()
    ready = c.start()
    assert ready["type"] == "ready" and ready["profile_id"] == "fake"
    msgs = list(c.render(job(img_paths)))
    assert [m["type"] for m in msgs] == ["progress"] * 4 + ["done"]
    assert msgs[-1]["out_size"] == [96, 64] and os.path.exists(msgs[-1]["out_path"])
    # second job on the same process: no restart
    msgs2 = list(c.render(job(img_paths, "r-2", steps=2)))
    assert msgs2[-1]["type"] == "done" and c.deaths == 0
    c.close()


def test_soft_cancel_keeps_worker_alive(img_paths, monkeypatch):
    monkeypatch.setenv("LIS_FAKE_STEP_SECONDS", "0.3")
    c = client()
    c.start()
    ev = threading.Event()
    seen = []
    for m in c.render(job(img_paths, steps=20), cancel=ev):
        seen.append(m["type"])
        if m["type"] == "progress" and m["step"] == 2:
            ev.set()
    assert seen[-1] == "failed"
    assert c.deaths == 0 and c.proc is not None and c.proc.poll() is None  # still alive, model "loaded"
    msgs = list(c.render(job(img_paths, "r-after")))  # works without restart
    assert msgs[-1]["type"] == "done"
    c.close()


def test_hang_is_killed_and_counted(img_paths, monkeypatch):
    monkeypatch.setattr(render, "JOB_TIMEOUT_S", 1.0)
    c = client(extra_args=["--hang"])
    c.start()
    msgs = list(c.render(job(img_paths)))
    assert msgs[-1]["type"] == "failed" and "exceeded" in msgs[-1]["detail"]
    assert c.deaths == 1 and c.available
    # It restarts transparently for the next job (the fake still hangs, so it dies again).
    msgs = list(c.render(job(img_paths, "r-2")))
    assert msgs[-1]["type"] == "failed" and c.deaths == 2 and c.available
    msgs = list(c.render(job(img_paths, "r-2b")))
    assert msgs[-1]["type"] == "failed" and c.deaths == 3 and not c.available  # three in a row
    with pytest.raises(WorkerUnavailable):
        list(c.render(job(img_paths, "r-3")))


def test_unanswered_cancel_is_killed(img_paths, monkeypatch):
    monkeypatch.setattr(render, "CANCEL_GRACE_S", 0.5)
    c = client(extra_args=["--hang"])
    c.start()
    ev = threading.Event()
    ev.set()
    t0 = time.time()
    msgs = list(c.render(job(img_paths), cancel=ev))
    assert msgs[-1]["type"] == "failed" and msgs[-1]["cause"] == "cancelled"
    assert time.time() - t0 < 5 and c.deaths == 0  # the user asked for it: not a death
    assert c.available
