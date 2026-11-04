import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from lis.app import UI, new_state
from lis.engine import Engine, EngineConfig
from lis.render import WorkerConfig


def test_handlers_drive_a_full_edit(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"profiles": {"fake": {"files": {}}}}))
    eng = Engine(EngineConfig(projects_dir=tmp_path / "projects", manifest_path=manifest, worker=WorkerConfig(module="lis.worker_fake", python=sys.executable),
                              segmenter="fake", profile_id="fake", color_match=False))
    ui = UI(eng)
    img = np.random.default_rng(0).integers(0, 256, size=(900, 1500, 3), dtype=np.uint8)
    src = tmp_path / "p.png"
    Image.fromarray(img).save(src)
    st = new_state()
    st, canvas, status, _, _, hist, _, *cleared = ui.open(str(src), st)
    assert cleared[0] == "" and cleared[1] is None
    assert st["scale"] < 1.0 and canvas.shape[1] == 1200 and hist["choices"][0][0] == "source" if isinstance(hist["choices"][0], tuple) else hist["choices"] == ["source"]
    # click in proxy coordinates; the handler maps back to canonical
    st, canvas, status, cands = ui.click(st, SimpleNamespace(index=[600, 360]))
    assert len(cands["choices"]) == 2 and st["candidates"]
    st, canvas, status, objs, _ = ui.keep("jacket", st)
    assert [c if isinstance(c, str) else c[0] for c in objs["choices"]] == ["jacket (target)"]
    st, status = ui.set_mode("protected", st)
    st, canvas, status, choices = ui.click(st, SimpleNamespace(index=[690, 360]))
    st, canvas, status, objs, _ = ui.keep("face", st)
    assert [c if isinstance(c, str) else c[0] for c in objs["choices"]] == ["jacket (target)", "face (protected)"]
    outs = list(ui.render("make the jacket copper", 1, 3, 32, "auto", 0.8, "jacket (target)", st))
    final = outs[-1]
    st, slider, md, status, hist, canvas_after, change_map = final
    assert canvas_after is not None and change_map is not None and not isinstance(change_map, dict)
    assert "Outside the edit area: unchanged" in md and "**Edit:**" in md
    assert slider is not None and "check: PASS" in md and "edited at" in md
    assert hist["value"].startswith("r-") and len(hist["choices"]) == 2
    out = np.asarray(Image.open(st["last_result"]["image_path"]).convert("RGB"))
    face = eng._project(st["project_id"]).mask(st["objects"][1]["mask_id"])
    assert np.array_equal(out[face], img[face])
    st, canvas, status, hist = ui.undo(st)
    assert hist["value"] == "source"
    path, note = ui.export("png", st)
    assert Path(path).exists() and "check" in note
    eng.close()
