import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lis.engine import Engine, EngineConfig
from lis.ground import (Found, Phrase, colour_fraction, default_pick, fill_small_holes, make_phrase, parse_instruction, rank,
                        singular)
from lis.render import WorkerConfig

# ---- parsing: the evaluation sentences and the awkward cases -------------------------------

CASES = [
    ("change the denim jacket into a black leather jacket, keep both faces unchanged", ["denim jacket"], ["faces"], False),
    ("make the hoodie forest green, keep the faces and the denim jacket unchanged", ["hoodie"], ["faces", "denim jacket"], False),
    ("make the mug matte navy blue ceramic, do not change the bottle", ["mug"], ["bottle"], False),
    ("change the sofa to a brown leather sofa, keep the armchair and the plant as they are", ["sofa"], ["armchair", "plant"], False),
    ("remove the potted plant and its pot, show the wall and floor behind it, keep the sofa unchanged", ["potted plant", "pot"], ["sofa"], False),
    ("paint the front door glossy dark green, leave the dog untouched", ["front door"], ["dog"], False),
    ("make the truck body polished copper metal, keep the wheels unchanged", ["truck body"], ["wheels"], False),
    ("make the dark red car pastel mint green, keep the pink car unchanged", ["dark red car"], ["pink car"], False),
    ("remove the yellow sign and its pole, show the building and trees behind it, keep the cars unchanged", ["yellow sign", "pole"], ["cars"], False),
    ("replace the background with a sunny seaside promenade with palm trees, keep the woman unchanged", [], ["woman"], True),
    ("add a round beige wool rug on the floor, keep the sofa and the armchair unchanged", ["floor"], ["sofa", "armchair"], False),
    ("make him look surprised", ["man"], [], False),
    ("make the sky darker", ["sky"], [], False),
    ("make the car on the left red", ["left car"], [], False),
    ("turn the black and white dog brown", ["black and white dog"], [], False),
    ("give the cat a red hat while keeping the dog unchanged", ["cat"], ["dog"], False),
    ("change everything except the dress to black and white", [], ["dress"], True),
    ("recolor the biggest balloon yellow", ["largest balloon"], [], False),
]


@pytest.mark.parametrize("text,targets,protect,background", CASES)
def test_parse_instruction(text, targets, protect, background):
    p = parse_instruction(text).to_json()
    assert p["targets"] == targets
    assert p["protect"] == protect
    assert p["background"] == background
    assert "keep" not in p["edit"] and "leave" not in p["edit"]  # keep clauses are enforced by masks, not sent


def test_parse_edge_cases():
    assert parse_instruction("").targets == []
    p = parse_instruction("add a hat")
    assert not p.targets and p.whole and any("where to add" in n for n in p.notes)  # the model places it; the note says how to choose
    p = parse_instruction("change the background to a beach")
    assert p.background and [x.text for x in p.protect] == ["person"] and p.notes  # the assumption is stated
    assert parse_instruction("make his face look surprised but keep his face unchanged").to_json()["protect"] == ["face"]


def test_phrases_singular_plural_position():
    assert make_phrase("both faces").plural and make_phrase("the dress").plural is False
    assert make_phrase("the glasses").plural is False
    assert make_phrase("the car on the right").position == "right"
    assert make_phrase("the front door").text == "front door" and make_phrase("the front door").position is None
    assert singular("faces") == "face" and singular("people") == "person" and singular("dress") == "dress"
    assert singular("berries") == "berry" and singular("boxes") == "box" and singular("pink cars") == "pink car"


def test_rank_by_position():
    found = [Found([500, 0, 600, 100], 0.9), Found([0, 0, 100, 100], 0.5)]
    assert rank(found, "left", (200, 700))[0].box[0] == 0
    assert rank(found, "right", (200, 700))[0].box[0] == 500
    assert rank(found, None, (200, 700))[0].score == 0.9


def test_colour_fraction_tells_pink_from_dark_red():
    img = np.zeros((10, 20, 3), np.uint8)
    img[:, :10] = (120, 20, 25)  # dark red
    img[:, 10:] = (210, 30, 150)  # magenta pink
    left, right = np.zeros((10, 20), bool), np.zeros((10, 20), bool)
    left[:, :10], right[:, 10:] = True, True
    assert colour_fraction(img, right, "pink") > 0.9 and colour_fraction(img, left, "pink") < 0.1
    assert colour_fraction(img, left, "red") > 0.9 and colour_fraction(img, right, "red") < 0.1


def test_default_pick_skips_protected_and_prefers_confident():
    a, b = np.zeros((10, 10), bool), np.zeros((10, 10), bool)
    a[:5], b[5:] = True, True
    items = [{"mask": a, "score": 0.6, "confident": True}, {"mask": b, "score": 0.2, "confident": False}]
    assert default_pick(items, False)[0] is items[0]
    assert default_pick(items, False, avoid=a)[0] is items[1]  # the confident one is protected: offer the weak one
    assert len(default_pick(items, True)) == 1  # plural takes only confident instances


def test_fill_small_holes():
    m = np.zeros((50, 50), bool)
    m[10:40, 10:40] = True
    m[20:22, 20:22] = False
    assert fill_small_holes(m)[21, 21]


# ---- engine and UI flow with fake models ---------------------------------------------------


def engine(tmp_path, table):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"profiles": {"klein-4b-bf16": {"files": {}}}, "last_verified": "t"}))
    return Engine(EngineConfig(projects_dir=tmp_path / "projects", manifest_path=manifest,
                               worker=WorkerConfig(module="lis.worker_fake", weights="", python=sys.executable),
                               segmenter="fake", grounder="fake", grounder_table=table))


def photo(tmp_path):
    img = np.random.default_rng(0).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    p = tmp_path / "photo.png"
    Image.fromarray(img).save(p)
    return p


TABLE = {"car": [[40, 200, 240, 400, 0.8], [400, 200, 600, 400, 0.7]], "face": [[300, 20, 360, 90, 0.7]], "jacket": [[250, 100, 400, 300, 0.6]]}


def test_ground_returns_instances_and_union(tmp_path):
    eng = engine(tmp_path, TABLE)
    pid = eng.open(photo(tmp_path))["project_id"]
    r = eng.ground(pid, "o1", "the car")
    labels = [c["label"] for c in r["candidates"]]
    assert labels[:2] == ["car 1", "car 2"] and labels[-1] == "all 2 together"
    r = eng.ground(pid, "o2", "cars")
    assert r["candidates"][0]["label"] == "all 2 cars"  # plural: the union comes first
    assert eng.ground(pid, "o3", "the boat")["candidates"] == []
    m = eng.commit_mask(pid, "o1", r["candidates"][0]["candidate_id"], "target", "cars")
    rec = eng._project(pid).mask_record(m["mask_id"])
    assert rec.prompts["text"]["text"] == "cars" and rec.prompts["text"]["grounder"] == "fake"


def test_propose_then_render_keeps_protected_pixels(tmp_path):
    eng = engine(tmp_path, TABLE)
    eng.client.start()
    try:
        pid = eng.open(photo(tmp_path))["project_id"]
        r = eng.propose(pid, "make the jacket red, keep the face unchanged")
        assert r["target"]["candidate_id"] and r["protect"][0]["candidate_id"] and r["edit"] == "make the jacket red"
        t = eng.commit_mask(pid, "a", r["target"]["candidate_id"], "target", "jacket")["mask_id"]
        f = eng.commit_mask(pid, "b", r["protect"][0]["candidate_id"], "protected", "face")["mask_id"]
        plan = {"schema_version": 2, "project_id": pid, "parent_revision": None,
                "operation": {"kind": "native_instruction_edit", "target_mask_id": t, "instruction": r["edit"], "band_px": 8, "strategy": "auto"},
                "protected_mask_ids": [f], "seed": 1, "steps": 2, "strength": 0.8, "backend_profile_id": "klein-4b-bf16", "network_policy": "offline"}
        res = [m for m in eng.render(plan) if m["type"] == "result"][0]
        assert res["status"] == "ok" and res["protected_check"] == {"status": "pass", "mismatched_pixels": 0}
    finally:
        eng.close()


def test_propose_background_and_conflict(tmp_path):
    eng = engine(tmp_path, {"woman": [[200, 50, 400, 450, 0.9]], "face": [[300, 20, 360, 90, 0.7]]})
    pid = eng.open(photo(tmp_path))["project_id"]
    r = eng.propose(pid, "replace the background with a beach, keep the woman unchanged")
    t = np.asarray(Image.open(r["target"]["png_path"])) > 127
    k = np.asarray(Image.open(r["protect"][0]["png_path"])) > 127
    assert r["target"]["background"] and not (t & k).any() and (t | k).all()
    # Same phrase to edit and keep: proposed as a conflict (the renderer then rejects it), never moved to another object.
    r = eng.propose(pid, "make his face look surprised but keep his face unchanged")
    t = np.asarray(Image.open(r["target"]["png_path"])) > 127
    k = np.asarray(Image.open(r["protect"][0]["png_path"])) > 127
    assert (t & ~k).sum() == 0
    # The user's corrections in the fields replace the parse.
    r = eng.propose(pid, "make his face look surprised", target_text="woman", protect_text="")
    assert r["target"]["phrases"] == ["woman"] and r["protect"] == []


def test_ui_suggest_use_and_find(tmp_path):
    from lis.app import UI, new_state

    eng = engine(tmp_path, TABLE)
    ui = UI(eng)
    st, *_ = ui.open(str(photo(tmp_path)), new_state())
    instr = "make the jacket red, keep the face unchanged"
    st, canvas, status, tf, pf, md = ui.suggest(instr, "", "", st)
    assert tf["value"] == "jacket" and pf["value"] == "face" and "sent to the model" in md and st["proposal"]
    # Render before using the proposal says so instead of rendering.
    out = list(ui.render(instr, 1, 2, 8, "auto", 0.8, None, st))
    assert "proposed but not kept" in out[-1][3]
    st, canvas, status, objs = ui.use_proposal(st)
    assert [o["kind"] for o in st["objects"]] == ["protected", "target"] and st["proposal"] is None
    st, canvas, status, cands = ui.find("the car", st)
    assert len(st["candidates"]) == 3 and st["box"] == [40, 200, 240, 400]
    st, _ = ui.pick("2. car 2 (score 0.70)", st)
    assert st["picked"] == 1 and st["box"] == [400, 200, 600, 400]


# ---- the edit check ------------------------------------------------------------------------


def test_verify_edit_recolour_and_unchanged():
    from lis.verify import verify_edit

    base = np.zeros((40, 40, 3), np.uint8)
    base[:, :] = (200, 30, 150)  # pink
    target = np.zeros((40, 40), bool)
    target[10:30, 10:30] = True
    prot = np.zeros((40, 40), bool)
    out = base.copy()
    out[target] = (10, 10, 10)
    assert verify_edit(base, out, target, prot, "make the pink car matte black", "pink car")["status"] == "verified"
    assert verify_edit(base, out, target, prot, "make the pink car yellow", "pink car")["status"] == "not_detected"
    assert verify_edit(base, base, target, prot, "make the pink car matte black", "pink car")["status"] == "unchanged"
    out2 = base.copy()
    out2[target] = (120, 90, 60)
    assert verify_edit(base, out2, target, prot, "give the car a leather texture", "car")["status"] == "changed"


def test_render_reports_outside_and_edit_checks(tmp_path):
    eng = engine(tmp_path, TABLE)
    eng.client.start()
    try:
        pid = eng.open(photo(tmp_path))["project_id"]
        r = eng.propose(pid, "make the jacket red, keep the face unchanged")
        t = eng.commit_mask(pid, "a", r["target"]["candidate_id"], "target", "jacket")["mask_id"]
        f = eng.commit_mask(pid, "b", r["protect"][0]["candidate_id"], "protected", "face")["mask_id"]
        plan = {"schema_version": 2, "project_id": pid, "parent_revision": None,
                "operation": {"kind": "native_instruction_edit", "target_mask_id": t, "instruction": r["edit"], "band_px": 8, "strategy": "auto"},
                "protected_mask_ids": [f], "seed": 1, "steps": 2, "strength": 0.8, "backend_profile_id": "klein-4b-bf16", "network_policy": "offline"}
        res = [m for m in eng.render(plan) if m["type"] == "result"][0]
        assert res["outside_check"]["status"] == "pass" and res["outside_check"]["mismatched_pixels"] == 0
        assert res["outside_check"]["changed_pixels"] > 0 and Path(res["change_map_path"]).exists()
        assert res["edit_check"]["status"] in ("verified", "not_detected", "changed", "unchanged")
        # Tampering outside the allowed area is caught on the file.
        p = eng._project(pid)
        out = np.asarray(Image.open(res["image_path"])).copy()
        allowed = np.asarray(Image.open(Path(res["image_path"]).parent / "allowed.png")) > 127
        ys, xs = np.nonzero(~allowed)
        out[ys[0], xs[0]] ^= 1
        Image.fromarray(out).save(res["image_path"])
        chk = eng._outside_check(p, Path(res["image_path"]), p.base_path(None), Path(res["image_path"]).parent / "allowed.png")
        assert chk["status"] == "fail" and chk["mismatched_pixels"] == 1
    finally:
        eng.close()


def test_parser_everyday_phrasings():
    from lis.ground import parse_instruction

    cases = {
        "can you make her dress green?": (["dress"], []),
        "please change the color of the sofa to dark green but don't touch the cushions": (["sofa"], ["cushions"]),
        "the car should be yellow": (["car"], []),
        "I want the jacket to be leather": (["jacket"], []),
        "swap the cat for a dog": (["cat"], []),
        "make the sign say OPEN": (["sign"], []),
        "make the mug navy blue and the bottle red": (["mug", "bottle"], []),
        "Make the sky dramatic. Keep the building unchanged.": (["sky"], ["building"]),
        "make the jacket and the trousers black": (["jacket", "trousers"], []),
        "make the dog black and white": (["dog"], []),
        "could you please remove the man on the left, keep the woman": (["left man"], ["woman"]),
        "make the dog toy red": (["dog toy"], []),
        "turn the plastic chair wooden": (["plastic chair"], []),
    }
    for text, (targets, protect) in cases.items():
        p = parse_instruction(text)
        assert [t.label() for t in p.targets] == targets, text
        assert [t.label() for t in p.protect] == protect, text


def test_parser_relations_whole_photo_and_implicit_keeps():
    from lis.ground import parse_instruction

    cases = {
        "make the cup next to the laptop red": (["cup next to laptop"], [], False, False),
        "erase the logo on his shirt": (["logo on shirt"], [], False, False),
        "remove the person in the background": (["smallest person"], [], False, False),
        "make the woman in the blue dress smile": (["woman in blue dress"], [], False, False),
        "make the man in red smile": (["man"], [], False, False),
        "make the car on the left blue": (["left car"], [], False, False),
        "make it look like winter": ([], [], False, True),
        "turn this into a watercolor painting": ([], [], False, True),
        "make the photo look like a film still, keep the faces": ([], ["faces"], False, True),
        "make the background blurry": ([], ["person"], True, False),
        "change the background to a beach with palm trees, keep the woman": ([], ["woman"], True, False),
        "make the shoes white, don't change anything else": (["shoes"], [], False, False),
        "keep the rest of the photo unchanged and make the sky pink": (["sky"], [], False, False),
        "change the dress to a red dress with a small white floral pattern, keep her face unchanged": (["dress"], ["face"], False, False),
        "turn the man with the hat into a statue": (["man with hat"], [], False, False),
    }
    for text, (targets, protect, background, whole) in cases.items():
        p = parse_instruction(text)
        assert [t.label() for t in p.targets] == targets, text
        assert [t.label() for t in p.protect] == protect, text
        assert (p.background, p.whole) == (background, whole), text
    # The renderer gets the request, not the politeness around it.
    assert parse_instruction("can you please make her dress red but keep her face").edit == "make her dress red"


def test_relation_picks_the_right_instance(tmp_path):
    # Two cups; only the right one is next to the laptop.
    eng = engine(tmp_path, {"cup": [[20, 300, 80, 380, 0.9], [420, 300, 480, 380, 0.85]], "laptop": [[500, 250, 620, 400, 0.9]]})
    pid = eng.open(photo(tmp_path))["project_id"]
    r = eng.propose(pid, "make the cup next to the laptop red")
    t = np.asarray(Image.open(r["target"]["png_path"])) > 127
    assert t[340, 450] and not t[340, 50]
    r = eng.propose(pid, "make it look like winter, keep the laptop")
    t = np.asarray(Image.open(r["target"]["png_path"])) > 127
    k = np.asarray(Image.open(r["protect"][0]["png_path"])) > 127
    assert r["target"]["whole"] and not (t & k).any() and (t | k).all()


def test_parse_benchmark_sentences():
    """Sentences from the Emu Edit test set that were once read wrongly."""
    cases = {
        "Change the color of the coffee table to white.": ["coffee table"],
        "Change the fluffy white clouds to dark blue.": ["fluffy white clouds"],
        "Change the texture of the bench from wood to chrome.": ["bench"],
        "Change all of the white parts of the truck to be filled in green": ["truck"],
    }
    for s, want in cases.items():
        assert [t.text for t in parse_instruction(s).targets] == want or [t.label() for t in parse_instruction(s).targets] == want, s
    for s in ("Change the image so that it appears there is rain and lightning in the image.", "Fill the bathroom with water",
              "I want to see this in a winter scene."):
        p = parse_instruction(s)
        assert p.whole and not p.targets, s
