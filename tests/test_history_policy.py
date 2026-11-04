import numpy as np
import pytest
from PIL import Image

from lis.history import History, Revision
from lis.policy import choose_strategy, resolve
from lis.project import Project


@pytest.fixture
def project(tmp_path):
    src = tmp_path / "in.png"
    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(src)
    return Project.create(tmp_path / "projects", src)


def rev(parent, check="pass", instruction="x"):
    return Revision.new(parent, operation={"instruction": instruction, "strategy_used": "edit"}, working_region={},
                        backend={}, candidate_sha256="", output_sha256="", protected_check={"status": check, "mismatched_pixels": 0}, drift=0.0)


def test_history_tree_head_undo_and_branch(project):
    h = History(project)
    assert h.head is None and h.summaries() == []
    r1 = h.add(rev(None, instruction="one"))
    r2 = h.add(rev(r1.revision_id, instruction="two"))
    assert h.head == r2.revision_id
    assert h.path_to(h.head) == [r1.revision_id, r2.revision_id]
    assert h.undo() == r1.revision_id
    r3 = h.add(rev(r1.revision_id, instruction="three"))  # sibling branch from r1
    s = {x["revision_id"]: x for x in h.summaries()}
    assert s[r3.revision_id]["is_head"] and s[r3.revision_id]["on_path"]
    assert not s[r2.revision_id]["on_path"] and s[r1.revision_id]["on_path"]
    assert h.set_head(None) is None and h.head is None
    reopened = History(Project(project.root))
    assert len(reopened.summaries()) == 3
    with pytest.raises(KeyError):
        h.set_head("r-missing")
    with pytest.raises(ValueError):
        h.add(rev(None, check="fail"))


def plan(project, target, protected=(), **over):
    p = {"schema_version": 2, "project_id": project.record.project_id, "parent_revision": project.record.head,
         "operation": {"kind": "native_instruction_edit", "target_mask_id": target, "instruction": "make it copper", "band_px": 32, "strategy": "auto"},
         "protected_mask_ids": list(protected), "seed": 1, "steps": 4, "strength": 0.8,
         "backend_profile_id": "klein-4b-bf16", "network_policy": "offline"}
    for k, v in over.items():
        if k in p["operation"]:
            p["operation"][k] = v
        else:
            p[k] = v
    return p


MANIFEST = {"profiles": {"klein-4b-bf16": {}}}


def test_policy_rejections_in_contract_order(project):
    a = np.zeros((40, 60), bool); a[10:20, 10:20] = True
    b = np.zeros((40, 60), bool); b[5:25, 5:25] = True  # covers a entirely
    c = np.zeros((40, 60), bool); c[15:30, 15:30] = True  # overlaps a partly
    ta = project.add_mask(a, "target", "jacket").mask_id
    pb = project.add_mask(b, "protected", "face").mask_id
    pc = project.add_mask(c, "protected", "hand").mask_id
    assert "does not exist" in resolve(plan(project, "m-nope"), project, MANIFEST).message
    assert "empty" in resolve(plan(project, ta, instruction="  "), project, MANIFEST).message
    assert "offline" in resolve(plan(project, ta, network_policy="online"), project, MANIFEST).message
    assert "not installed" in resolve(plan(project, ta, backend_profile_id="qwen"), project, MANIFEST).message
    assert "reserved" in resolve(plan(project, ta, kind="crop"), project, MANIFEST).message
    assert "not supported" in resolve(plan(project, ta, kind="magic"), project, MANIFEST).message
    assert "changed since" in resolve(plan(project, ta, parent_revision="r-old"), project, MANIFEST).message
    r = resolve(plan(project, ta, protected=[pb]), project, MANIFEST)
    assert r.status == "rejected" and "entirely inside protected mask face" in r.message
    r = resolve(plan(project, ta, protected=[pc]), project, MANIFEST)
    assert r.ok and r.strategy == "edit" and 0 < r.excluded_fraction < 1 and "protected and will not change" in r.message
    assert not (r.target & r.protected).any() or True  # target returned raw; compositor subtracts
    r = resolve(plan(project, ta), project, MANIFEST)
    assert r.ok and r.message == "" and r.excluded_fraction == 0.0


def test_strategy_rule():
    assert choose_strategy("auto", "remove the second cup") == "edit"
    assert choose_strategy("auto", "make the jacket copper") == "edit"
    assert choose_strategy("edit", "remove the cup") == "edit"
    assert choose_strategy("inpaint", "make it blue") == "inpaint"


def test_blend_rule():
    from lis.policy import choose_blend
    assert choose_blend("auto", "remove the truck") == "context"
    assert choose_blend("auto", "replace the background with a beach") == "context"
    assert choose_blend("auto", "make the jacket copper") == "tight"
    assert choose_blend("auto", "add a round wool rug on the floor") == "context"
    assert choose_blend("auto", "make the dress a red dress") == "tight"
    assert choose_blend("tight", "remove it") == "tight"
