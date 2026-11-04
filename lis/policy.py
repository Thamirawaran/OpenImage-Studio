"""Route table and conflict rejection.

The rejections of the edit plan contract:
unknown mask id, empty instruction, non-offline network policy, unknown backend profile,
reserved kind, target wholly inside the protected union, and a plan whose parent is not
the current head. Plus the partial-overlap message and the auto strategy rule (auto means
edit, measured: the inpaint pipeline restores what it is asked to remove). Nothing here touches a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from lis.project import Project

SCHEMA_VERSION = 2
SUPPORTED_KINDS = ("native_instruction_edit", "move")  # move: removal by the model, then the object's own pixels placed at an offset
RESERVED_KINDS = ("crop", "rotate", "recolor_exact")
STRATEGIES = ("auto", "edit", "inpaint")
BLENDS = ("auto", "tight", "context")
CONTEXT_PATTERN = re.compile(r"\b(remove|erase|delete|get rid of|take out|wipe|replace the background|change the background|background|add|place|put|insert|give|wear|wearing|dress (?:her|him|them|the \w+) (?:up )?in|hold|holding|carry|replace|swap|into|(?:change|turn|switch|convert|make)\s+.{1,60}?\s+(?:to|into)\s+(?:a|an)|(?:change|turn|convert)\s+.{1,60}?\s+to\s+(?:a|an)\s)\b", re.IGNORECASE)
REMOVAL_PATTERN = re.compile(r"\b(remove|erase|delete|fill( in)?|clean up|get rid of|take out|wipe)\b", re.IGNORECASE)


@dataclass
class Resolution:
    status: str  # "ok" | "rejected"
    message: str = ""
    strategy: str | None = None  # "edit" | "inpaint" when ok
    blend: str = "tight"  # "tight" | "context" when ok
    target: np.ndarray | None = None  # bool mask
    protected: np.ndarray | None = None  # bool union, may be all False
    excluded_fraction: float = 0.0  # of the target that lies inside protected

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def choose_strategy(requested: str, instruction: str) -> str:
    """auto always means edit. `instruction` is kept for future rules."""
    if requested in ("edit", "inpaint"):
        return requested
    return "edit"


def choose_blend(requested: str, instruction: str) -> str:
    """Removals, additions and background changes need the model's consequences (shadows, the
    scene behind, the full shape of an added object) and a wide edge; changes to an object
    itself need a tight edge. Added objects were cut to the drawn area's straight edges with a
    tight edge (evaluation 3 Nov, rug task)."""
    if requested in ("tight", "context"):
        return requested
    return "context" if CONTEXT_PATTERN.search(instruction) else "tight"


def resolve(plan: dict, project: Project, manifest: dict) -> Resolution:
    """Validate an EditPlan against the project and the asset manifest. Order follows the
    contract so the first problem a user can fix is the one they see."""
    def rej(msg: str) -> Resolution:
        return Resolution("rejected", msg)

    if plan.get("schema_version") != SCHEMA_VERSION:
        return rej(f"This plan is schema {plan.get('schema_version')}; the engine speaks schema {SCHEMA_VERSION}.")
    op = plan.get("operation") or {}
    target_id = op.get("target_mask_id")
    protected_ids = list(plan.get("protected_mask_ids") or [])
    known = {m.mask_id for m in project.record.masks}
    for mid in [target_id, *protected_ids]:
        if mid not in known:
            return rej(f"Mask {mid!r} does not exist in this project.")
    instruction = (op.get("instruction") or "").strip()
    if not instruction:
        return rej("The instruction is empty. Say what should change.")
    if plan.get("network_policy") != "offline":
        return rej("Only network_policy 'offline' is accepted. This application never uses the network during an edit.")
    profile = plan.get("backend_profile_id")
    if profile not in (manifest.get("profiles") or {}):
        return rej(f"Backend profile {profile!r} is not installed. Run scripts/download_assets.py and check assets/manifest.json.")
    kind = op.get("kind")
    if kind in RESERVED_KINDS:
        return rej(f"Operation {kind!r} is reserved and not available in this version.")
    if kind not in SUPPORTED_KINDS:
        return rej(f"Operation {kind!r} is not supported.")
    if kind == "move":
        off = op.get("offset")
        if not (isinstance(off, (list, tuple)) and len(off) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in off)) or off == [0, 0] or tuple(off) == (0, 0):
            return rej("A move needs an offset of two whole numbers, not both zero.")
    if op.get("strategy", "auto") not in STRATEGIES:
        return rej(f"Strategy {op.get('strategy')!r} is not one of {STRATEGIES}.")
    if op.get("blend", "auto") not in BLENDS:
        return rej(f"Blend {op.get('blend')!r} is not one of {BLENDS}.")
    if plan.get("parent_revision") != project.record.head:
        return rej("The image changed since this plan was made. Look at the current result and render again.")

    target = project.mask(target_id)
    protected = project.union(protected_ids)
    target_px = int(target.sum())
    if target_px == 0:
        return rej(f"Mask {project.mask_record(target_id).name!r} is empty.")
    inside = int((target & protected).sum())
    if inside == target_px:
        names = ", ".join(project.mask_record(m).name for m in protected_ids if (target & project.mask(m)).any())
        return rej(f"The target is entirely inside protected mask {names}. Unprotect it or pick another target.")

    frac = inside / target_px
    msg = ""
    if inside:
        msg = f"{frac:.0%} of the target is protected and will not change."
    return Resolution("ok", msg, strategy=choose_strategy(op.get("strategy", "auto"), instruction),
                      blend=choose_blend(op.get("blend", "auto"), instruction), target=target, protected=protected,
                      excluded_fraction=frac)
