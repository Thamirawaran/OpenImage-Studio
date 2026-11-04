"""Revisions.

Immutable revision tree with a head pointer. set_head is both undo and revert; rendering
from a non-head revision creates a sibling branch.
Revisions are stored as dicts inside project.json; this module is the only writer.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from lis.project import Project, new_id


@dataclass
class Revision:
    revision_id: str
    parent: str | None
    created_at: str
    operation: dict  # kind, strategy_requested, strategy_used, instruction, target_mask_id,
    # protected_mask_ids, band_px, seed, steps, strength
    working_region: dict  # x, y, w, h, scale
    backend: dict  # profile_id, manifest_sha256 (per file), last_verified
    candidate_sha256: str
    output_sha256: str
    protected_check: dict  # status: pass | not_applicable, mismatched_pixels
    drift: float
    timing_ms: dict = field(default_factory=dict)
    outside_check: dict = field(default_factory=dict)  # status: pass, mismatched_pixels (changed outside the edit area), changed_pixels
    edit_check: dict = field(default_factory=dict)  # status: verified | not_detected | changed | unchanged | not_checked

    @staticmethod
    def new(parent: str | None, **fields) -> "Revision":
        return Revision(revision_id=new_id("r"), parent=parent,
                        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields)


class History:
    def __init__(self, project: Project):
        self.project = project

    # ---- reads ---------------------------------------------------------------------

    @property
    def head(self) -> str | None:
        return self.project.record.head

    def get(self, revision_id: str) -> Revision:
        for r in self.project.record.revisions:
            if r["revision_id"] == revision_id:
                return Revision(**r)
        raise KeyError(revision_id)

    def exists(self, revision_id: str | None) -> bool:
        return revision_id is None or any(r["revision_id"] == revision_id for r in self.project.record.revisions)

    def path_to(self, revision_id: str | None) -> list[str]:
        """Revision ids from the root to `revision_id`, oldest first."""
        path: list[str] = []
        while revision_id is not None:
            path.append(revision_id)
            revision_id = self.get(revision_id).parent
        return path[::-1]

    def summaries(self) -> list[dict]:
        on_path = set(self.path_to(self.head))
        out = []
        for r in self.project.record.revisions:
            out.append({
                "revision_id": r["revision_id"], "parent": r["parent"], "instruction": r["operation"].get("instruction", ""),
                "strategy_used": r["operation"].get("strategy_used"), "protected_check": r["protected_check"],
                "created_at": r["created_at"], "is_head": r["revision_id"] == self.head,
                "on_path": r["revision_id"] in on_path,
            })
        return out

    # ---- writes --------------------------------------------------------------------

    def add(self, rev: Revision) -> Revision:
        """Append an immutable revision and move head to it. A failed render never gets here."""
        if rev.protected_check.get("status") not in ("pass", "not_applicable"):
            raise ValueError("only revisions with a passing or not-applicable protected check are recorded")
        if not self.exists(rev.parent):
            raise KeyError(rev.parent)
        if any(r["revision_id"] == rev.revision_id for r in self.project.record.revisions):
            raise ValueError("duplicate revision id")
        self.project.record.revisions.append(asdict(rev))
        self.project.record.head = rev.revision_id
        self.project.save()
        return rev

    def set_head(self, revision_id: str | None) -> str | None:
        """Undo is set_head(parent); revert is set_head(any). None shows the canonical source."""
        if not self.exists(revision_id):
            raise KeyError(revision_id)
        self.project.record.head = revision_id
        self.project.save()
        return revision_id

    def undo(self) -> str | None:
        if self.head is None:
            return None
        return self.set_head(self.get(self.head).parent)
