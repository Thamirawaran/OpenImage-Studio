"""Did the requested edit happen? The protected check proves what did not change; this
module checks, as far as a machine can, that what was asked for did change.

By the kind of instruction:
- removal ("remove the sign"): the grounder looks for the phrase again on the result; the
  edit is verified when no confident instance remains over the target.
- addition ("add a rug on the floor"): the grounder looks for the added thing; verified when
  a confident instance lies mostly inside the target.
- recolour (the edit names a colour the target phrase does not): verified when the share of
  target pixels with that colour rose clearly.
- anything else (material, pattern, background, expression): the target changed by more
  than noise. Whether it matches the words is the user's judgement, and the UI says so.

Never alters pixels and never blocks a result: it is a report, with a suggestion to try
another seed when the edit was not detected.
"""

from __future__ import annotations

import re

import numpy as np

from lis.ground import BACKGROUND_WORDS, _COLOURS, colour_fraction, find_phrase, make_phrase, parse_instruction

CHANGED_MEAN = 6.0  # mean |after - before| over the editable target, 0..255, that counts as a change
UNCHANGED_MEAN = 2.0
COLOUR_MIN_AFTER = 0.25
COLOUR_MIN_RISE = 0.15
STILL_THERE = 0.3  # an instance covering this share of itself inside the target means "still there"


def new_colours(edit: str, target_text: str) -> list[str]:
    """Colour words in the edit that do not describe the target as it is now."""
    target_words = set(target_text.lower().split())
    words = [w.strip(".,").lower() for w in edit.split()]
    return [w for w in words if w in _COLOURS and w not in target_words]


def verify_edit(base: np.ndarray, out: np.ndarray, target: np.ndarray, protected: np.ndarray, edit: str,
                target_text: str | None = None, grounder=None, segmenter=None, min_change: float = UNCHANGED_MEAN) -> dict:
    """`min_change` below 0 skips the "unchanged" shortcut; the evaluation uses it for controls."""
    editable = target & ~protected
    if not editable.any():
        return {"status": "not_checked", "kind": None, "message": "nothing editable to check"}
    d = np.abs(out.astype(np.int16) - base.astype(np.int16)).max(axis=-1)
    mean_change = float(d[editable].mean())
    parsed = parse_instruction(edit)
    phrases = parsed.targets
    if target_text and not phrases and target_text not in BACKGROUND_WORDS and target_text != "whole photo":
        ph = make_phrase(target_text)
        phrases = [ph] if ph else []
    ttext = target_text or " ".join(p.text for p in phrases)
    evidence = {"mean_change": round(mean_change, 2)}

    if mean_change < min_change:
        return {"status": "unchanged", "kind": None, "evidence": evidence,
                "message": "the target looks unchanged; try another seed or reword the instruction"}

    removal = bool(phrases) and any(edit.lower().lstrip().startswith(v) for v in ("remove", "erase", "delete", "get rid of", "take out", "take away", "wipe", "clear away", "clean up"))
    if removal and grounder is not None:
        grounder.set_image(out)
        if segmenter is not None and getattr(grounder, "name", "").startswith("dino"):
            segmenter.set_image(out)
        left = []
        for ph in phrases:
            for it in find_phrase(grounder, segmenter, out, ph):
                if it["confident"] and (it["mask"] & target).sum() >= STILL_THERE * max(1, it["mask"].sum()):
                    left.append(ph.text)
                    break
        evidence["still_found"] = left
        if left:
            return {"status": "not_detected", "kind": "removal", "evidence": evidence,
                    "message": f"{', '.join(left)} still found where it was; try another seed"}
        return {"status": "verified", "kind": "removal", "evidence": evidence,
                "message": f"{', '.join(p.text for p in phrases)} no longer found where it was"}

    if parsed.adds and grounder is not None:
        ph = make_phrase(parsed.adds)
        if ph is not None:
            grounder.set_image(out)
            if segmenter is not None and getattr(grounder, "name", "").startswith("dino"):
                segmenter.set_image(out)
            # An added thing may reach past the selection (a hat above the head): count where it
            # changed the photo too. Detectors know "hat" better than "wide straw sun hat".
            where = target | (d > 24)
            hits = []
            for q in [ph] + ([make_phrase(ph.text.split()[-1])] if len(ph.text.split()) > 1 else []):
                if q is None:
                    continue
                hits = [it for it in find_phrase(grounder, segmenter, out, q)
                        if it["confident"] and (it["mask"] & where).sum() >= 0.5 * max(1, it["mask"].sum())]
                if hits:
                    break
            evidence["found_added"] = len(hits)
            if hits:
                return {"status": "verified", "kind": "addition", "evidence": evidence, "message": f"{ph.text} found in the selected area"}
            return {"status": "not_detected", "kind": "addition", "evidence": evidence,
                    "message": f"{ph.text} not found in the selected area; try another seed"}

    colours = new_colours(parsed.edit, ttext)
    if colours and not parsed.background and not parsed.whole and target_text != "whole photo":
        pastel = bool(re.search(r"\b(pastel|pale|light|baby|powder|soft)\b", parsed.edit, re.I))
        before = max(colour_fraction(base, editable, c, pastel) for c in colours)
        after = max(colour_fraction(out, editable, c, pastel) for c in colours)
        evidence.update(colour=colours, colour_before=round(before, 3), colour_after=round(after, 3))
        if after >= COLOUR_MIN_AFTER and after - before >= COLOUR_MIN_RISE:
            return {"status": "verified", "kind": "recolour", "evidence": evidence,
                    "message": f"the target is now {round(after * 100)}% {colours[0]} (was {round(before * 100)}%)"}
        return {"status": "not_detected", "kind": "recolour", "evidence": evidence,
                "message": f"the target does not look {' '.join(colours)} ({round(after * 100)}%); try another seed"}

    if mean_change >= CHANGED_MEAN:
        return {"status": "changed", "kind": "other", "evidence": evidence,
                "message": "the target changed; whether it matches the words is yours to judge"}
    return {"status": "unchanged", "kind": "other", "evidence": evidence,
            "message": "the target barely changed; try another seed or reword the instruction"}
