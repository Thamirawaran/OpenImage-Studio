"""Selection from words. Two parts:

1. `parse_instruction` splits a plain-language instruction into the thing to edit, the things
   to keep unchanged, and the edit text the renderer receives. Rules, not a language model:
   predictable, instant, and every result is shown in editable fields before use.
2. Grounders turn a phrase into object candidates on the current image. `DinoGrounder`
   (Grounding DINO or MM Grounding DINO, Apache 2.0) returns boxes that SAM turns into masks;
   `Sam3Grounder` (Meta SAM 3 concept prompts) returns masks directly. `FakeGrounder` needs
   no model and is used in tests.

Nothing here decides what gets rendered: the engine shows every proposal on the canvas and
the user keeps it, fixes it with clicks, or replaces it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# ---- parsing --------------------------------------------------------------------------------

DETERMINERS = r"(?:the|a|an|this|that|these|those|his|her|their|my|its|our|your|both|all(?: of)?(?: the)?|every|each)"
_DET_RE = re.compile(rf"^(?:{DETERMINERS}\s+)+", re.I)

# Words that end a target noun phrase after a noun: what the thing should become.
_COLOURS = ("red green blue yellow orange purple violet pink magenta black white grey gray brown beige tan cream gold golden "
            "silver bronze copper navy teal turquoise cyan mint olive maroon burgundy crimson scarlet lavender lilac ivory "
            "khaki charcoal emerald ruby amber coral peach rose").split()
_MODIFIERS = ("dark light bright pale deep pastel matte matt glossy shiny polished brushed metallic transparent translucent "
              "wooden wood leather denim metal steel chrome glass plastic ceramic marble stone brick velvet silk cotton wool "
              "woollen woolen fur furry striped spotted checked plaid floral patterned plain rusty old new vintage modern "
              "colored coloured colour color more less much very slightly bigger smaller larger taller shorter "
              "look looking appear seem like "
              "dramatic sunny cloudy stormy snowy rainy foggy misty clear darker brighter lighter warmer cooler colder hotter "
              "happy happier sad smiling vibrant vivid saturated muted realistic moody cinematic sharper softer cleaner "
              "shorter longer curly straight wavy blonde blond brunette bald younger older").split()
# Verbs that end a target noun phrase: "make the sign say OPEN", "make the man wear a hat".
_STOP_VERBS = set("say says read reads show shows have has wear wears wearing hold holds holding become becomes be is are look looks "
                  "smile smiling laugh laughing frown frowning cry crying wave waving sit sitting stand standing".split())
_STOP_AFTER_NOUN = set(_COLOURS) | set(_MODIFIERS) | _STOP_VERBS | {"to", "into", "in", "with", "as", "so", "and", "a", "an", "the", "from", "on", "at", "for"}
_ADJ_SUFFIXES = ("ic", "ful", "ous", "ive", "ish", "less", "able")
_ATTRIBUTE_WORDS = set(_COLOURS) | set(_MODIFIERS)
# Words that name a colour only when a colour follows: "forest green", "sky blue", "wine red".
_COLOUR_PREFIXES = set("forest lime wine royal baby hot neon jet electric sky ocean sea blood bottle hunter midnight powder dusty dusky rose".split())

BACKGROUND_WORDS = {"background", "backdrop", "surroundings", "scenery", "setting", "environment", "everything else"}
WHOLE_WORDS = {"photo", "image", "picture", "scene", "shot", "whole photo", "whole image", "whole picture", "whole scene", "whole thing", "entire photo",
               "entire image", "lighting", "light", "colours", "colors", "mood", "style", "look", "season", "weather", "time of day", "everything"}
PRONOUNS = {"him": "man", "he": "man", "her": "woman", "she": "woman", "them": "people", "they": "people", "it": None, "this": None, "that": None}

_PROTECT_RE = re.compile(
    r"^(?:but\s+|and\s+|while\s+|please\s+|just\s+|only\s+)*"
    r"(?:keep|keeping|leave|leaving|preserve|preserving|protect|protecting|lock|locking|"
    r"don'?t\s+(?:change|touch|edit|alter|modify)|do\s+not\s+(?:change|touch|edit|alter|modify)|"
    r"without\s+(?:changing|touching|editing|altering|modifying)|not\s+(?:changing|touching))\s+"
    r"(?P<x>.+?)"
    r"(?:\s+(?:unchanged|untouched|intact|alone|as\s+(?:it\s+is|they\s+are|is|before)|the\s+same|exactly\s+as\s+(?:it\s+is|they\s+are)))?"
    r"\s*[.!]?$",
    re.I,
)
_EXCEPT_RE = re.compile(r"^(?P<head>.*?)\b(?:everything|all)\s+(?:except|but|other\s+than|apart\s+from)\s+(?:for\s+)?(?P<x>.+)$", re.I)

_VERBS_REMOVE = r"remove|erase|delete|get\s+rid\s+of|take\s+out|take\s+away|wipe\s+out|wipe|clear\s+away|clean\s+up"
_VERBS_CHANGE = (r"make|turn|change|paint|repaint|colou?r|recolou?r|dye|tint|convert|transform|restyle|swap|switch|"
                 r"give|edit|fix|retouch|brighten|darken|sharpen|blur|smooth|update|redo|modify|alter|shade")
_VERBS_ADD = r"add|place|put|insert|draw|plant|hang|set|drop"

_REMOVE_RE = re.compile(rf"^(?:please\s+)?(?:{_VERBS_REMOVE})\s+(?P<x>.+?)(?:\s+(?:from|off|out\s+of)\s+.+)?$", re.I)
_REPLACE_RE = re.compile(r"^(?:please\s+)?(?:replace|substitute|swap\s+out|swap|switch|exchange|trade)\s+(?P<x>.+?)\s+(?:with|by|for)\s+(?P<y>.+)$", re.I)

# Polite and indirect phrasings, rewritten to the imperative before matching (the model still gets the user's words).
_POLITE_RE = re.compile(r"^(?:(?:hey|hi|ok|okay|so|now|and|also|then)[,\s]+)*(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|please\s+|kindly\s+|"
                        r"i\s+(?:want|need|would\s+like|'d\s+like|d\s+like)\s+(?:you\s+)?to\s+|i'd\s+like\s+(?:you\s+)?to\s+|let'?s\s+|try\s+to\s+|help\s+me\s+)+", re.I)
_WANT_RE = re.compile(r"^(?:i\s+(?:want|need|would\s+like|'d\s+like)|i'd\s+like)\s+(?P<x>.+?)\s+to\s+(?:be|look|become)\s+(?P<y>.+)$", re.I)
_SHOULD_RE = re.compile(r"^(?P<x>(?:the|his|her|their|my|its|our|your|this|that)\s+.+?)\s+(?:should|must|needs?\s+to|has\s+to|ought\s+to)\s+(?:be|look|become)\s+(?P<y>.+)$", re.I)
# A new pose or action moves the whole body (and what is behind it), which an edit confined to
# the person's outline cannot do: these redraw the whole picture from the photo.
_POSE_ACTS = (r"sit(?:ting)?|stand(?:ing)?|lie\s+down|lying|kneel(?:ing)?|jump(?:ing)?|danc(?:e|ing)|run(?:ning)?|walk(?:ing)?|"
              r"wav(?:e|ing)|lean(?:ing)?|squat(?:ting)?|stretch(?:ing)?|turn(?:ing)?\s+(?:around|away|to\s+the\s+(?:left|right|side))|"
              r"look(?:ing)?\s+(?:at|into|towards?)\s+the\s+camera|face\s+the\s+camera|hold(?:ing)?\s+\w+|hug(?:ging)?|point(?:ing)?|"
              r"cross\s+(?:her|his|their)\s+(?:arms|legs)|raise\s+(?:her|his|their)\s+(?:arms?|hands?)|put\s+(?:her|his|their)\s+hands?\s+on|"
              r"with\s+(?:her|his|their)\s+(?:arms|legs)\s+(?:crossed|folded|raised|up|open)|arms\s+(?:crossed|folded|raised)")
_POSE_RE = re.compile(r"\b(?:poses?|posing|posture|stance)\b"
                      rf"|\b(?:make|have|let|get|show)\s+(?:her|him|them|me|us|the\s+(?:woman|man|girl|boy|person|lady|guy|child|kid|baby|dog|cat|people|couple|family|bride|groom|model|kids|children))\s+(?:{_POSE_ACTS})\b"
                      rf"|\b(?:she|he|they|i)\s+(?:is|are|am|should\s+be|to\s+be|was|were)\s+(?:{_POSE_ACTS})\b", re.I)

# Something worn or held by a person: "give her sunglasses", "make him wear a suit", "she should
# wear a red scarf", "dress her in a leather jacket". The person is the edit area and the new
# thing may extend past their outline (the context edge), so a hat above the head is accepted.
_WEAR_RE = re.compile(r"^(?:please\s+)?(?:"
                      r"give\s+(?P<x1>her|him|them|the\s+(?:woman|man|girl|boy|person|lady|guy|child|kid|baby|people))\s+(?P<y1>.+)"
                      r"|(?:make|let|have)\s+(?P<x2>her|him|them|the\s+(?:woman|man|girl|boy|person|lady|guy|child|kid))\s+(?:wear|hold|carry|put\s+on)\s+(?P<y2>.+)"
                      r"|(?P<x3>she|he|they)\s+(?:should|could|can|must|would)\s+(?:wear|hold|carry|have)\s+(?P<y3>.+)"
                      r"|dress\s+(?P<x4>her|him|them|the\s+(?:woman|man|girl|boy|person|lady|guy|child|kid))\s+(?:up\s+)?in\s+(?P<y4>.+)"
                      r")$", re.I)
_SUBJECT_WORD = {"she": "woman", "her": "woman", "he": "man", "him": "man", "they": "person", "them": "person"}

# Edits of the whole photo: "make it look like winter", "turn this into a painting", "make the photo warmer".
_WHOLE_RE = re.compile(r"^(?:(?:make|turn|change|convert|transform|give|edit|fix|retouch|brighten|darken|sharpen|restyle|render|color\s+grade|colou?r\s*grade|grade|update|modify|alter)\s+"
                       r"(?:it|this|everything|the\s+whole\s+(?:thing|photo|image|picture|scene)|the\s+(?:entire\s+)?(?:photo|image|picture|scene|shot|lighting|light|colou?rs|mood|style|look|season|weather|time\s+of\s+day))\b"
                       r"|^(?:give|add)\s+(?:it|this|the\s+(?:photo|image|picture))\s+|^(?:make|turn)\s+(?:it|this)\s+(?:look|feel|seem)\b)", re.I)
_COLOUR_OF_RE = re.compile(r"\b(?:the\s+)?(?:colou?r|shade|tone|hue)\s+of\s+", re.I)


def _normalise(clause: str) -> str:
    c = clause.strip().rstrip("?!.").strip()
    c = re.sub(r"[,\s]+please$", "", c, flags=re.I)
    m = _WANT_RE.match(c) or _SHOULD_RE.match(c)
    if m:
        return f"make {m.group('x')} {m.group('y')}"
    c = _POLITE_RE.sub("", c)
    c = re.sub(r"^(?:and|also|then|now|so|but)\s+", "", c, flags=re.I)
    c = _COLOUR_OF_RE.sub("", c)
    return c
_ADD_RE = re.compile(rf"^(?:please\s+)?(?:{_VERBS_ADD})\s+(?P<y>.+?)\s+(?P<prep>on|onto|to|in|into|at|over|above|under|beneath|below|beside|next\s+to|near|behind|in\s+front\s+of|inside)\s+(?P<x>.+)$", re.I)
_CHANGE_RE = re.compile(rf"^(?:please\s+)?(?:{_VERBS_CHANGE})\s+(?P<x>.+)$", re.I)

_SPATIAL = {
    "left": "left", "leftmost": "left", "right": "right", "rightmost": "right", "top": "top", "upper": "top",
    "bottom": "bottom", "lower": "bottom", "middle": "centre", "center": "centre", "centre": "centre", "central": "centre",
    "largest": "largest", "biggest": "largest", "big": "largest", "smallest": "smallest", "small": "smallest", "tiny": "smallest",
    "front": None,  # "front door" is a name, not a position
}
_SPATIAL_PHRASE_RE = re.compile(r"\s+(?:on|at|in)\s+the\s+(left|right|top|bottom|middle|center|centre)(?:\s+side)?\b", re.I)
_SINGULAR_S = {"glass", "dress", "bus", "grass", "class", "boss", "cross", "mattress", "canvas", "lens", "atlas", "iris", "cactus",
               "status", "ottoman", "bass", "moss", "glasses", "sunglasses", "jeans", "trousers", "pants", "shorts", "scissors", "headphones"}


@dataclass
class Phrase:
    """One thing to find. `text` goes to the grounder; `position` and `plural` rank and group."""

    text: str
    position: str | None = None  # left | right | top | bottom | centre | largest | smallest
    plural: bool = False
    rel: str | None = None  # on | next to | above | below | has: how it relates to `ref`
    ref: "Phrase | None" = None  # "the cup next to the laptop": the laptop
    rel_word: str | None = None  # the word the user wrote ("in", "beside"), for the label

    def label(self) -> str:
        base = (f"{self.position} " if self.position else "") + self.text
        if self.ref is not None:
            base += f" {self.rel_word or self.rel} {self.ref.label()}"
        return base


@dataclass
class Parsed:
    instruction: str
    edit: str  # what the renderer receives: the instruction without the keep clauses
    targets: list[Phrase] = field(default_factory=list)
    protect: list[Phrase] = field(default_factory=list)
    background: bool = False  # the target is everything except the protected objects
    whole: bool = False  # the whole photo is edited ("make it look like winter"), except kept objects
    redraw: bool = False  # a new pose or action: the whole picture is drawn again from the photo
    adds: str | None = None  # the thing being added, when the target is a place
    add_where: str | None = None  # how it relates to the place: "on", "behind", "next to"...
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"edit": self.edit, "targets": [p.label() for p in self.targets], "protect": [p.label() for p in self.protect],
                "background": self.background, "whole": self.whole, "redraw": self.redraw, "adds": self.adds, "notes": self.notes}


def _clauses(text: str) -> list[str]:
    """Split on commas, semicolons, full stops and the joins that start a keep clause."""
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"\s*[,;]\s*|(?<=[a-z])\.\s+|\s+(?=(?:but|while|and)\s+(?:keep(?:ing)?|leav(?:e|ing)|preserv(?:e|ing)|protect(?:ing)?|don'?t|do\s+not|without)\b)"
                     r"|\s+(?=without\s+(?:changing|touching|editing|altering|modifying)\b)"
                     rf"|(?:(?<=unchanged)|(?<=the same)|(?<=untouched)|(?<=intact)|(?<=as it is)|(?<=alone))\s+(?=and\s+(?:{_VERBS_CHANGE}|{_VERBS_REMOVE}|{_VERBS_ADD}|replace)\b)", text, flags=re.I)
    return [p.strip() for p in parts if p and p.strip()]


def _split_list(text: str) -> list[str]:
    """'the faces and the denim jacket' -> two items. 'black and white dog' stays one."""
    pieces = [p.strip() for p in re.split(r"\s*,\s*(?:and\s+|or\s+)?|\s+(?:and|or|&)\s+", text.strip()) if p.strip()]
    merged: list[str] = []
    i = 0
    while i < len(pieces):
        bare = _DET_RE.sub("", pieces[i]).lower()
        if len(bare.split()) == 1 and bare in _COLOURS and i + 1 < len(pieces):
            merged.append(f"{pieces[i]} and {pieces[i + 1]}")
            i += 2
        else:
            merged.append(pieces[i])
            i += 1
    return merged


def _is_plural(text: str, raw: str) -> bool:
    if re.match(r"^(?:both|all|every|each|these|those|the\s+two|two|three|four)\b", raw.strip(), re.I):
        return True
    last = text.split()[-1].lower() if text.split() else ""
    if last in _SINGULAR_S:
        return False
    if last in {"people", "men", "women", "children", "feet", "teeth", "mice", "geese", "leaves", "knives", "shelves"}:
        return True
    return len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us", "is", "ous"))


# "the cup next to the laptop", "the logo on his shirt", "the man with the hat", "the person in the background".
_REL_WORDS = {"on": "on", "in": "on", "inside": "on", "at": "on", "over": "on", "next to": "next to", "near": "next to", "beside": "next to",
              "by": "next to", "behind": "next to", "in front of": "next to", "close to": "next to", "above": "above", "under": "below",
              "underneath": "below", "beneath": "below", "below": "below", "with": "has", "wearing": "has", "holding": "has"}
_REL_RE = re.compile(r"\s+(in\s+front\s+of|next\s+to|close\s+to|on\s+top\s+of|inside|on|in|at|over|near|beside|by|behind|above|underneath|under|beneath|below|with|wearing|holding)\s+(?P<ref>.+)$", re.I)
_DEPTH_WORDS = {"background": "smallest", "distance": "smallest", "back": "smallest", "foreground": "largest", "front": "largest"}
# Keep clauses that only restate the contract: everything outside the edit area is kept anyway.
_IMPLICIT_KEEP = re.compile(r"^(?:(?:the\s+)?rest(?:\s+of\s+(?:the|this|my)\s+(?:photo|image|picture|scene))?|any(?:thing|\s+thing)\s+else|every(?:thing|\s+thing)\s+else|nothing\s+else|all\s+else|everything|the\s+others?|other\s+(?:parts|things))$", re.I)


def _split_relation(raw: str) -> tuple[str, str, str] | None:
    """'the cup next to the laptop' -> ('the cup', 'next to', 'the laptop'). A colour after 'in'
    ('the man in red') or a material after 'with' stays part of the name."""
    m = _REL_RE.search(raw)
    if not m or m.start() == 0:
        return None
    word = re.sub(r"\s+", " ", m.group(1).lower())
    ref = m.group("ref").strip()
    bare = _DET_RE.sub("", ref).lower()
    if word == "on top of":
        word = "on"
    if re.sub(r"\s+side$", "", bare) in _SPATIAL:  # "on the left" is a position, handled by make_phrase
        return None
    if word in {"in", "with"} and all(w in _ATTRIBUTE_WORDS for w in bare.split()):
        return None
    return raw[:m.start()].strip(), word, ref


def make_phrase(raw: str) -> Phrase | None:
    raw = raw.strip().strip(".!?").strip()
    if not raw:
        return None
    rel = _split_relation(raw)
    if rel:
        head, word, ref_raw = rel
        bare = _DET_RE.sub("", ref_raw).strip().lower()
        ph = make_phrase(head)
        if ph is None:
            return None
        if bare in _DEPTH_WORDS and word in {"in", "at", "on"}:
            ph.position = ph.position or _DEPTH_WORDS[bare]  # "the person in the background": the farthest one
            return ph
        ref = make_phrase(ref_raw)
        if ref is not None and ref.text != ph.text:
            ref.ref, ref.rel = None, None  # one level is enough
            ph.rel, ph.ref, ph.rel_word = _REL_WORDS[word], ref, word
        return ph
    low = raw.lower()
    if low in PRONOUNS:
        name = PRONOUNS[low]
        return Phrase(name) if name else None
    position = None
    m = _SPATIAL_PHRASE_RE.search(raw)
    if m:
        position = _SPATIAL[m.group(1).lower()]
        raw = (raw[:m.start()] + raw[m.end():]).strip()
    plural = _is_plural(_DET_RE.sub("", raw), raw)
    text = _DET_RE.sub("", raw).strip()
    words = text.split()
    kept = []
    for w in words:
        pos = _SPATIAL.get(w.lower(), "absent")
        if pos != "absent" and pos is not None and len(words) > 1 and position is None:
            position = pos
            continue
        kept.append(w)
    text = " ".join(kept).strip()
    if not text:
        return None
    return Phrase(text.lower(), position, plural)


def _noun_phrase(rest: str) -> str:
    """For 'make X Y' verbs: X ends at the first noun followed by a word saying what it
    becomes. 'the dark red car pastel mint green' -> 'the dark red car'."""
    rest = re.split(r"\s+(?:to|into|so\s+(?:it|that|they))\s+", rest, maxsplit=1, flags=re.I)[0]
    words = rest.split()
    body_start = 0
    while body_start < len(words) and re.fullmatch(DETERMINERS, words[body_start], re.I):
        body_start += 1
    for i in range(body_start, len(words)):
        w = words[i].lower().strip(".,")
        if w in _ATTRIBUTE_WORDS or w in _STOP_AFTER_NOUN:
            continue
        nxt = words[i + 1].lower().strip(".,") if i + 1 < len(words) else None
        nxt2 = words[i + 2].lower().strip(".,") if i + 2 < len(words) else None
        if nxt in _COLOUR_PREFIXES and nxt2 in _COLOURS:
            return " ".join(words[: i + 1])
        if nxt is not None and nxt.endswith(_ADJ_SUFFIXES) and len(nxt) > 4 and nxt not in {"music", "magic", "plastic", "attic", "fabric", "brick", "public"}:
            return " ".join(words[: i + 1])
        if nxt is None or nxt in _STOP_AFTER_NOUN or nxt.endswith(("ed", "er", "ly")) and nxt not in {"flower", "sweater", "computer", "poster", "heater", "speaker", "counter", "folder", "boulder", "slipper", "trailer", "tower", "container", "printer", "drawer", "shoulder", "corner", "stroller", "bed", "shed", "sled"}:
            return " ".join(words[: i + 1])
    return rest


def parse_instruction(instruction: str) -> Parsed:
    text = (instruction or "").strip()
    out = Parsed(instruction=text, edit=text)
    if not text:
        return out
    clauses = _clauses(text)
    edit_parts: list[str] = []
    for c in clauses:
        m = _PROTECT_RE.match(_POLITE_RE.sub("", c.strip()))
        if m:
            for item in _split_list(m.group("x")):
                if _IMPLICIT_KEEP.match(item.strip()):
                    continue
                ph = make_phrase(item)
                if ph and ph.text not in {p.text for p in out.protect}:
                    out.protect.append(ph)
            continue
        edit_parts.append(c)
    # The renderer gets the request itself: "can you please make her dress red" -> "make her dress red".
    cleaned = [_normalise(c) or c for c in edit_parts]
    out.edit = ", ".join(cleaned) if cleaned else text

    for c in (_normalise(c) for c in edit_parts):
        # "change everything except the dress to red": target is the complement.
        m = _EXCEPT_RE.match(c)
        if m:
            x = re.split(r"\s+(?:to|into|with)\s+", m.group("x"), maxsplit=1, flags=re.I)[0]
            for item in _split_list(x):
                ph = make_phrase(item)
                if ph and ph.text not in {p.text for p in out.protect}:
                    out.protect.append(ph)
            out.background = True
            break
        m = _REPLACE_RE.match(c)
        if m:
            out.targets = [p for p in (make_phrase(i) for i in _split_list(m.group("x"))) if p]
            break
        m = _REMOVE_RE.match(c)
        if m:
            out.targets = [p for p in (make_phrase(i) for i in _split_list(m.group("x"))) if p]
            break
        m = _WEAR_RE.match(c)
        if m:
            who = next(g for g in (m.group("x1"), m.group("x2"), m.group("x3"), m.group("x4")) if g).lower()
            what = next(g for g in (m.group("y1"), m.group("y2"), m.group("y3"), m.group("y4")) if g).strip()
            who = _SUBJECT_WORD.get(who, re.sub(r"^the\s+", "", who))
            out.adds = what
            out.targets = [Phrase(who, None, False)]
            out.notes.append(f"the {who} is the edit area; {what!r} may reach past their outline")
            break
        m = _ADD_RE.match(c)
        if m:
            out.adds = m.group("y").strip()
            out.add_where = re.sub(r"\s+", " ", m.group("prep").lower())
            out.targets = [p for p in (make_phrase(i) for i in _split_list(m.group("x"))) if p]
            out.notes.append(f"the selection is where {out.adds!r} goes")
            break
        m = _CHANGE_RE.match(c)
        if m:
            x = m.group("x")
            # "the colour of the coffee table", "the texture of the bench", "all of the white parts of the truck": the object is what follows
            x = re.sub(r"^(?:all\s+(?:of\s+)?)?(?:the\s+)?(?:(?:colou?r|texture|material|pattern|shade|tone|hue|surface|finish)|(?:\w+\s+)?(?:parts?|areas?|bits?|portions?|sections?))\s+of\s+(?=\S)", "", x, flags=re.I)
            x = re.sub(r"^all\s+(?:of\s+)?(?=the\s)", "", x, flags=re.I)
            # "change X to Y" names X in full, colour words included ("the fluffy white clouds to dark blue")
            explicit_to = bool(re.match(r"^(?:please\s+)?(?:change|turn|convert|transform|switch)\b", c, re.I)) and bool(re.search(r"\s(?:to|into)\s", x, re.I))
            sp = _SPATIAL_PHRASE_RE.search(x)  # "the car on the left red": keep the position, then find the noun
            if sp:
                x = (x[:sp.start()] + x[sp.end():]).strip()
            # "the mug navy blue and the bottle red", "the jacket and the trousers black": one phrase per part.
            parts = re.split(r"\s+and\s+(?=(?:the|his|her|their|my|its|our|your|a|an)\s)", x, flags=re.I)
            out.targets = []
            for part in parts:
                # What follows "to"/"into" is what it becomes ("to a red dress with a white pattern"), never a relation.
                named = re.split(r"(?<!next)(?<!close)\s+(?:to|into)\s+", part, maxsplit=1, flags=re.I)[0]
                rel = _split_relation(named)
                if rel and rel[1] not in {"in", "with"} or rel and re.match(rf"^{DETERMINERS}\s", rel[2], re.I):
                    # "the cup next to the laptop red": the name before the relation, the reference up to what it becomes
                    head, word, rest = rel
                    found = [make_phrase(f"{head} {word} {_noun_phrase(rest)}")]
                elif explicit_to:
                    found = [make_phrase(i) for i in _split_list(re.split(r"\s+from\s+", named, maxsplit=1, flags=re.I)[0])]
                else:
                    found = [make_phrase(i) for i in _split_list(_noun_phrase(part))]
                for p in found:
                    if p and p.text not in {t.text for t in out.targets}:
                        out.targets.append(p)
            if sp and out.targets and out.targets[0].position is None:
                out.targets[0].position = _SPATIAL[sp.group(1).lower()]
            break
        if re.match(rf"^(?:please\s+)?(?:{_VERBS_ADD})\b", c, re.I):
            out.notes.append("say where to add it (for example 'on the table'), or draw the area")
            break
    if any(p.text in BACKGROUND_WORDS or p.text.split()[0] in BACKGROUND_WORDS for p in out.targets):
        out.background = True  # "make the background blurry"
        out.targets = [p for p in out.targets if p.text not in BACKGROUND_WORDS and p.text.split()[0] not in BACKGROUND_WORDS]
    if out.targets and all((t.text in WHOLE_WORDS and t.ref is None or t.text.split()[0] in {"image", "photo", "picture", "scene"}) for t in out.targets) and not out.adds:
        out.targets = []  # "make the photo warmer", "change the lighting to sunset"
        out.whole = True
    if not out.targets and not out.background and not out.adds and _WHOLE_RE.search(" ".join(_normalise(c) for c in edit_parts)):
        out.whole = True
    if _POSE_RE.search(" ".join(_normalise(c) for c in edit_parts)) and not re.match(rf"^(?:please\s+)?(?:{_VERBS_REMOVE})\b", " ".join(edit_parts), re.I):
        out.targets, out.background, out.adds, out.whole, out.redraw = [], False, None, True, True
        out.notes = [n for n in out.notes if "could not tell" not in n and "where to add" not in n]
        out.notes.append("a new pose moves the whole body, so the whole picture is drawn again from your photo; anything you mark to keep stays exact")
    if out.background and not out.protect:
        out.notes.append("background means everything except the kept objects; none were named, so the main subject is taken as 'person'")
        out.protect.append(Phrase("person", None, True))
    if not out.targets and not out.background and not out.whole:
        # "fill the bathroom with water", "I want to see this in a winter scene", "add a monkey riding the ram":
        # nothing names a part, so the model gets the whole photo (kept areas still stay exact)
        adding = any("where to add" in n for n in out.notes)
        out.notes = [n for n in out.notes if "where to add" not in n]
        out.whole = True
        out.notes.append("no place was named, so the model chooses where to add it in the whole photo; say where to add it (for example 'on the table') or draw the area to choose"
                         if adding else "no single part was named, so the whole photo is the edit area; click or name an object to narrow it")
    return out


# ---- grounders ------------------------------------------------------------------------------


@dataclass
class Found:
    box: list[int]  # x0, y0, x1, y1 in canonical pixels
    score: float
    mask: np.ndarray | None = None  # set by grounders that segment (SAM 3)


def box_iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(found: list[Found], iou: float = 0.6) -> list[Found]:
    keep: list[Found] = []
    for f in sorted(found, key=lambda f: -f.score):
        if all(box_iou(f.box, k.box) < iou for k in keep):
            keep.append(f)
    return keep


def rank(found: list[Found], position: str | None, shape: tuple[int, int]) -> list[Found]:
    """Order detections: by the spatial word when there is one, otherwise by score."""
    if not position or len(found) < 2:
        return sorted(found, key=lambda f: -f.score)
    H, W = shape
    cx = lambda f: (f.box[0] + f.box[2]) / 2
    cy = lambda f: (f.box[1] + f.box[3]) / 2
    area = lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1])
    key = {"left": cx, "right": lambda f: -cx(f), "top": cy, "bottom": lambda f: -cy(f),
           "centre": lambda f: abs(cx(f) - W / 2) + abs(cy(f) - H / 2), "largest": lambda f: -area(f), "smallest": area}[position]
    return sorted(found, key=key)


class FakeGrounder:
    """Returns boxes from a fixed table, for tests: {phrase text: [[x0, y0, x1, y1, score], ...]}."""

    name = "fake"

    def __init__(self, table: dict[str, list[list[float]]] | None = None):
        self.table = table or {}
        self.shape: tuple[int, int] | None = None

    def set_image(self, image: np.ndarray) -> None:
        self.shape = image.shape[:2]

    def find(self, text: str) -> list[Found]:
        if text in self.table:
            return [Found([int(v) for v in r[:4]], float(r[4])) for r in self.table[text]]
        if self.table or self.shape is None:
            return []
        H, W = self.shape  # no table: a box chosen by the phrase, for trying the interface without models
        k = sum(text.encode()) % 3
        x0 = [W // 12, W // 3, 7 * W // 12][k]
        return [Found([x0, H // 3, x0 + W // 3, 2 * H // 3], 0.8)]


class DinoGrounder:
    """Grounding DINO or MM Grounding DINO through Transformers' zero-shot detection classes.
    One phrase per query, so scores are comparable across detections of that phrase."""

    def __init__(self, weights_dir: str, device: str = "cuda", threshold: float = 0.1, text_threshold: float = 0.1):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.name = f"dino:{weights_dir.rstrip('/').split('/')[-1]}"
        self.device = device
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(weights_dir, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(weights_dir, local_files_only=True).to(device).eval()
        self._torch = torch
        self.image = None
        self.shape: tuple[int, int] | None = None

    def set_image(self, image: np.ndarray) -> None:
        from PIL import Image

        self.image = Image.fromarray(image, mode="RGB")
        self.shape = image.shape[:2]

    def find(self, text: str) -> list[Found]:
        assert self.image is not None, "set_image first"
        query = text.strip().lower().rstrip(".") + "."
        inputs = self.processor(images=self.image, text=query, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=self.threshold, text_threshold=self.text_threshold,
            target_sizes=[self.shape])[0]
        H, W = self.shape
        found = []
        for box, score in zip(res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy()):
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                found.append(Found([x0, y0, x1, y1], float(score)))
        return nms(found)


class Sam3Grounder:
    """Meta SAM 3 concept prompts: one phrase gives every matching instance with its mask.
    Transformers v5.22.1 (Sam3Model, Sam3Processor, post_process_instance_segmentation). Scores
    below `threshold` are kept as weak alternates, as with DINO; bf16 by default."""

    def __init__(self, weights_dir: str, device: str = "cuda", threshold: float = 0.1, mask_threshold: float = 0.5, dtype: str = "bfloat16"):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.name = "sam3"
        self.dtype = getattr(torch, dtype)
        self.device = device
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.processor = Sam3Processor.from_pretrained(weights_dir, local_files_only=True)
        self.model = Sam3Model.from_pretrained(weights_dir, dtype=self.dtype, local_files_only=True).to(device).eval()
        self._torch = torch
        self.vision = None
        self.sizes = None
        self.shape: tuple[int, int] | None = None

    def set_image(self, image: np.ndarray) -> None:
        from PIL import Image

        inputs = self.processor(images=Image.fromarray(image, mode="RGB"), return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            self.vision = self.model.get_vision_features(pixel_values=inputs.pixel_values.to(self.dtype))
        self.sizes = inputs.get("original_sizes").tolist()
        self.shape = image.shape[:2]

    def find(self, text: str) -> list[Found]:
        assert self.vision is not None, "set_image first"
        text_inputs = self.processor(text=text.strip().lower(), return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            outputs = self.model(vision_embeds=self.vision, **text_inputs)
        res = self.processor.post_process_instance_segmentation(outputs, threshold=self.threshold, mask_threshold=self.mask_threshold,
                                                                target_sizes=self.sizes)[0]
        found = []
        for m, b, s in zip(res["masks"], res["boxes"], res["scores"]):
            mask = np.asarray(m.cpu()).astype(bool)
            if mask.any():
                found.append(Found([int(round(float(v))) for v in b], float(s), mask))
        return sorted(found, key=lambda f: -f.score)


def mask_bbox_iou(mask: np.ndarray, box) -> float:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    return box_iou([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], box)


def segment_found(segmenter, found: list[Found]) -> list[tuple[Found, np.ndarray]]:
    """Box to mask with SAM for grounders that return boxes. Of SAM's three masks, take the
    one that best fills the box (a whole sofa, not one cushion), weighted with SAM's score."""
    from lis.segment import ObjectState

    out = []
    for f in found:
        if f.mask is not None:
            out.append((f, f.mask))
            continue
        cands = segmenter.select(ObjectState(box=list(f.box)))
        if not cands:
            continue
        best = max(cands, key=lambda c: 0.5 * c.score + 0.5 * mask_bbox_iou(c.mask, f.box))
        out.append((f, fill_small_holes(best.mask)))
    return out


def fill_small_holes(mask: np.ndarray, fraction: float = 0.02) -> np.ndarray:
    """Specular highlights punch holes in SAM masks of faces and glossy objects; a protected
    face with holes would let its highlights change. Fill enclosed holes under `fraction` of the area."""
    from scipy import ndimage

    holes = ndimage.binary_fill_holes(mask) & ~mask
    if not holes.any():
        return mask
    lab, n = ndimage.label(holes)
    sizes = ndimage.sum(holes, lab, index=range(1, n + 1))
    small = [i + 1 for i, sz in enumerate(sizes) if sz < fraction * mask.sum()]
    return mask | np.isin(lab, small) if small else mask


CONFIDENT = 0.3  # detections below this are offered as alternates, never chosen ahead of a confident one

# ---- query form and colour check ----------------------------------------------------------

_IRREGULAR = {"people": "person", "men": "man", "women": "woman", "children": "child", "feet": "foot", "teeth": "tooth",
              "mice": "mouse", "geese": "goose", "leaves": "leaf", "knives": "knife", "shelves": "shelf", "wolves": "wolf"}


def singular(text: str) -> str:
    """Detectors are trained on singular class names: 'faces' finds whole people, 'face' finds faces."""
    words = text.split()
    if not words:
        return text
    last = words[-1]
    if last in _IRREGULAR:
        last = _IRREGULAR[last]
    elif last in _SINGULAR_S:
        pass
    elif last.endswith("ies") and len(last) > 4:
        last = last[:-3] + "y"
    elif last.endswith(("ches", "shes", "xes", "sses", "zes")):
        last = last[:-2]
    elif last.endswith("s") and not last.endswith(("ss", "us", "is")):
        last = last[:-1]
    return " ".join(words[:-1] + [last])


def colour_words(text: str) -> list[str]:
    return [w for w in text.split() if w in _COLOURS]


ACHROMATIC = {"black", "charcoal", "white", "ivory", "grey", "gray", "silver"}
PASTEL = {"mint", "lavender", "lilac", "peach", "cream", "ivory", "beige", "khaki"}


def colour_fraction(image: np.ndarray, mask: np.ndarray, colour: str, pastel: bool = False) -> float:
    """Share of the mask's pixels whose hue, saturation and value fit a basic colour name.
    Coarse on purpose: it only has to tell a pink car from a dark red one. For chromatic
    colours the share is taken over pixels bright enough to show a colour (tyres, shadows and
    interiors cannot); pastel colours accept lower saturation (measured on the mint car)."""
    px = image[mask].astype(np.float32) / 255.0
    if not len(px):
        return 0.0
    if len(px) > 50000:
        px = px[np.random.default_rng(0).choice(len(px), 50000, replace=False)]
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    mx, mn = px.max(1), px.min(1)
    v, d = mx, mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    h = np.zeros_like(v)
    nz = d > 1e-6
    rm, gm = nz & (mx == r), nz & (mx == g) & (mx != r)
    bm = nz & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / d[rm]) % 6
    h[gm] = (b[gm] - r[gm]) / d[gm] + 2
    h[bm] = (r[bm] - g[bm]) / d[bm] + 4
    h = h * 60.0
    pastel = pastel or colour in PASTEL
    chroma = (s > (0.12 if pastel else 0.25)) & (v > 0.15)
    hue = lambda a, b_: chroma & ((h >= a) & (h < b_) if a < b_ else ((h >= a) | (h < b_)))
    tests = {
        "red": hue(345, 15), "crimson": hue(340, 10), "scarlet": hue(350, 15), "maroon": hue(340, 15) & (v < 0.55),
        "burgundy": hue(330, 15) & (v < 0.55), "ruby": hue(335, 10),
        "orange": hue(15, 40), "amber": hue(30, 50), "coral": hue(0, 25) & (v > 0.6), "peach": hue(15, 40) & (s < 0.55) & (v > 0.7),
        "yellow": hue(40, 70), "gold": hue(38, 58), "golden": hue(38, 58),
        "green": hue(70, 165), "olive": hue(50, 90) & (v < 0.6), "emerald": hue(130, 170), "mint": hue(130, 200) & (v > 0.5),
        "teal": hue(165, 200), "turquoise": hue(165, 195), "cyan": hue(175, 200),
        "blue": hue(195, 255), "navy": hue(200, 255) & (v < 0.5),
        "purple": hue(255, 295), "violet": hue(255, 295), "lavender": hue(250, 300) & (v > 0.6), "lilac": hue(260, 310) & (v > 0.6),
        "pink": hue(290, 345) | (hue(345, 15) & (s < 0.55) & (v > 0.7)), "magenta": hue(290, 335), "rose": hue(320, 355),
        "brown": hue(5, 45) & (v < 0.65), "tan": hue(20, 45) & (s < 0.6), "beige": (h >= 20) & (h < 55) & (s > 0.08) & (s < 0.4) & (v > 0.55),
        "cream": (h >= 30) & (h < 60) & (s < 0.3) & (v > 0.75), "khaki": hue(40, 65) & (s < 0.55),
        "black": v < 0.2, "charcoal": (v < 0.35) & (s < 0.25), "white": (v > 0.78) & (s < 0.15), "ivory": (v > 0.8) & (s < 0.2),
        "grey": (s < 0.15) & (v >= 0.2) & (v <= 0.8), "gray": (s < 0.15) & (v >= 0.2) & (v <= 0.8),
        "silver": (s < 0.15) & (v > 0.5), "bronze": hue(20, 45) & (v < 0.7), "copper": hue(10, 35),
    }
    t = tests.get(colour)
    if t is None:
        return 0.0
    if colour not in ACHROMATIC:
        lit = v >= 0.2
        if lit.sum() >= 0.2 * len(v):
            return float(t[lit].mean())
    return float(t.mean())


def find_phrase(grounder, segmenter, image: np.ndarray, phrase: Phrase) -> list[dict]:
    """Every instance of the phrase, ranked, each with its mask. Steps: query the singular form;
    when the phrase names a colour, also query the bare noun and rank by how much of each mask
    has that colour (detectors are weak on colour words); then apply the spatial word if any."""
    query = singular(phrase.text)
    found = grounder.find(query)
    colours = colour_words(phrase.text)
    head = query.split()[-1]
    if colours and head != query:
        for f in grounder.find(head):
            same = [k for k in found if box_iou(f.box, k.box) >= 0.6]
            if same:  # the same object: keep the box, take the stronger evidence that it exists
                same[0].score = max(same[0].score, f.score)
            else:
                found.append(f)
    pairs = segment_found(segmenter, found)
    items = [{"found": f, "mask": m, "score": f.score, "colour": None} for f, m in pairs]
    for it in items:
        it["confident"] = it["score"] >= CONFIDENT
    if colours:
        for it in items:
            it["colour"] = float(np.mean([colour_fraction(image, it["mask"], c) for c in colours]))
        items.sort(key=lambda it: (not it["confident"], -(it["colour"] + 0.3 * it["score"])))
    else:
        items.sort(key=lambda it: (not it["confident"], -it["score"]))
    if phrase.position and len(items) > 1:
        order = rank([it["found"] for it in items], phrase.position, image.shape[:2])
        items = sorted(items, key=lambda it: (not it["confident"], next(i for i, f in enumerate(order) if f is it["found"])))
    if phrase.ref is not None and items:
        refs = find_phrase(grounder, segmenter, image, phrase.ref)
        if refs:
            items = relate(items, refs[0]["mask"], phrase.rel)
            for it in items:
                it["ref_found"] = True
    return items


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def relation_score(mask: np.ndarray, ref: np.ndarray, rel: str) -> float:
    """How well one candidate fits 'X <rel> REF', higher is better. on/has: overlap over the
    smaller area (a logo on a shirt, a woman in a dress); next to: closeness of the boxes;
    above/below: on the right side, then close."""
    a, b = _bbox(mask), _bbox(ref)
    if a is None or b is None:
        return -1e9
    inter = float(np.count_nonzero(mask & ref))
    if rel in {"on", "has"}:
        return inter / max(1.0, min(float(mask.sum()), float(ref.sum())))
    diag = float(np.hypot(*mask.shape))
    gx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    gy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    gap = float(np.hypot(gx, gy)) / diag
    if rel == "next to":
        return -gap
    acy, bcy = (a[1] + a[3]) / 2, (b[1] + b[3]) / 2
    side = (bcy - acy) if rel == "above" else (acy - bcy)
    return (0.0 if side > 0 else -1.0) - gap


def relate(items: list[dict], ref: np.ndarray, rel: str | None) -> list[dict]:
    """Order candidates by how well they fit the relation to the reference object. A candidate
    that is the reference itself (the cup next to the cup) goes last."""
    if not rel:
        return items
    def key(it):
        m = it["mask"]
        same = np.count_nonzero(m & ref) / max(1, np.count_nonzero(m | ref)) > 0.8
        return (same, not it["confident"], -relation_score(m, ref, rel))
    return sorted(items, key=key)


def default_pick(items: list[dict], plural: bool, avoid: np.ndarray | None = None, rel: float = 0.6) -> list[dict]:
    """What the proposal uses before the user looks. Candidates mostly inside `avoid` (the
    protected objects, when picking a target) are skipped: a protected target cannot be
    edited. Singular: the first confident candidate, else the best weak one (flagged).
    Plural: every confident candidate scoring at least `rel` times the best."""
    def ok(it):
        if avoid is None or not avoid.any():
            return True
        return (it["mask"] & avoid).sum() < 0.5 * max(1, it["mask"].sum())
    usable = [it for it in items if ok(it)]
    confident = [it for it in usable if it["confident"]]
    if not confident:
        return usable[:1]
    if not plural:
        return confident[:1]
    best = max(it["score"] for it in confident)
    return [it for it in confident if it["score"] >= rel * best]
