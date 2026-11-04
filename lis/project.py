"""The project store.

Owns the canonical source (decoded once, EXIF orientation applied once, RGB uint8), its
sha256, every mask as a PNG in canonical pixel space, and project.json. The source array
is never written to after load.

Layout:
  projects/<project-id>/original.<ext>   untouched upload
  projects/<project-id>/source.png       canonical RGB
  projects/<project-id>/masks/<id>.png   0/255 single channel, plus <id>.json with its prompts
  projects/<project-id>/runs/<rev>/...   per-revision files (owned by engine/history)
  projects/<project-id>/project.json     masks, revisions (tree), head
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import io

import numpy as np
from PIL import Image, ImageCms, ImageOps

try:  # iPhone photos (HEIC/HEIF); optional so the editor still starts without it
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF = True
except Exception:  # noqa: BLE001
    HEIF = False

SCHEMA_VERSION = 1
MAX_PIXELS = 40_000_000  # larger photos are opened scaled down to this, and the user is told
MIN_SIDE = 16
Image.MAX_IMAGE_PIXELS = 400_000_000  # decode big photos (then scale them); refuse only absurd ones


class PhotoError(ValueError):
    """A file that cannot become a project, with a message for the user."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _to_rgb(im: Image.Image) -> Image.Image:
    """RGB uint8 from any mode: transparency over white (not black), 16-bit scaled, not clipped."""
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(im).astype(np.float64)
        top = 65535.0 if a.max() > 255 else 255.0
        a = np.clip(a / top * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return Image.fromarray(a, mode="L").convert("RGB")
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA", "PA", "RGBa", "La"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, im).convert("RGB")
    return im if im.mode == "RGB" else im.convert("RGB")


def decode_canonical(path: Path) -> np.ndarray:
    """Decode once, apply EXIF orientation once, drop alpha, return RGB uint8 (H, W, 3)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.asarray(_to_rgb(im), dtype=np.uint8).copy()


def decode_photo(path: Path) -> tuple[np.ndarray, list[str]]:
    """A user's photo as the canonical RGB array, with notes on what was done to it: EXIF
    orientation, the embedded colour profile converted to sRGB (iPhone photos are Display P3),
    transparency over white, 16-bit scaled to 8, and photos above MAX_PIXELS scaled down."""
    notes: list[str] = []
    try:
        im = Image.open(path)
        im.load()
    except Exception as e:  # noqa: BLE001
        ext = Path(path).suffix.lower()
        if ext in (".heic", ".heif") and not HEIF:
            raise PhotoError("HEIC photos need the pillow-heif package: run scripts/install.sh again.") from e
        raise PhotoError("This file is not an image the editor can read.") from e
    with im:
        if getattr(im, "n_frames", 1) > 1:
            notes.append("the file has several frames; the first is used")
        im = ImageOps.exif_transpose(im)
        icc = im.info.get("icc_profile")
        if icc and im.mode in ("RGB", "RGBA", "CMYK"):
            try:
                src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                name = ImageCms.getProfileDescription(src).strip()
                if "srgb" not in name.lower():
                    im = ImageCms.profileToProfile(im if im.mode != "RGBA" else im.convert("RGB"), src, ImageCms.createProfile("sRGB"),
                                                   outputMode="RGB")
                    notes.append(f"colours converted from {name} to sRGB")
            except Exception:  # noqa: BLE001  a broken profile: keep the pixels as they are
                pass
        rgb = _to_rgb(im)
    w, h = rgb.size
    if min(w, h) < MIN_SIDE:
        raise PhotoError(f"The photo is {w}x{h}; the editor needs at least {MIN_SIDE} pixels on each side.")
    if w * h > MAX_PIXELS:
        k = (MAX_PIXELS / float(w * h)) ** 0.5
        nw, nh = max(MIN_SIDE, int(w * k)), max(MIN_SIDE, int(h * k))
        rgb = rgb.resize((nw, nh), Image.LANCZOS)
        notes.append(f"opened at {nw}x{nh} (from {w}x{h}); the editor's limit is {MAX_PIXELS // 1_000_000} megapixels")
    return np.asarray(rgb, dtype=np.uint8).copy(), notes


def load_mask_png(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        m = np.asarray(im.convert("L"))
    if m.shape != shape:
        raise ValueError(f"mask {path.name} is {m.shape}, canonical is {shape}")
    return m >= 128


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    if mask.dtype != bool or mask.ndim != 2:
        raise ValueError("mask must be a bool (H, W) array")
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(path, format="PNG", compress_level=3)


def save_rgb_png(img: np.ndarray, path: Path) -> None:
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("image must be uint8 (H, W, 3)")
    # Lossless at any level; 3 writes a 24 MP photo in 2 s instead of 5 s for about 11 percent more bytes.
    Image.fromarray(img, mode="RGB").save(path, format="PNG", compress_level=3)  # Pillow writes no EXIF here


@dataclass
class MaskRecord:
    mask_id: str
    kind: str  # "target" | "protected", fixed at creation
    name: str
    prompts: dict = field(default_factory=dict)  # clicks, boxes, brush delta file names
    sha256: str = ""


@dataclass
class ProjectRecord:
    schema_version: int
    project_id: str
    width: int
    height: int
    original_name: str
    source_sha256: str
    masks: list[MaskRecord] = field(default_factory=list)
    revisions: list[dict] = field(default_factory=list)  # owned by history.py
    head: str | None = None  # None means the canonical source is shown
    import_notes: list[str] = field(default_factory=list)  # what opening did to the photo (scaled, colour converted)
    sample: str | None = None  # which start-page sample this is a copy of, so an untouched copy is reused


class Project:
    """One opened photo and everything derived from it, on disk under `root`."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.record = self._read()
        self._source: np.ndarray | None = None

    # ---- creation and persistence -------------------------------------------------

    @classmethod
    def create(cls, projects_dir: Path, image_path: Path) -> "Project":
        image_path = Path(image_path)
        project_id = new_id("p")
        root = Path(projects_dir) / project_id
        (root / "masks").mkdir(parents=True, exist_ok=False)
        (root / "runs").mkdir()
        shutil.copy2(image_path, root / f"original{image_path.suffix.lower()}")
        try:
            source, notes = decode_photo(image_path)
        except PhotoError:
            shutil.rmtree(root, ignore_errors=True)
            raise
        save_rgb_png(source, root / "source.png")
        record = ProjectRecord(
            schema_version=SCHEMA_VERSION,
            project_id=project_id,
            width=int(source.shape[1]),
            height=int(source.shape[0]),
            original_name=image_path.name,
            source_sha256=sha256_file(root / "source.png"),
            import_notes=notes,
        )
        (root / "project.json").write_text(json.dumps(asdict(record), indent=2))
        return cls(root)

    def _read(self) -> ProjectRecord:
        data = json.loads((self.root / "project.json").read_text())
        data["masks"] = [MaskRecord(**m) for m in data.get("masks", [])]
        return ProjectRecord(**data)

    def save(self) -> None:
        tmp = self.root / "project.json.tmp"
        tmp.write_text(json.dumps(asdict(self.record), indent=2))
        tmp.replace(self.root / "project.json")

    # ---- canonical source and bases ------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return (self.record.height, self.record.width)

    def source(self) -> np.ndarray:
        """The canonical source, read once, returned read-only."""
        if self._source is None:
            arr = decode_canonical(self.root / "source.png")
            arr.setflags(write=False)
            self._source = arr
        return self._source

    def base_path(self, revision_id: str | None) -> Path:
        """The image a revision builds on: a parent's out.png, or the canonical source."""
        if revision_id is None:
            return self.root / "source.png"
        return self.root / "runs" / revision_id / "out.png"

    def base(self, revision_id: str | None) -> np.ndarray:
        return decode_canonical(self.base_path(revision_id))

    # ---- masks ---------------------------------------------------------------------

    def add_mask(self, mask: np.ndarray, kind: str, name: str, prompts: dict | None = None) -> MaskRecord:
        if kind not in ("target", "protected"):
            raise ValueError("kind must be target or protected")
        if mask.shape != self.shape:
            raise ValueError(f"mask is {mask.shape}, canonical is {self.shape}")
        mask_id = new_id("m")
        path = self.root / "masks" / f"{mask_id}.png"
        save_mask_png(mask, path)
        rec = MaskRecord(mask_id=mask_id, kind=kind, name=name, prompts=prompts or {}, sha256=sha256_file(path))
        (self.root / "masks" / f"{mask_id}.json").write_text(json.dumps(asdict(rec), indent=2))
        self.record.masks.append(rec)
        self.save()
        return rec

    def update_mask(self, mask_id: str, mask: np.ndarray, prompts_update: dict | None = None) -> MaskRecord:
        rec = self.mask_record(mask_id)
        path = self.root / "masks" / f"{mask_id}.png"
        save_mask_png(mask, path)
        rec.sha256 = sha256_file(path)
        if prompts_update:
            rec.prompts.update(prompts_update)
        (self.root / "masks" / f"{mask_id}.json").write_text(json.dumps(asdict(rec), indent=2))
        self.save()
        return rec

    def mask_record(self, mask_id: str) -> MaskRecord:
        for m in self.record.masks:
            if m.mask_id == mask_id:
                return m
        raise KeyError(mask_id)

    def mask(self, mask_id: str) -> np.ndarray:
        self.mask_record(mask_id)  # raises KeyError for unknown ids
        return load_mask_png(self.root / "masks" / f"{mask_id}.png", self.shape)

    def union(self, mask_ids: list[str]) -> np.ndarray:
        out = np.zeros(self.shape, dtype=bool)
        for mid in mask_ids:
            out |= self.mask(mid)
        return out
