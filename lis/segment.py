"""Selection module (in-process, not a separate process): SAM click and box prompts.

Named segment, not select, because `select` is a Python standard library module.
Computes image embeddings once per base and caches them so each click round is cheap.
Returns candidate masks with scores in canonical pixel space. Backend and checkpoint come
from config; `FakeSegmenter` needs no model and is used for tests and UI work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

MIN_SCORE = 0.3
HOLE_FRACTION = 0.005


@dataclass
class Candidate:
    mask: np.ndarray  # bool (H, W)
    score: float
    logits: np.ndarray | None = None  # low-resolution logits for refinement, model space


@dataclass
class ObjectState:
    """Accumulated prompts for one in-progress selection."""

    points: list[list[int]] = field(default_factory=list)  # [x, y]
    labels: list[int] = field(default_factory=list)  # 1 positive, 0 negative
    box: list[int] | None = None  # x0, y0, x1, y1
    logits: np.ndarray | None = None  # picked candidate's logits, fed back next round

    def prompts(self) -> dict:
        return {"clicks": [[*p, l] for p, l in zip(self.points, self.labels)], "box": self.box}


def cleanup(mask: np.ndarray, points: list[list[int]], labels: list[int]) -> np.ndarray:
    """Keep the largest 8-connected component unless positive clicks fall in several; fill
    holes below HOLE_FRACTION of the mask area unless a negative click lies inside the hole."""
    if not mask.any():
        return mask
    structure = np.ones((3, 3), bool)
    comp, n = ndimage.label(mask, structure=structure)
    if n > 1:
        pos_ids = {int(comp[y, x]) for (x, y), l in zip(points, labels) if l == 1 and 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]}
        pos_ids.discard(0)
        if len(pos_ids) <= 1:
            sizes = ndimage.sum(mask, comp, index=range(1, n + 1))
            keep = int(np.argmax(sizes)) + 1 if not pos_ids else next(iter(pos_ids))
            mask = comp == keep
        else:
            mask = np.isin(comp, list(pos_ids))
    holes, hn = ndimage.label(~mask, structure=structure)
    if hn > 1:
        area = int(mask.sum())
        hole_sizes = ndimage.sum(~mask, holes, index=range(1, hn + 1))
        neg_ids = {int(holes[y, x]) for (x, y), l in zip(points, labels) if l == 0 and 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]}
        # Background touching the border is not a hole.
        border = set(np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))) - {0}
        fill = [i + 1 for i, s in enumerate(hole_sizes) if (i + 1) not in border and (i + 1) not in neg_ids and s < HOLE_FRACTION * area]
        if fill:
            mask = mask | np.isin(holes, fill)
    return mask


class FakeSegmenter:
    """A disc of radius 80 around the first positive click, or the box's inscribed ellipse."""

    backend = "fake"

    def __init__(self):
        self.shape: tuple[int, int] | None = None

    def set_image(self, image: np.ndarray) -> None:
        self.shape = image.shape[:2]

    def select(self, state: ObjectState) -> list[Candidate]:
        assert self.shape is not None
        H, W = self.shape
        yy, xx = np.ogrid[:H, :W]
        if state.box is not None:
            x0, y0, x1, y1 = state.box
            cy, cx, ry, rx = (y0 + y1) / 2, (x0 + x1) / 2, max(1, (y1 - y0) / 2), max(1, (x1 - x0) / 2)
            mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        else:
            pos = [p for p, l in zip(state.points, state.labels) if l == 1]
            if not pos:
                return []
            x, y = pos[0]
            mask = (yy - y) ** 2 + (xx - x) ** 2 <= 80 * 80
        for (x, y), l in zip(state.points, state.labels):
            if l == 0:
                mask &= (yy - y) ** 2 + (xx - x) ** 2 > 30 * 30
        return [Candidate(mask, 0.9), Candidate(mask & ((yy - (state.points[0][1] if state.points else H // 2)) <= 0), 0.6)]


class SamSegmenter:
    """Transformers Sam2Model or Sam3TrackerModel; same API (verified at v5.22.1)."""

    def __init__(self, weights_dir: str, backend: str = "sam2", device: str = "cuda"):
        import torch

        self.backend = backend
        self.device = device
        self.dtype = torch.bfloat16
        if backend == "sam2":
            from transformers import Sam2Model as Model, Sam2Processor as Processor
        elif backend == "sam3_tracker":
            from transformers import Sam3TrackerModel as Model, Sam3TrackerProcessor as Processor
        else:
            raise ValueError(f"unknown segmenter backend {backend!r}")
        self.model = Model.from_pretrained(weights_dir, dtype=self.dtype, local_files_only=True).to(device).eval()
        self.processor = Processor.from_pretrained(weights_dir, local_files_only=True)
        self.embeddings = None
        self.original_sizes = None
        self.shape: tuple[int, int] | None = None

    def set_image(self, image: np.ndarray) -> None:
        import torch
        from PIL import Image

        pil = Image.fromarray(image, mode="RGB")
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.embeddings = self.model.get_image_embeddings(inputs["pixel_values"].to(self.dtype))
        self.original_sizes = inputs["original_sizes"]
        self.shape = image.shape[:2]

    def select(self, state: ObjectState) -> list[Candidate]:
        import torch

        assert self.embeddings is not None, "set_image first"
        kwargs: dict = {}
        if state.points:
            kwargs["input_points"] = [[state.points]]  # (1 image, 1 object, N points, 2)
            kwargs["input_labels"] = [[state.labels]]
        if state.box is not None:
            kwargs["input_boxes"] = [[state.box]]  # (1 image, 1 object, 4)
        if not kwargs:
            return []
        inputs = self.processor(images=None, original_sizes=self.original_sizes, return_tensors="pt", **kwargs).to(self.device)
        model_kwargs = {k: v for k, v in inputs.items() if k in ("input_points", "input_labels", "input_boxes")}
        if "input_points" in model_kwargs:
            model_kwargs["input_points"] = model_kwargs["input_points"].to(self.dtype)
        if "input_boxes" in model_kwargs:
            model_kwargs["input_boxes"] = model_kwargs["input_boxes"].to(self.dtype)
        if state.logits is not None:
            m = torch.as_tensor(state.logits)
            while m.dim() < 4:  # the picked candidate is (h, w); the model wants (images, objects, h, w)
                m = m[None]
            model_kwargs["input_masks"] = m.to(self.device, self.dtype)
        with torch.no_grad():
            out = self.model(image_embeddings=self.embeddings, multimask_output=True, **model_kwargs)
        scores = out.iou_scores.float().cpu().numpy().reshape(-1)  # (3,)
        low = out.pred_masks.float().cpu()  # (1, 1, 3, h, w)
        masks = self.processor.post_process_masks(low, self.original_sizes.cpu())[0]  # (1, 3, H, W) bool
        masks = masks.numpy().reshape(-1, *self.shape)
        cands = [Candidate(cleanup(masks[i], state.points, state.labels), float(scores[i]), low[0, 0, i].numpy())
                 for i in np.argsort(-scores) if scores[i] >= MIN_SCORE]
        return cands
