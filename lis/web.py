"""The editor: a small local web server (FastAPI) and a single-page front end in lis/static.

The page talks only to this server, the server only to lis.engine. Nothing is fetched from
the internet: no fonts, scripts or images from a CDN. The server binds to 127.0.0.1 unless
--host is given.

Per-project UI state (the selection list in use) lives in the project folder as ui.json, so
a reload or a restart reopens the same work.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import secrets
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from lis import adjust as adjust_ops
from lis.engine import Engine, EngineBusy
from lis.ground import parse_instruction
from lis.project import PhotoError
from lis.project import save_mask_png

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 80 * 1024 * 1024
VIEW_MAX = 4096  # long side of the sharpest copy the page shows (when zoomed in); the engine, checks and export use full resolution
VIEW_FIT = 2048  # long side of the copy shown first: enough for a screen at fit, a fraction of the bytes over a network
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
# Photos to try the editor on, with a sentence that works on each (samples/, in the repository).
SAMPLES = [
    {"id": "cars", "path": "samples/cars.jpg", "say": "make the pink car matte black, keep the dark red car unchanged"},
    {"id": "living_room", "path": "samples/living_room.jpg", "say": "remove the potted plant and its pot, keep the sofa unchanged"},
    {"id": "street_dress", "path": "samples/street_dress.jpg", "say": "replace the background with a sunny seaside promenade with palm trees, keep the woman unchanged"},
    {"id": "friends_park", "path": "samples/friends_park.jpg", "say": "make the red hoodie forest green, keep the faces unchanged"},
]


def _safe(s: str) -> str:
    if not s or not SAFE_ID.match(s) or ".." in s:
        raise HTTPException(400, "bad id")
    return s


class Job:
    def __init__(self, project_id: str):
        self.id = uuid.uuid4().hex[:12]
        self.project_id = project_id
        self.events: list[dict] = []
        self.done = False
        self.result: dict | None = None
        self.cancel = threading.Event()
        self.started = time.time()
        self.variants: list[str] = []  # revision ids, when one job renders several seeds
        self.many = False  # the job was asked for several versions
        self.kind = "render"  # or "generate": new pictures from words, not tied to a project
        self.images: list[dict] = []  # generate: the pictures made so far
        self.prompt = ""


class Studio:
    """Holds the engine (built in the background so the page opens at once), the jobs and
    the per-project selection lists."""

    def __init__(self, engine_factory, projects: Path):
        self.projects = Path(projects)
        self._areas: dict = {}
        self.engine: Engine | None = None
        self.status = {"state": "starting", "message": "Loading the models. This takes about a minute the first time."}
        self.lock = threading.Lock()  # one engine call at a time (SAM and the grounder are not thread-safe)
        self.prefetching = threading.Event()  # set while a background prefetch holds the lock
        self.jobs: dict[str, Job] = {}
        self.info: dict = {}
        self._factory = engine_factory

    # ---- start-up ---------------------------------------------------------------------

    def start(self, background: bool = True) -> None:
        def run():
            try:
                eng = self._factory()
                self.status = {"state": "starting", "message": "Loading the editing model on the GPU."}
                ready = eng.client.start()
                self.engine = eng
                self.info = {"grounder": eng.cfg.grounder, "segmenter": eng.cfg.segmenter, "profile": eng.cfg.profile_id,
                             "small_gpu": eng.cfg.park_selection,
                             "worker": ready if isinstance(ready, dict) else {}}
                self.status = {"state": "ready", "message": "Ready"}
            except Exception as e:  # shown on the page, never hidden
                self.status = {"state": "error", "message": f"Could not start: {e}"}
                print(f"start failed: {e!r}", file=sys.stderr)

        if background:
            threading.Thread(target=run, daemon=True).start()
        else:
            run()

    def eng(self) -> Engine:
        if self.engine is None:
            raise HTTPException(503, self.status["message"])
        return self.engine

    def idle(self, pid: str) -> None:
        if any(j.project_id == pid and not j.done for j in self.jobs.values()):
            raise HTTPException(409, "An edit is running on this photo. Change the selection when it finishes.")

    def prefetch(self, pid: str) -> None:
        """Compute the selection models' image features in the background, so the first click
        or Find on this image does not wait for them. Skipped when the editor is busy."""
        def run():
            if self.engine is None or not self.lock.acquire(timeout=0.5):
                return
            self.prefetching.set()
            try:
                self.engine.prefetch(pid)
            except Exception as e:  # noqa: BLE001  (a missed prefetch only costs the old wait)
                print(f"prefetch skipped: {e!r}", file=sys.stderr)
            finally:
                self.prefetching.clear()
                self.lock.release()

        threading.Thread(target=run, daemon=True).start()

    def locked(self, fn, *a, **kw):
        # A background prefetch is work the call would have done itself: wait for it, never 409.
        if not self.lock.acquire(timeout=1.0) and not (self.prefetching.is_set() and self.lock.acquire(timeout=30.0)):
            raise HTTPException(409, "Busy: an edit is running or a file is being saved. Try again when it finishes.")
        try:
            return fn(*a, **kw)
        except EngineBusy as e:
            raise HTTPException(409, str(e))
        except KeyError as e:
            raise HTTPException(404, f"not found: {e}")
        finally:
            self.lock.release()

    # ---- per-project UI state -----------------------------------------------------------

    def proj(self, pid: str):
        try:
            return self.eng()._project(_safe(pid))
        except KeyError:
            raise HTTPException(404, "no such project")

    def ui_state(self, pid: str) -> dict:
        f = self.projects / _safe(pid) / "ui.json"
        if f.exists():
            return json.loads(f.read_text())
        return {"objects": [], "instruction": "", "seed": 1234}

    def save_ui(self, pid: str, s: dict) -> None:
        f = self.projects / _safe(pid) / "ui.json"
        f.write_text(json.dumps(s, indent=2))

    def project_view(self, pid: str) -> dict:
        p = self.proj(pid)
        s = self.ui_state(pid)
        known = {m.mask_id: m for m in p.record.masks}
        objects = []
        for o in s["objects"]:
            m = known.get(o["mask_id"])
            if m is None:
                continue
            key = (pid, m.mask_id, m.sha256)
            area = self._areas.get(key)
            if area is None:  # decoding a 24 MP mask on every view was slow; a mask's area never changes for a given hash
                area = self._areas[key] = int(np.count_nonzero(np.asarray(Image.open(p.root / "masks" / f"{m.mask_id}.png").convert("L")) >= 128))
            objects.append({"mask_id": m.mask_id, "kind": m.kind, "name": m.name, "area": area,
                            "url": f"/api/projects/{pid}/mask/masks/{m.mask_id}.png?v={m.sha256[:10]}",
                            "from_words": bool((m.prompts or {}).get("text"))})
        revs = []
        for r in p.record.revisions:
            revs.append(revision_view(pid, r))
        head = p.record.head
        job = next((j for j in self.jobs.values() if j.project_id == pid and not j.done and j.kind == "render"), None)
        return {"project_id": pid, "name": p.record.original_name, "width": p.record.width, "height": p.record.height,
                "head": head, "image_url": image_url(pid, head), "png_url": png_url(pid, head), "objects": objects, "revisions": revs,
                "instruction": s.get("instruction", ""), "seed": s.get("seed", 1234), "sel_for": s.get("sel_for"), "running_job": job.id if job else None,
                "import_notes": list(getattr(p.record, "import_notes", []) or [])}


def image_url(pid: str, rev: str | None) -> str:
    """What the page shows first: a JPEG at most VIEW_FIT pixels on the long side, so photos
    arrive quickly over a network; the page asks for /full/ (VIEW_MAX, quality 95) when zoomed in.
    Export and every check use the exact PNG (png_url)."""
    return f"/api/projects/{pid}/view/{rev or 'source'}.jpg"


def png_url(pid: str, rev: str | None) -> str:
    return f"/api/projects/{pid}/image/{rev or 'source'}.png"


def _op_words(op: dict) -> str:
    if op.get("kind") == "move":
        dx, dy = (op.get("offset") or [0, 0])[:2]
        return f"Move {abs(dx)} px {'right' if dx >= 0 else 'left'}, {abs(dy)} px {'down' if dy >= 0 else 'up'}"
    return op.get("instruction", "")


def revision_view(pid: str, r: dict) -> dict:
    op = r.get("operation", {})
    return {"revision_id": r["revision_id"], "parent": r["parent"], "instruction": _op_words(op),
            "image_url": image_url(pid, r["revision_id"]), "png_url": png_url(pid, r["revision_id"]), "thumb_url": f"/api/projects/{pid}/thumb/{r['revision_id']}.jpg",
            "parent_url": image_url(pid, r["parent"]), "change_map_url": f"/api/projects/{pid}/changemap/{r['revision_id']}.jpg",
            "checks": checks_view(r), "seed": op.get("seed"), "created_at": r.get("created_at"),
            "seconds": round((r.get("timing_ms") or {}).get("total", 0) / 1000, 1), "timing_ms": r.get("timing_ms") or {},
            "worker_size": (r.get("working_region") or {}).get("worker_size"), "scale": (r.get("working_region") or {}).get("scale"),
            "blend": op.get("blend_used"), "strategy": op.get("strategy_used"), "kind": op.get("kind")}


def checks_view(r: dict) -> dict:
    """The three checks in plain words, for the result card."""
    pc = r.get("protected_check") or {}
    oc = r.get("outside_check") or {}
    ec = r.get("edit_check") or {}
    rows = []
    if pc.get("status") == "not_applicable":
        rows.append({"key": "protected", "status": "info", "title": "Kept areas", "detail": "None marked. Mark what must not change to have it proven."})
    else:
        ok = pc.get("status") == "pass"
        rows.append({"key": "protected", "status": "pass" if ok else "fail", "title": "Kept areas unchanged" if ok else "Kept areas changed",
                     "detail": f"{pc.get('mismatched_pixels', 0):,} pixels differ, checked on the saved file"})
    if oc.get("status") in ("pass", "fail"):
        ok = oc["status"] == "pass"
        rows.append({"key": "outside", "status": "pass" if ok else "fail", "title": "Nothing changed outside the edit area" if ok else "Pixels changed outside the edit area",
                     "detail": f"{oc.get('mismatched_pixels', 0):,} pixels outside differ; {oc.get('changed_fraction', 0) * 100:.1f}% of the photo changed"})
    if (r.get("operation") or {}).get("kind") == "adjust":  # no model ran: say so instead of an edit check
        rows.append({"key": "edit", "status": "info", "title": "No model used",
                     "detail": "A fixed formula computed on this computer. Nothing was generated, so there is no edit check."})
        overall = "fail" if any(x["status"] == "fail" for x in rows) else "pass"
        return {"overall": overall, "rows": rows}
    est = ec.get("status", "not_checked")
    title = {"verified": "Edit verified", "not_detected": "Edit not detected", "changed": "The edit area changed",
             "unchanged": "The edit area did not change", "not_checked": "Edit not checked"}.get(est, "Edit not checked")
    st = {"verified": "pass", "not_detected": "warn", "unchanged": "warn", "changed": "info"}.get(est, "info")
    rows.append({"key": "edit", "status": st, "title": title, "detail": ec.get("message") or ""})
    overall = "fail" if any(x["status"] == "fail" for x in rows) else ("warn" if any(x["status"] == "warn" for x in rows) else "pass")
    return {"overall": overall, "rows": rows}


def decode_png_mask(data_url: str, shape: tuple[int, int]) -> np.ndarray:
    raw = base64.b64decode(data_url.split(",", 1)[1] if "," in data_url else data_url)
    im = Image.open(io.BytesIO(raw))
    a = np.asarray(im.convert("RGBA"))
    m = (a[..., 3] > 0) & (a[..., :3].max(-1) > 0)
    if m.shape != shape:
        m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)) >= 128
    return m


LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}


def create_app(studio: Studio, access_key: str | None = None) -> FastAPI:
    """`access_key`: with --lan, every device needs the key once (it then gets a cookie). Without
    it, only requests addressed to this machine's own name are answered, so a web page elsewhere
    cannot reach the editor through DNS rebinding and read the photos."""
    app = FastAPI(title="OpenImage Studio", docs_url=None, redoc_url=None, openapi_url=None)
    S = studio

    @app.middleware("http")
    async def no_store_api(request: Request, call_next):
        host = (request.headers.get("host") or "").lower()
        hostname = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        client = request.client.host if request.client else ""
        local = hostname in LOCAL_HOSTS and client in ("127.0.0.1", "::1", "testclient")
        if access_key is None:
            if hostname not in LOCAL_HOSTS and hostname != "testserver":
                return PlainTextResponse("OpenImage Studio answers only at http://127.0.0.1 on this computer.", status_code=403)
        elif not local:
            given = request.query_params.get("key") or request.cookies.get("lis_key")
            if not given or not secrets.compare_digest(given, access_key):
                return HTMLResponse("<!doctype html><meta name=viewport content='width=device-width'><body style='font:16px system-ui;padding:24px;"
                                    "background:#11141a;color:#e8ebf1'>Open the address printed in the terminal where OpenImage Studio "
                                    "is running (it ends with <b>?key=...</b>).</body>", status_code=403)
            if request.query_params.get("key"):
                resp = await call_next(request)
                resp.set_cookie("lis_key", access_key, httponly=True, samesite="strict", max_age=30 * 86400)
                return resp
        resp = await call_next(request)
        if request.url.path.startswith("/api/") and "cache-control" not in resp.headers:
            resp.headers["Cache-Control"] = "no-store"
        if request.url.path.startswith("/static/"):  # always revalidate: an updated editor must not run yesterday's script
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    @app.get("/", response_class=HTMLResponse)
    def index():
        # Each script and stylesheet URL carries its file's modification time, so a new version is fetched at once.
        html = (STATIC / "index.html").read_text()
        for name in ("app.js", "app.css", "icon.svg"):
            html = html.replace(f"/static/{name}\"", f"/static/{name}?v={int((STATIC / name).stat().st_mtime)}\"")
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def cached_jpeg(src: Path, dst: Path, size: int | None, quality: int) -> Path:
        if not dst.exists():
            dst.parent.mkdir(exist_ok=True)
            im = Image.open(src).convert("RGB")
            if size and max(im.size) > size:
                im.thumbnail((size, size), Image.LANCZOS)
            tmp = dst.with_suffix(".tmp")
            im.save(tmp, format="JPEG", quality=quality, subsampling=0 if quality >= 90 else 2)
            tmp.replace(dst)
        return dst

    @app.get("/api/status")
    def status():
        return {**S.status, **S.info}

    # ---- projects -------------------------------------------------------------------

    @app.get("/api/samples")
    def samples():
        return [{"id": x["id"], "say": x["say"], "thumb_url": f"/api/samples/{x['id']}.jpg"} for x in SAMPLES if (ROOT / x["path"]).exists()]

    @app.get("/api/samples/{sid}.jpg")
    def sample_thumb(sid: str):
        x = next((x for x in SAMPLES if x["id"] == sid), None)
        if x is None:
            raise HTTPException(404, "no such sample")
        return FileResponse(cached_jpeg(ROOT / x["path"], S.projects / ".samples" / f"{sid}.jpg", 360, 85), media_type="image/jpeg", headers=IMMUTABLE)

    @app.post("/api/samples/{sid}")
    def open_sample(sid: str):
        x = next((x for x in SAMPLES if x["id"] == sid), None)
        if x is None:
            raise HTTPException(404, "no such sample")
        # Reopen an untouched copy of this sample instead of making another one each time.
        name = Path(x["path"]).name
        if S.projects.exists():
            for d in S.projects.iterdir():
                f = d / "project.json"
                try:
                    rec = json.loads(f.read_text())
                except Exception:  # noqa: BLE001
                    continue
                st = S.ui_state(rec["project_id"]) if (d / "ui.json").exists() else {}
                if (rec.get("original_name") == name and not rec.get("revisions") and not st.get("objects")
                        and rec.get("sample") == sid):
                    st["instruction"] = x["say"]
                    S.save_ui(rec["project_id"], st)
                    S.prefetch(rec["project_id"])
                    return S.project_view(rec["project_id"])
        o = S.eng().open(ROOT / x["path"], embed=False)
        S.save_ui(o["project_id"], {"objects": [], "instruction": x["say"], "seed": 1234})
        pj = S.proj(o["project_id"])
        pj.record.sample = sid
        pj.save()
        S.prefetch(o["project_id"])
        return S.project_view(o["project_id"])

    @app.get("/api/projects")
    def recent():
        out = []
        if S.projects.exists():
            for d in S.projects.iterdir():
                f = d / "project.json"
                if not f.exists():
                    continue
                try:
                    rec = json.loads(f.read_text())
                except Exception:
                    continue
                out.append({"project_id": rec["project_id"], "name": rec.get("original_name", ""), "width": rec["width"], "height": rec["height"],
                            "edits": len(rec.get("revisions", [])), "modified": f.stat().st_mtime,
                            "thumb_url": f"/api/projects/{rec['project_id']}/thumb/{rec.get('head') or 'source'}.jpg"})
        out.sort(key=lambda r: -r["modified"])
        return out[:24]

    @app.post("/api/projects")
    async def upload(request: Request, name: str = "photo.png"):
        body = await request.body()
        if not body:
            raise HTTPException(400, "empty upload")
        if len(body) > MAX_UPLOAD:
            raise HTTPException(413, "The photo is larger than 80 MB.")
        S.eng()
        ext = Path(name).suffix.lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".gif"):
            raise HTTPException(415, "Open a JPEG, PNG, HEIC, WebP, TIFF or BMP photo.")
        up = S.projects / ".uploads"
        up.mkdir(parents=True, exist_ok=True)
        tdir = up / uuid.uuid4().hex[:8]
        tdir.mkdir()
        tmp = tdir / (re.sub(r"[^A-Za-z0-9_. -]", "_", Path(name).name)[-80:] or "photo.png")
        tmp.write_bytes(body)
        try:
            try:
                o = S.eng().open(tmp, embed=False)  # file work only, so a photo can be opened while another renders
            except PhotoError as e:
                raise HTTPException(415, str(e))
        finally:
            shutil.rmtree(tdir, ignore_errors=True)
        S.save_ui(o["project_id"], {"objects": [], "instruction": "", "seed": 1234})
        S.prefetch(o["project_id"])
        return S.project_view(o["project_id"])

    @app.delete("/api/projects/{pid}")
    def delete_project(pid: str):
        """Delete a project folder: the copy of the photo, its masks, versions and exports. The
        photo the user opened is never touched."""
        root = S.projects / _safe(pid)
        if not (root / "project.json").exists():
            raise HTTPException(404, "no such project")
        if any(j.project_id == pid and not j.done for j in S.jobs.values()):
            raise HTTPException(409, "An edit is running on this photo.")
        S.engine.projects.pop(pid, None) if S.engine else None
        shutil.rmtree(root)
        return {"ok": True}

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        S.prefetch(pid)
        return S.project_view(pid)

    @app.put("/api/projects/{pid}/prefs")
    async def prefs(pid: str, request: Request):
        body = await request.json()
        s = S.ui_state(pid)
        for k in ("instruction", "seed", "sel_for"):
            if k in body:
                s[k] = body[k]
        S.save_ui(pid, s)
        return {"ok": True}

    # ---- files ------------------------------------------------------------------------

    @app.get("/api/projects/{pid}/image/{rev}.png")
    def image(pid: str, rev: str):
        p = S.proj(pid)
        path = p.base_path(None if rev == "source" else _safe(rev))
        if not path.exists():
            raise HTTPException(404, "no such image")
        return FileResponse(path, media_type="image/png", headers=IMMUTABLE)

    @app.get("/api/projects/{pid}/view/{rev}.jpg")
    def view(pid: str, rev: str):
        p = S.proj(pid)
        src = p.base_path(None if rev == "source" else _safe(rev))
        if not src.exists():
            raise HTTPException(404, "no such image")
        return FileResponse(cached_jpeg(src, p.root / "view" / f"{rev}-fit.jpg", VIEW_FIT, 88), media_type="image/jpeg", headers=IMMUTABLE)

    @app.get("/api/projects/{pid}/full/{rev}.jpg")
    def view_full(pid: str, rev: str):
        """The sharp copy, asked for only when the page is zoomed in past the fit copy."""
        p = S.proj(pid)
        src = p.base_path(None if rev == "source" else _safe(rev))
        if not src.exists():
            raise HTTPException(404, "no such image")
        return FileResponse(cached_jpeg(src, p.root / "view" / f"{rev}.jpg", VIEW_MAX, 95), media_type="image/jpeg", headers=IMMUTABLE)

    @app.get("/api/projects/{pid}/thumb/{rev}.jpg")
    def thumb(pid: str, rev: str):
        p = S.proj(pid)
        src = p.base_path(None if rev == "source" else _safe(rev))
        if not src.exists():
            raise HTTPException(404, "no such image")
        return FileResponse(cached_jpeg(src, p.root / "thumbs" / f"{rev}.jpg", 360, 85), media_type="image/jpeg", headers=IMMUTABLE)

    @app.get("/api/projects/{pid}/mask/{folder}/{mid}.png")
    def mask_file(pid: str, folder: str, mid: str):
        if folder not in ("masks", "candidates"):
            raise HTTPException(404, "no such folder")
        p = S.proj(pid)
        f = p.root / folder / f"{_safe(mid)}.png"
        if not f.exists():
            raise HTTPException(404, "no such mask")
        return FileResponse(f, media_type="image/png", headers={"Cache-Control": "no-cache"})

    @app.get("/api/projects/{pid}/changemap/{rev}.jpg")
    def change_map(pid: str, rev: str):
        p = S.proj(pid)
        f = p.root / "runs" / _safe(rev) / "change_map.png"
        if not f.exists():
            raise HTTPException(404, "no change map for this version")
        return FileResponse(cached_jpeg(f, p.root / "view" / f"map-{rev}.jpg", VIEW_FIT, 88), media_type="image/jpeg", headers=IMMUTABLE)

    # ---- selection ---------------------------------------------------------------------

    def cand_view(pid: str, c: dict) -> dict:
        return {"candidate_id": c["candidate_id"], "score": c.get("score"), "label": c.get("label") or "", "box": c.get("box"),
                "confident": c.get("confident", True), "url": f"/api/projects/{pid}/mask/candidates/{c['candidate_id']}.png",
                "share": c.get("share")}

    @app.post("/api/projects/{pid}/select")
    async def select(pid: str, request: Request):
        b = await request.json()
        S.proj(pid)
        pts = [(int(x), int(y), int(l)) for x, y, l in b.get("points", [])]
        box = tuple(int(v) for v in b["box"]) if b.get("box") else None
        if not pts and not box:
            return {"candidates": []}
        r = S.locked(S.eng().select, pid, _safe(b["object_id"]), pts, box)
        return {"candidates": [cand_view(pid, c) for c in r["candidates"]]}

    @app.post("/api/projects/{pid}/find")
    async def find(pid: str, request: Request):
        b = await request.json()
        S.proj(pid)
        try:
            r = S.locked(S.eng().ground, pid, _safe(b["object_id"]), b.get("text", ""))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"phrase": r.get("phrase"), "candidates": [cand_view(pid, c) for c in r["candidates"]]}

    @app.post("/api/projects/{pid}/propose")
    async def propose(pid: str, request: Request):
        b = await request.json()
        S.proj(pid)
        instruction = (b.get("instruction") or "").strip()
        if not instruction:
            raise HTTPException(400, "Describe the edit first.")
        try:
            r = S.locked(S.eng().propose, pid, instruction, b.get("target_text"), b.get("protect_text"))
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        s = S.ui_state(pid)
        s["instruction"] = instruction
        S.save_ui(pid, s)
        try:  # the worker reads the sentence now, while the user looks at the proposal
            S.locked(S.eng().prepare_text, parse_instruction(instruction).edit)
        except HTTPException:
            pass
        t = r["target"]
        target = None
        if t.get("candidate_id"):
            target = {**cand_view(pid, t), "name": "background" if t["background"] else "whole photo" if t.get("whole") else " and ".join(t["phrases"]),
                      "weak": t["weak"], "background": t["background"], "whole": bool(t.get("whole"))}
        protect = []
        for e in r["protect"]:
            protect.append({"phrase": e["phrase"], "found": e["found"], "used": e["used"], "weak": e["weak"],
                            **(cand_view(pid, e) if e.get("candidate_id") else {"candidate_id": None})})
        return {"target": target, "target_phrases": t["phrases"], "background": t["background"], "protect": protect, "notes": r["notes"],
                "whole": bool(t.get("whole")), "edit": r["edit"],
                "fields": {"target": "background" if t["background"] else "whole photo" if t.get("whole") else ", ".join(t["phrases"]),
                                              "protect": ", ".join(e["phrase"] for e in r["protect"])}}

    @app.post("/api/projects/{pid}/objects")
    async def add_objects(pid: str, request: Request):
        """Keep candidates as objects. Body: {items: [{object_id, candidate_id|null, kind, name, invert, brush_png}], replace: bool}."""
        S.idle(pid)  # the running edit uses the selection as it was; changing it now would mislead
        b = await request.json()
        p = S.proj(pid)
        s = S.ui_state(pid)
        if b.get("replace"):
            s["objects"] = []
        elif b.get("replace_ids") is not None:  # replace only these; selections made since then stay
            drop = set(b["replace_ids"])
            s["objects"] = [o for o in s["objects"] if o["mask_id"] not in drop]
        eng = S.eng()
        for it in b.get("items", []):
            kind = it.get("kind", "target")
            if kind not in ("target", "protected"):
                raise HTTPException(400, "kind must be target or protected")
            brush = None
            if it.get("brush_png"):
                m = decode_png_mask(it["brush_png"], p.shape)
                brush = p.root / "masks" / f"brush-{uuid.uuid4().hex[:8]}.png"
                save_mask_png(m, brush)
            name = (it.get("name") or "").strip() or ("edit area" if kind == "target" else "kept area")
            r = S.locked(eng.commit_mask, pid, _safe(it.get("object_id") or f"obj-{uuid.uuid4().hex[:8]}"), it.get("candidate_id"), kind, name,
                         brush_add_png=str(brush) if brush else None, invert=bool(it.get("invert")))
            s["objects"].append({"mask_id": r["mask_id"]})
        S.save_ui(pid, s)
        return S.project_view(pid)

    @app.post("/api/projects/{pid}/objects/{mid}/brush")
    async def brush(pid: str, mid: str, request: Request):
        S.idle(pid)  # the running edit uses the selection as it was; changing it now would mislead
        b = await request.json()
        p = S.proj(pid)
        m = decode_png_mask(b["png"], p.shape)
        f = p.root / "masks" / f"brush-{uuid.uuid4().hex[:8]}.png"
        save_mask_png(m, f)
        op = b.get("op", "add")
        S.locked(S.eng().update_mask, pid, _safe(mid), brush_add_png=str(f) if op == "add" else None, brush_subtract_png=str(f) if op != "add" else None)
        return S.project_view(pid)

    @app.post("/api/projects/{pid}/objects/{mid}/grow")
    async def grow(pid: str, mid: str, request: Request):
        """Grow (px > 0) or shrink (px < 0) an area by a distance in pixels."""
        S.idle(pid)  # the running edit uses the selection as it was; changing it now would mislead
        from scipy import ndimage

        b = await request.json()
        px = int(b.get("px", 0))
        p = S.proj(pid)
        m = p.mask(_safe(mid))
        if px == 0 or not m.any():
            return S.project_view(pid)
        if px > 0:
            new = ndimage.distance_transform_edt(~m) <= px
            delta, op = new & ~m, "add"
        else:
            new = ndimage.distance_transform_edt(m) > -px
            delta, op = m & ~new, "subtract"
        f = p.root / "masks" / f"grow-{uuid.uuid4().hex[:8]}.png"
        save_mask_png(delta, f)
        S.locked(S.eng().update_mask, pid, mid, brush_add_png=str(f) if op == "add" else None, brush_subtract_png=str(f) if op != "add" else None)
        return S.project_view(pid)

    @app.patch("/api/projects/{pid}/objects/{mid}")
    async def edit_object(pid: str, mid: str, request: Request):
        """Rename, or switch between edit and keep. Mask records are immutable in kind, so a switch
        stores the same pixels as a new mask of the other kind and swaps it into the list."""
        S.idle(pid)  # the running edit uses the selection as it was; changing it now would mislead
        b = await request.json()
        p = S.proj(pid)
        s = S.ui_state(pid)
        rec = p.mask_record(_safe(mid))
        new_kind = b.get("kind", rec.kind)
        new_name = (b.get("name") or rec.name).strip() or rec.name
        if new_kind != rec.kind:
            nr = p.add_mask(p.mask(mid), new_kind, new_name, {**(rec.prompts or {}), "switched_from": mid})
            s["objects"] = [{"mask_id": nr.mask_id} if o["mask_id"] == mid else o for o in s["objects"]]
        elif new_name != rec.name:
            rec.name = new_name
            (p.root / "masks" / f"{mid}.json").write_text(json.dumps(rec.__dict__, indent=2, default=str))
            p.save()
        S.save_ui(pid, s)
        return S.project_view(pid)

    @app.delete("/api/projects/{pid}/objects/{mid}")
    def remove_object(pid: str, mid: str):
        S.idle(pid)  # the running edit uses the selection as it was; changing it now would mislead
        s = S.ui_state(pid)
        s["objects"] = [o for o in s["objects"] if o["mask_id"] != mid]
        S.save_ui(pid, s)
        return S.project_view(pid)

    @app.delete("/api/projects/{pid}/objects")
    def clear_objects(pid: str):
        S.idle(pid)  # the running edit uses the selection as it was; changing it now would mislead
        s = S.ui_state(pid)
        s["objects"] = []
        S.save_ui(pid, s)
        return S.project_view(pid)

    # ---- render ------------------------------------------------------------------------

    @app.post("/api/projects/{pid}/render")
    async def render(pid: str, request: Request):
        """Body: {instruction, seed, steps, band, strategy, strength, blend, retry_of, grow_from, seeds}.
        retry_of: render again from that revision's parent (another seed). grow_from: add that
        revision's expansion hint to its target first. seeds: render one version per seed (at
        most 4), all from the same parent, so the user can pick; the first good one is shown."""
        b = await request.json()
        p = S.proj(pid)
        eng = S.eng()
        if any(j.project_id == pid and not j.done for j in S.jobs.values()) or eng._render_lock.locked():
            raise HTTPException(409, "An edit is already running.")
        move = b.get("move")  # {"dx", "dy", "name"}: move the edit area's object by this many pixels
        move_sent = None
        if move is None and b.get("retry_of"):  # another seed or more versions of a move: the same offset
            prev = next((r for r in p.record.revisions if r["revision_id"] == _safe(b["retry_of"])), None)
            if prev and (prev.get("operation") or {}).get("kind") == "move":
                off = prev["operation"].get("offset") or [0, 0]
                move = {"dx": off[0], "dy": off[1], "shadow": ((prev["operation"].get("shadow") or {}).get("asked", True))}
                move_sent = prev["operation"].get("instruction")  # the same words for the model
        if move is not None:
            try:
                move = {"dx": int(move["dx"]), "dy": int(move["dy"]), "shadow": bool(move.get("shadow", True))}  # the name is read from the body below
            except (KeyError, TypeError, ValueError):
                raise HTTPException(400, "A move needs whole-pixel dx and dy.")
            if move["dx"] == 0 and move["dy"] == 0:
                raise HTTPException(400, "Drag the object to where it should go first.")
        instruction = (b.get("instruction") or "").strip()
        if not instruction and move is None:
            raise HTTPException(400, "Describe the edit first.")
        s = S.ui_state(pid)
        view = S.project_view(pid)

        def prepare():
            if b.get("grow_from"):
                rid = _safe(b["grow_from"])
                meta = json.loads((p.root / "runs" / rid / "meta.json").read_text())
                hint = p.root / "runs" / rid / "expansion_hint.png"
                hints = list((p.root / "runs" / rid).glob("*hint*.png"))
                if not hint.exists() and hints:
                    hint = hints[0]
                if hint.exists():
                    tid = meta["operation"]["target_mask_id"]
                    parts = (p.mask_record(tid).prompts or {}).get("union_of") or []
                    # several edit areas are rebuilt into a union at each render, so grow one of the areas, not the old union
                    live = [m for m in parts if any(o["mask_id"] == m for o in S.ui_state(pid)["objects"])]
                    eng.update_mask(pid, live[0] if live else tid, brush_add_png=str(hint))
                eng.set_head(pid, meta["parent"])
            elif b.get("retry_of"):
                rid = _safe(b["retry_of"])
                rev = next((r for r in p.record.revisions if r["revision_id"] == rid), None)
                if rev is None:
                    raise HTTPException(404, "no such version")
                eng.set_head(pid, rev["parent"])

        shown = p.record.head  # what is on screen now; restored if no new version comes out
        S.locked(prepare)
        view = S.project_view(pid)
        targets = [o for o in view["objects"] if o["kind"] == "target"]
        protected = [o for o in view["objects"] if o["kind"] == "protected"]
        if not targets:
            raise HTTPException(400, "Nothing is selected to edit. Press Find, click the object, or paint it.")
        if len(targets) == 1:
            target_id = targets[0]["mask_id"]
        else:  # several edit areas are edited together
            union = p.union([o["mask_id"] for o in targets])
            target_id = p.add_mask(union, "target", " and ".join(o["name"] for o in targets), {"union_of": [o["mask_id"] for o in targets]}).mask_id
        if move is not None:  # the model only fills the old place; the object is copied, not redrawn
            what = re.sub(r"[^A-Za-z0-9 ,'-]", "", str(b.get("move", {}).get("name") or "")).strip()[:60] if isinstance(b.get("move"), dict) else ""
            what = what or " and ".join(o["name"] for o in targets if o["name"] not in ("edit area", "kept area")) or "object"
            instruction = instruction or f"move the {what}"
            sent = move_sent or f"remove the {what} and fill in what was behind it"
        else:
            sent = parse_instruction(instruction).edit  # keep clauses are enforced by masks, not sent to the model
        plan = {"schema_version": 2, "project_id": pid, "parent_revision": p.record.head,
                "operation": {"kind": "move" if move is not None else "native_instruction_edit", "target_mask_id": target_id, "instruction": sent,
                              "band_px": int(b.get("band", 8)), "strategy": b.get("strategy", "auto"), "blend": b.get("blend", "auto"),
                              **({"offset": [move["dx"], move["dy"]], "shadow": move["shadow"]} if move is not None else {})},
                "protected_mask_ids": [o["mask_id"] for o in protected], "seed": int(b.get("seed", 1234)), "steps": int(b.get("steps", 4)),
                "strength": float(b.get("strength", 0.8)), "backend_profile_id": eng.cfg.profile_id, "network_policy": "offline"}
        s["instruction"] = instruction
        s["seed"] = int(b.get("seed", 1234))
        S.save_ui(pid, s)
        seeds = [int(x) for x in (b.get("seeds") or [])][:4] or [plan["seed"]]
        job = Job(pid)
        job.many = len(seeds) > 1
        S.jobs[job.id] = job

        def run():
            if not S.lock.acquire(timeout=30):
                job.result = {"status": "failed", "message": "The editor was busy. Try again."}
                job.done = True
                return
            try:
                oks, last = [], None
                for k, sd in enumerate(seeds):
                    if job.cancel.is_set():
                        break
                    if k:
                        eng.set_head(pid, plan["parent_revision"])  # every version starts from the same image
                    for msg in eng.render({**plan, "seed": sd}, job.cancel):
                        if msg["type"] == "progress":
                            job.events.append({"step": msg.get("step"), "of": msg.get("of"), "note": msg.get("note"), "attempt": msg.get("attempt"),
                                               "variant": k + 1, "variants": len(seeds), "t": round(time.time() - job.started, 1)})
                        else:
                            last = msg
                    if last and last.get("status") == "ok":
                        oks.append(last)
                    elif last and last.get("status") == "cancelled":
                        break
                if oks:
                    job.variants = [o["revision_id"] for o in oks]
                    eng.set_head(pid, oks[0]["revision_id"])  # show the first; the others are one click away
                    job.result = oks[0]
                else:
                    if p.record.head != shown:  # cancelled or failed: back to the image the user was looking at
                        eng.set_head(pid, shown)
                    job.result = last
            except EngineBusy as e:
                job.result = {"status": "failed", "message": str(e)}
            except Exception as e:
                job.result = {"status": "failed", "message": f"The edit failed: {e}"}
                print(f"render failed: {e!r}", file=sys.stderr)
            finally:
                try:
                    if (job.result or {}).get("status") != "ok" and p.record.head != shown:
                        eng.set_head(pid, shown)
                except Exception:  # noqa: BLE001
                    pass
                S.lock.release()
                job.done = True
                S.prefetch(pid)  # the next click on the new image is quick

        threading.Thread(target=run, daemon=True).start()
        return {"job_id": job.id, "sent": sent}

    # ---- new pictures from words ---------------------------------------------------------

    GEN_KEEP = 12  # generation folders kept; older ones (pictures nobody opened) are removed

    @app.post("/api/generate")
    async def generate(request: Request):
        """Body: {prompt, size: square|landscape|portrait|wide|tall, count 1 to 4, seed}. Makes
        `count` pictures, one seed each, shown as they finish; one is opened as a new project."""
        b = await request.json()
        prompt = re.sub(r"\s+", " ", (b.get("prompt") or "")).strip()
        if not prompt:
            raise HTTPException(400, "Describe the picture first.")
        if len(prompt) > 1500:
            raise HTTPException(400, "Use a shorter description (at most 1500 characters).")
        size = b.get("size") or "square"
        refs: list[Path] = []
        if b.get("ref_project"):  # the image on screen in that project
            rp = S.proj(_safe(b["ref_project"]))
            refs.append(rp.base_path(rp.record.head))
        for rid in (b.get("ref_uploads") or [])[:2]:
            f = S.projects / ".refs" / f"{_safe(rid)}.png"
            if not f.exists():
                raise HTTPException(404, "That reference photo is no longer here. Add it again.")
            refs.append(f)
        if size not in Engine.GEN_SIZES and not (size == "photo" and refs):
            raise HTTPException(400, "Choose square, landscape, portrait, wide or tall.")
        count = max(1, min(4, int(b.get("count") or 4)))
        seed = int(b["seed"]) if b.get("seed") not in (None, "") else int.from_bytes(os.urandom(3), "big")
        eng = S.eng()
        if any(not j.done for j in S.jobs.values()) or eng._render_lock.locked():
            raise HTTPException(409, "Busy: an edit or a picture is being made. Try again when it finishes.")
        gen_root = S.projects / ".generated"
        gen_root.mkdir(parents=True, exist_ok=True)
        old = sorted((d for d in gen_root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)
        for d in old[: max(0, len(old) - GEN_KEEP + 1)]:
            shutil.rmtree(d, ignore_errors=True)
        job = Job("")
        job.kind, job.prompt, job.many = "generate", prompt, count > 1
        gdir = gen_root / job.id
        gdir.mkdir()
        (gdir / "prompt.json").write_text(json.dumps({"prompt": prompt, "size": size, "seed": seed, "count": count, "refs": len(refs)}))
        S.jobs[job.id] = job

        _run_pictures(job, count, lambda k: eng.generate(prompt, size, seed + k, gdir / f"{k}.png", job.cancel, refs=refs), seed)
        return {"job_id": job.id, "seed": seed, "count": count}

    @app.post("/api/refs")
    async def add_reference(request: Request, name: str = "ref.png"):
        """A second photo to draw from (someone or something to put into the open photo). Kept
        under projects/.refs as a lossless copy; never sent anywhere."""
        body = await request.body()
        if not body:
            raise HTTPException(400, "empty upload")
        if len(body) > MAX_UPLOAD:
            raise HTTPException(413, "The photo is larger than 80 MB.")
        d = S.projects / ".refs"
        d.mkdir(parents=True, exist_ok=True)
        rid = uuid.uuid4().hex[:12]
        tmp = d / f"{rid}{Path(name).suffix.lower() or '.png'}"
        tmp.write_bytes(body)
        try:
            from lis.project import decode_photo, save_rgb_png

            arr, _notes = decode_photo(tmp)
        except PhotoError as e:
            raise HTTPException(415, str(e))
        finally:
            tmp.unlink(missing_ok=True)
        save_rgb_png(arr, d / f"{rid}.png")
        return {"ref_id": rid, "width": int(arr.shape[1]), "height": int(arr.shape[0])}

    EXTEND_SHAPES = ("wider", "taller", "all", "square", "16:9", "9:16", "4:3", "3:4")

    @app.post("/api/projects/{pid}/extend")
    async def extend(pid: str, request: Request):
        """Body: {shape, hint, count 1 to 4, seed}. Adds new picture past the edges of the image on
        screen; each version opens as a new project (the photo's size changes, so the masks and
        history of this one cannot follow)."""
        b = await request.json()
        p = S.proj(pid)
        shape = b.get("shape") or "wider"
        if shape not in EXTEND_SHAPES:
            raise HTTPException(400, "Choose wider, taller, all sides, square, 16:9, 9:16, 4:3 or 3:4.")
        hint = re.sub(r"\s+", " ", (b.get("hint") or "")).strip()[:400]
        count = max(1, min(4, int(b.get("count") or 2)))
        seed = int(b["seed"]) if b.get("seed") not in (None, "") else int.from_bytes(os.urandom(3), "big")
        eng = S.eng()
        if any(not j.done for j in S.jobs.values()) or eng._render_lock.locked():
            raise HTTPException(409, "Busy: an edit or a picture is being made. Try again when it finishes.")
        L, R, T, B = Engine.extend_sides(p.record.height, p.record.width, shape)
        if L + R + T + B == 0:
            raise HTTPException(400, "The photo already has that shape.")
        gen_root = S.projects / ".generated"
        gen_root.mkdir(parents=True, exist_ok=True)
        job = Job(pid)
        job.kind, job.prompt, job.many = "generate", hint, count > 1
        gdir = gen_root / job.id
        gdir.mkdir()
        (gdir / "prompt.json").write_text(json.dumps({"kind": "extend", "from": p.record.original_name, "from_project": pid, "revision": p.record.head,
                                                       "shape": shape, "hint": hint, "seed": seed, "count": count, "sides": [L, R, T, B]}))
        S.jobs[job.id] = job
        kept = [o["mask_id"] for o in S.project_view(pid)["objects"] if o["kind"] == "protected"]
        _run_pictures(job, count, lambda k: eng.extend(pid, shape, hint, seed + k, gdir / f"{k}.png", job.cancel, protected_ids=kept), seed)
        return {"job_id": job.id, "seed": seed, "count": count, "size": [p.record.width + L + R, p.record.height + T + B]}

    def _run_pictures(job: Job, count: int, make, seed: int) -> None:
        def run():
            if not S.lock.acquire(timeout=30):
                job.result = {"status": "failed", "message": "The editor was busy. Try again."}
                job.done = True
                return
            last = None
            try:
                for k in range(count):
                    if job.cancel.is_set():
                        break
                    for msg in make(k):
                        if msg["type"] == "progress":
                            job.events.append({"step": msg["step"], "of": msg["of"], "variant": k + 1, "variants": count,
                                               "t": round(time.time() - job.started, 1)})
                        else:
                            last = msg
                    if last and last["status"] == "ok":
                        job.images.append({"i": k, "seed": seed + k, "width": last["width"], "height": last["height"],
                                           "url": f"/api/generated/{job.id}/{k}.jpg", "timing_ms": last["timing_ms"]})
                    else:
                        break
                if job.images:
                    job.result = {"status": "ok", "message": "" if len(job.images) == count else (last or {}).get("message", "")}
                else:
                    job.result = last or {"status": "cancelled", "message": ""}
            except EngineBusy as e:
                job.result = {"status": "failed", "message": str(e)}
            except Exception as e:  # noqa: BLE001
                job.result = {"status": "failed", "message": f"Making the picture failed: {e}"}
                print(f"picture job failed: {e!r}", file=sys.stderr)
            finally:
                S.lock.release()
                job.done = True

        threading.Thread(target=run, daemon=True).start()

    def _generated(jid: str, i: int) -> Path:
        f = S.projects / ".generated" / _safe(jid) / f"{int(i)}.png"
        if not f.exists():
            raise HTTPException(404, "That picture is no longer kept. Make it again.")
        return f

    @app.get("/api/generated/{jid}/{i}.jpg")
    def generated_view(jid: str, i: int):
        f = _generated(jid, i)
        return FileResponse(cached_jpeg(f, f.parent / f"{int(i)}.jpg", None, 92), media_type="image/jpeg", headers=IMMUTABLE)

    @app.post("/api/generated/{jid}/{i}/open")
    def generated_open(jid: str, i: int):
        """Open one made picture as a new project (a lossless copy), with how it was made noted."""
        f = _generated(jid, i)
        meta = json.loads((f.parent / "prompt.json").read_text())
        if meta.get("kind") == "extend":
            named = f.parent / f"{Path(meta['from']).stem}-extended-{int(i) + 1}.png"
        else:
            words = re.sub(r"[^A-Za-z0-9 ]", "", meta["prompt"]).strip().split()[:6] or ["picture"]
            named = f.parent / ("-".join(w.lower() for w in words) + f"-{int(i) + 1}.png")
        if not named.exists():
            shutil.copy2(f, named)
        o = S.eng().open(named, embed=False)
        pj = S.proj(o["project_id"])
        if meta.get("kind") == "extend":
            L, R, T, B = meta["sides"]
            added = ", ".join(f"{v} px {n}" for n, v in (("left", L), ("right", R), ("top", T), ("bottom", B)) if v)
            pj.record.import_notes = [f"extended from {meta['from']} ({added}) by FLUX.2 Klein 4B on this computer, seed {meta['seed'] + int(i)}"
                                      + (f', asked for "{meta["hint"]}"' if meta.get("hint") else "")
                                      + "; the original's pixels are exact except a narrow cross-fade inside each new edge"]
        else:
            drawn = {0: "", 1: " from the photo it was made from", 2: " from two reference photos"}.get(int(meta.get("refs") or 0), " from reference photos")
            pj.record.import_notes = [f'made from the words "{meta["prompt"]}"{drawn} (seed {meta["seed"] + int(i)}, {meta["size"]}) by FLUX.2 Klein 4B on this computer']
        pj.save()
        S.save_ui(o["project_id"], {"objects": [], "instruction": "", "seed": 1234})
        S.prefetch(o["project_id"])
        return S.project_view(o["project_id"])

    @app.get("/api/jobs/{jid}")
    def job_status(jid: str, after: int = 0):
        job = S.jobs.get(jid)
        if job is None:
            raise HTTPException(404, "no such job")
        out = {"job_id": jid, "done": job.done, "events": job.events[after:], "n": len(job.events), "elapsed": round(time.time() - job.started, 1)}
        if job.kind == "generate":
            out["images"] = list(job.images)
            if job.done:
                r = job.result or {}
                out["status"], out["message"] = r.get("status", "failed"), r.get("message", "")
            return out
        if job.done:
            r = job.result or {}
            out["status"] = r.get("status", "failed")
            out["message"] = r.get("message", "")
            if r.get("status") == "ok":
                p = S.proj(job.project_id)
                rev = next(x for x in p.record.revisions if x["revision_id"] == r["revision_id"])
                out["revision"] = revision_view(job.project_id, rev)
                out["expansion_hint"] = bool(r.get("expansion_hint_mask_path"))
                if job.many and job.variants:  # even one finished version is offered next to the one shown before
                    by_id = {x["revision_id"]: x for x in p.record.revisions}
                    out["variants"] = [revision_view(job.project_id, by_id[v]) for v in job.variants if v in by_id]
            out["project"] = S.project_view(job.project_id)
        return out

    @app.post("/api/jobs/{jid}/cancel")
    def job_cancel(jid: str):
        job = S.jobs.get(jid)
        if job is None:
            raise HTTPException(404, "no such job")
        job.cancel.set()
        return {"ok": True}

    # ---- history and export ------------------------------------------------------------

    @app.post("/api/projects/{pid}/head")
    async def head(pid: str, request: Request):
        b = await request.json()
        rid = b.get("revision_id")
        S.locked(S.eng().set_head, pid, None if rid in (None, "source") else _safe(rid))
        S.prefetch(pid)
        return S.project_view(pid)

    @app.post("/api/projects/{pid}/undo")
    def undo(pid: str):
        p = S.proj(pid)
        if p.record.head is None:
            return S.project_view(pid)
        came_from = p.record.head
        parent = next(r["parent"] for r in p.record.revisions if r["revision_id"] == came_from)
        S.locked(S.eng().set_head, pid, parent)
        st = S.ui_state(pid)
        st["redo_to"] = came_from  # redo returns to this version, not to the newest sibling
        S.save_ui(pid, st)
        return S.project_view(pid)

    @app.post("/api/projects/{pid}/redo")
    def redo(pid: str):
        p = S.proj(pid)
        kids = [r for r in p.record.revisions if r["parent"] == p.record.head]
        if kids:
            want = S.ui_state(pid).get("redo_to")
            pick = next((r for r in kids if r["revision_id"] == want), kids[-1])
            S.locked(S.eng().set_head, pid, pick["revision_id"])
        return S.project_view(pid)

    @app.get("/api/projects/{pid}/export")
    def export(pid: str, fmt: str = "png", rev: str | None = None):
        p = S.proj(pid)
        rid = p.record.head if rev is None else (None if rev == "source" else _safe(rev))
        out_dir = p.root / "exports"
        out_dir.mkdir(exist_ok=True)
        stem = Path(p.record.original_name).stem or "photo"
        ext = "jpg" if fmt == "jpeg" else "png"
        path = out_dir / f"{stem}-{rid or 'original'}.{ext}"
        r = S.locked(S.eng().export, pid, rid, path, "jpeg" if fmt == "jpeg" else "png")
        if not r["ok"]:
            path.unlink(missing_ok=True)
            return JSONResponse({"detail": "The check on the saved file failed, so this version is not exported."}, status_code=409)
        return FileResponse(path, media_type="image/jpeg" if fmt == "jpeg" else "image/png", filename=path.name,
                            headers={"X-Check": r["protected_check"]["status"], "Cache-Control": "no-store"})

    # ---- adjust and geometry: no model (lis/adjust.py) ------------------------------------

    @app.post("/api/projects/{pid}/adjust")
    async def adjust(pid: str, request: Request):
        """Body: {brightness, contrast, saturation, warmth (whole numbers, -100..100), parent}.
        Applies to the edit areas in use, or the whole photo when there are none; kept areas are
        copied exactly. `parent` (a revision id, or "source") is the image the sliders were moved
        on; a different head is refused. Makes a revision with the same file checks as a render."""
        S.idle(pid)
        b = await request.json()
        p = S.proj(pid)
        eng = S.eng()
        if eng._render_lock.locked():
            raise HTTPException(409, "An edit is running. Adjust when it finishes.")
        view = S.project_view(pid)
        targets = [o["mask_id"] for o in view["objects"] if o["kind"] == "target"]
        protected = [o["mask_id"] for o in view["objects"] if o["kind"] == "protected"]
        parent = b.get("parent", "head")
        parent = None if parent == "source" else parent
        try:
            r = S.locked(adjust_ops.adjust, eng, pid, b, targets, protected, parent)
        except adjust_ops.AdjustError as e:
            raise HTTPException(400, str(e))
        if r["status"] != "ok":
            return JSONResponse({"detail": r["message"], "status": r["status"]}, status_code=500)
        rev = next(x for x in p.record.revisions if x["revision_id"] == r["revision_id"])
        return {"status": "ok", "revision": revision_view(pid, rev), "project": S.project_view(pid)}

    @app.post("/api/projects/{pid}/transform")
    async def transform(pid: str, request: Request):
        """Body: {op: rotate_left | rotate_right | flip_horizontal | crop, box: [x, y, w, h] for crop}.
        Makes a NEW project from the image on screen (lossless PNG) with an import note; the
        masks and history of this project cannot follow a change of geometry, so it is left as is."""
        S.idle(pid)
        b = await request.json()
        p = S.proj(pid)
        eng = S.eng()
        try:
            o = adjust_ops.transform(eng, pid, b.get("op", ""), b.get("box"))
        except adjust_ops.AdjustError as e:
            raise HTTPException(400, str(e))
        old = S.ui_state(pid)
        S.save_ui(o["project_id"], {"objects": [], "instruction": old.get("instruction", ""), "seed": old.get("seed", 1234)})
        return {**S.project_view(o["project_id"]), "note": o["note"], "from_project": p.record.project_id}

    return app


# ---- main -------------------------------------------------------------------------------


def main(argv=None) -> None:
    from lis.app import make_engine

    ap = argparse.ArgumentParser(description="OpenImage Studio editor")
    ap.add_argument("--fake", action="store_true", help="no models: fake selection and a fake editor, for trying the interface")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--profile", default="auto", help="memory profile: auto (default: resident, fast or group, by free video memory) | group | group_fp8 | fast | resident | offload | offload_fp8")
    ap.add_argument("--segmenter", default="sam2", choices=["sam2", "sam3_tracker"])
    ap.add_argument("--grounder", default="auto", choices=["auto", "dino", "sam3", "none"])
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 keeps it on this machine; 0.0.0.0 lets other devices on your network open it")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--projects", default=str(ROOT / "projects"))
    ap.add_argument("--open", action="store_true", help="open the browser")
    ap.add_argument("--key", default=None, help="access key other devices need (set by --lan in scripts/start.sh)")
    ap.add_argument("--small-gpu", default="auto", choices=["auto", "on", "off"],
                    help="move the selection models off the GPU during a render; auto turns it on below 12 GB of VRAM")
    a = ap.parse_args(argv)
    projects = Path(a.projects)
    projects.mkdir(parents=True, exist_ok=True)
    studio = Studio(lambda: make_engine(a.fake, a.offload, a.segmenter, projects, a.profile, a.grounder, a.small_gpu), projects)
    studio.start()
    key = a.key
    if a.host not in ("127.0.0.1", "localhost", "::1") and not key:
        key = secrets.token_urlsafe(9)
        print(f"Other devices need the key: open http://<this computer's address>:{a.port}/?key={key}", file=sys.stderr, flush=True)
    app = create_app(studio, key)
    import uvicorn

    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '127.0.0.1') else a.host}:{a.port}"
    print(f"OpenImage Studio: open {url}", file=sys.stderr, flush=True)
    if a.open:
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
