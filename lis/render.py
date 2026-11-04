"""Client side of the render worker.

Starts a worker module in a subprocess, sends one job at a time as JSON lines, streams
progress, and forwards cancel as a soft interrupt that keeps the model loaded. Hard kill
only on no answer within CANCEL_GRACE_S of a cancel, a job over JOB_TIMEOUT_S, or out of
memory. After MAX_DEATHS consecutive deaths the client stops
restarting and answers every job with worker_unavailable.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

CANCEL_GRACE_S = 30.0
JOB_TIMEOUT_S = 900.0  # a slow machine that reads the weights from disk can take minutes per edit
LOAD_TIMEOUT_S = 600.0
MAX_DEATHS = 3  # consecutive: a finished job resets the count, a cancel is not a death


class WorkerUnavailable(RuntimeError):
    pass


@dataclass
class WorkerConfig:
    module: str = "lis.worker_klein"  # or lis.worker_fake
    weights: str = ""
    offload: bool = False
    profile: str | None = None  # memory profile name; overrides offload
    python: str = sys.executable
    extra_args: list[str] = field(default_factory=list)


class RenderClient:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.proc: subprocess.Popen | None = None
        self.lines: "queue.Queue[dict | None]" = queue.Queue()
        self.ready: dict | None = None
        self.deaths = 0
        self.log_path: str | None = None
        self._stderr_tail: list[str] = []

    # ---- lifecycle -------------------------------------------------------------------

    @property
    def available(self) -> bool:
        return self.deaths < MAX_DEATHS

    def start(self) -> dict:
        """Start the worker and wait for `ready`. Raises WorkerUnavailable after too many deaths."""
        if not self.available:
            raise WorkerUnavailable(f"render worker died {self.deaths} times; restart the application (log: {self.log_path})")
        cmd = [self.config.python, "-m", self.config.module, "--weights", self.config.weights, *self.config.extra_args]
        if self.config.profile:
            cmd += ["--profile", self.config.profile]
        elif self.config.offload:
            cmd.append("--offload")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.lines = queue.Queue()
        # Each pump binds to its own process and queue, so a killed worker's EOF never lands
        # in the queue of its replacement.
        threading.Thread(target=self._pump_stdout, args=(self.proc, self.lines), daemon=True).start()
        threading.Thread(target=self._pump_stderr, args=(self.proc,), daemon=True).start()
        self._send({"type": "load"})
        msg = self._wait(LOAD_TIMEOUT_S)
        if msg is None or msg.get("type") != "ready":
            self._record_death()
            raise WorkerUnavailable(f"worker did not become ready: {msg}")
        self.ready = msg
        return msg

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self._send({"type": "quit"})
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        self.proc = None

    def _kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()

    def _record_death(self) -> None:
        self.deaths += 1
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None
        self.ready = None

    def _pump_stdout(self, proc: subprocess.Popen, q: "queue.Queue[dict | None]") -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                q.put(json.loads(line))
            except json.JSONDecodeError:
                self._stderr_tail.append(f"[stdout?] {line[:200]}")
        q.put(None)  # EOF: the process exited

    def _pump_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            self._stderr_tail.append(line.rstrip())
            del self._stderr_tail[:-200]

    def _send(self, msg: dict) -> None:
        if not self.proc or self.proc.stdin is None or self.proc.poll() is not None:
            raise WorkerUnavailable("worker process is not running")
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _wait(self, timeout: float) -> dict | None:
        try:
            return self.lines.get(timeout=timeout)
        except queue.Empty:
            return {"type": "timeout"}

    # ---- jobs ------------------------------------------------------------------------

    def prepare(self, text: str) -> None:
        """Ask the worker to encode a sentence while it is idle. No answer is sent; a job with
        the same sentence then finds it ready. Ignored when the worker is not running."""
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self._send({"type": "prepare", "instruction": text})
        except Exception:  # noqa: BLE001  (only a speed-up)
            pass

    def render(self, job: dict, cancel: threading.Event | None = None) -> Iterator[dict]:
        """Yield progress dicts, then exactly one terminal dict: done, failed.

        `cancel` is checked between messages; when set, a soft cancel is sent and the worker
        gets CANCEL_GRACE_S to answer before it is killed.
        """
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        self._send(job)
        deadline = time.time() + JOB_TIMEOUT_S
        cancel_sent_at: float | None = None
        while True:
            msg = self._wait(0.25)
            now = time.time()
            if msg is None:  # process exited
                self._record_death()
                yield {"type": "failed", "id": job["id"], "cause": "error", "detail": "worker process exited", "stderr": self._stderr_tail[-20:]}
                return
            if msg.get("type") == "timeout":
                if cancel is not None and cancel.is_set() and cancel_sent_at is None:
                    self._send({"type": "cancel", "id": job["id"]})
                    cancel_sent_at = now
                if cancel_sent_at is not None and now - cancel_sent_at > CANCEL_GRACE_S:
                    self._kill()  # the user asked for it; not counted as a death
                    yield {"type": "failed", "id": job["id"], "cause": "cancelled", "detail": "worker did not answer the cancel; killed and will restart"}
                    return
                if now > deadline:
                    self._record_death()
                    yield {"type": "failed", "id": job["id"], "cause": "error", "detail": f"job exceeded {JOB_TIMEOUT_S:.0f} s; worker killed"}
                    return
                continue
            if msg.get("id") not in (job["id"], None):
                continue  # stale message from an earlier job
            if msg["type"] == "progress":
                yield msg
                if cancel is not None and cancel.is_set() and cancel_sent_at is None:
                    self._send({"type": "cancel", "id": job["id"]})
                    cancel_sent_at = now
                continue
            if msg["type"] in ("done", "failed"):
                if msg["type"] == "done":
                    self.deaths = 0
                if msg["type"] == "failed" and msg.get("cause") == "out_of_memory":
                    pass  # the engine decides whether to retry at a smaller working region
                yield msg
                return
