"""Persistence --- factor ledger + reproducibility manifest.

Everything needed to explain and replay a run lands under
``artifacts/runs/<run_name>/``: a JSONL ledger (one factor decision per line)
and a manifest (config + data version + model/provider + seed).
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import tempfile
import threading
import warnings

from .providers.cli_base import redact_secrets


class RunLockedError(RuntimeError):
    """Raised when another process/thread owns the same ``run_name``."""


_PROCESS_LOCKS: dict[str, dict] = {}
_PROCESS_LOCKS_GUARD = threading.RLock()


def _redact_obj(value):
    """Recursively redact strings before any JSON artifact reaches disk."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {key: _redact_obj(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_obj(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_obj(item) for item in value)
    return value


def _safe_run_name(run_name: str) -> str:
    """Validate ``run_name`` as a single path segment (no traversal / absolutes).

    The run name becomes a directory under ``artifacts/runs/``; a value like
    ``../../etc`` or ``/tmp/x`` would let a run write outside the artifacts tree.
    """
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be a non-empty string")
    name = run_name.strip()
    # os.path.splitdrive catches Windows drive-relative names like "C:foo" or
    # "a:b" that os.path.isabs reports False for but os.path.join still resets to
    # the drive root (escaping artifacts/runs); reject any ':' for good measure.
    if name in (".", "..") or any(sep in name for sep in ("/", "\\", ":")) \
            or os.path.isabs(name) or (os.path.altsep and os.path.altsep in name) \
            or os.path.splitdrive(name)[0]:
        raise ValueError(
            f"run_name {run_name!r} must be a single path segment "
            "(no '/', '\\\\', ':', '..', drive letters, or absolute paths)"
        )
    return name


class Store:
    def __init__(self, root: str, run_name: str):
        self.root = root
        self.run_name = _safe_run_name(run_name)
        self.run_dir = os.path.join(root, "runs", self.run_name)
        self.ledger_path = os.path.join(self.run_dir, "factors.jsonl")
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.trace_path = os.path.join(self.run_dir, "llm_trace.jsonl")
        self.lock_path = os.path.join(self.run_dir, ".owner.lock")
        self._claim_depth = 0

    def ensure(self) -> None:
        os.makedirs(self.run_dir, exist_ok=True)

    def acquire(self) -> None:
        """Claim exclusive ownership of this run directory until ``release``.

        The OS lock is held for the complete run lifecycle, covering reset,
        manifest writes, ledger appends and trace appends as one transaction
        domain. It is re-entrant only on the same thread, which lets the TUI
        claim before opening ``run.log`` and then call the library runner.
        """
        self.ensure()
        key = os.path.normcase(os.path.abspath(self.lock_path))
        owner = threading.get_ident()
        with _PROCESS_LOCKS_GUARD:
            held = _PROCESS_LOCKS.get(key)
            if held is not None:
                if held["thread"] != owner:
                    raise RunLockedError(
                        f"run '{self.run_name}' is already active in this process"
                    )
                held["count"] += 1
                self._claim_depth += 1
                return

            fh = open(self.lock_path, "a+b")
            try:
                fh.seek(0, os.SEEK_END)
                if fh.tell() == 0:
                    fh.write(b"\0")
                    fh.flush()
                fh.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:  # pragma: no cover - exercised by Linux/macOS CI
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, BlockingIOError) as exc:
                fh.close()
                raise RunLockedError(
                    f"run '{self.run_name}' is already owned by another process; "
                    f"wait for it to finish or choose a different run_name"
                ) from exc
            _PROCESS_LOCKS[key] = {"fh": fh, "thread": owner, "count": 1}
            self._claim_depth = 1

    def release(self) -> None:
        """Release a prior :meth:`acquire`; safe to call when unclaimed."""
        if self._claim_depth <= 0:
            return
        key = os.path.normcase(os.path.abspath(self.lock_path))
        with _PROCESS_LOCKS_GUARD:
            held = _PROCESS_LOCKS.get(key)
            self._claim_depth -= 1
            if held is None:
                return
            held["count"] -= 1
            if held["count"] > 0:
                return
            fh = held["fh"]
            try:
                fh.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - exercised by Linux/macOS CI
                    import fcntl

                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()
                _PROCESS_LOCKS.pop(key, None)

    @contextmanager
    def ownership(self):
        """Context manager holding this run's cross-process ownership lock."""
        self.acquire()
        try:
            yield self
        finally:
            self.release()

    def _atomic_write(self, path: str, payload: bytes) -> None:
        """Replace ``path`` atomically with fully flushed bytes."""
        self.ensure()
        fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.",
                                   suffix=".tmp", dir=self.run_dir)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def write_manifest(self, manifest: dict) -> None:
        clean = _redact_obj(manifest)
        if clean.get("config", {}).get("math_contract") == "strict-math-v2":
            from .factor.contract import POLICY_HASH
            if clean.get("math_policy_hash", POLICY_HASH) != POLICY_HASH:
                raise ValueError("MATH_POLICY_MISMATCH: cannot relabel an existing manifest")
            clean["math_policy_hash"] = POLICY_HASH
        payload = json.dumps(clean, indent=2, ensure_ascii=False).encode("utf-8")
        self._atomic_write(self.manifest_path, payload)

    def read_manifest(self) -> dict:
        if not os.path.exists(self.manifest_path):
            return {}
        with open(self.manifest_path, encoding="utf-8") as fh:
            return json.load(fh)

    def _append_line(self, path: str, obj: dict) -> None:
        """Append one JSON line, healing a missing trailing newline first.

        If a previous append was cut off by a hard kill --- bytes on disk with no
        final newline --- a naive append would glue the new record onto that
        partial line, and the reader (which drops the one resulting bad line)
        would lose BOTH, silently swallowing the new record. So if the file does
        not already end in a newline, emit one first to isolate the fragment on
        its own (skippable) line.
        """
        self.ensure()
        needs_nl = False
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "rb") as fh:
                fh.seek(-1, os.SEEK_END)
                needs_nl = fh.read(1) != b"\n"
        clean = _redact_obj(obj)
        with open(path, "a", encoding="utf-8") as fh:
            if needs_nl:
                fh.write("\n")
            fh.write(json.dumps(clean, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _append_line_atomic(self, path: str, obj: dict) -> None:
        """Atomically append one JSON line by replacing the complete file."""
        self.ensure()
        old = b""
        if os.path.exists(path):
            with open(path, "rb") as fh:
                old = fh.read()
        if old and not old.endswith(b"\n"):
            old += b"\n"
        clean = _redact_obj(obj)
        line = (json.dumps(clean, ensure_ascii=False) + "\n").encode("utf-8")
        self._atomic_write(path, old + line)

    def append_factor(self, record: dict) -> None:
        self._append_line(self.ledger_path, record)

    def read_factors(self) -> list[dict]:
        if not os.path.exists(self.ledger_path):
            return []
        out = []
        with open(self.ledger_path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    # one corrupt/truncated line (e.g. an interrupted append)
                    # must not sink the whole ledger --- skip it and warn.
                    warnings.warn(
                        f"skipping unreadable ledger line {lineno} in "
                        f"{self.ledger_path}: {exc}",
                        RuntimeWarning, stacklevel=2,
                    )
        return out

    def append_trace(self, entry: dict) -> None:
        """Append one LLM-completion trace entry (for replay); written live."""
        self._append_line_atomic(self.trace_path, entry)

    def read_trace(self) -> list[dict]:
        if not os.path.exists(self.trace_path):
            return []
        out = []
        with open(self.trace_path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    warnings.warn(
                        f"skipping unreadable trace line {lineno} in "
                        f"{self.trace_path}: {exc}",
                        RuntimeWarning, stacklevel=2,
                    )
        return out

    def truncate_trace(self, n: int) -> None:
        """Keep only the first ``n`` trace entries; drop everything after.

        Used on resume to discard *orphan* completions --- entries a date wrote
        live before the process was killed, whose ledger row never landed. Left
        in place they would desynchronise the trace from the ledger and corrupt
        replay (the re-run date would consume the orphans instead of its own
        completions). Always rewrites the file from the parsed entries (when it
        exists), so a half-written final line a hard kill may have left is
        dropped too --- even when the count of parseable entries already equals
        ``n`` and nothing would otherwise be removed.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if not os.path.exists(self.trace_path):
            return
        entries = self.read_trace()[:n]
        payload = "".join(
            json.dumps(_redact_obj(entry), ensure_ascii=False) + "\n"
            for entry in entries
        ).encode("utf-8")
        self._atomic_write(self.trace_path, payload)

    def reset(self) -> None:
        """Remove ledger + trace + manifest for a fresh run under this run_name.

        The manifest is cleared too so a fresh run that crashes before it writes
        a new manifest cannot leave the *previous* run's manifest sitting next to
        a fresh partial ledger/trace (which would make ``replay``/``log`` read a
        stale config against new data).
        """
        for path in (self.ledger_path, self.trace_path, self.manifest_path):
            if os.path.exists(path):
                os.remove(path)
