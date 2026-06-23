"""Durable artifacts with fingerprint sidecars and a workdir lock.

Every artifact `<file>` is paired with a sidecar `<file>.meta.json` holding
{input_fingerprint, output_sha256, stage, written_at}. Writes are atomic:
content goes to a temp file in the same directory, is renamed into place, and
the sidecar is written last — so a crash at any point leaves either no
artifact or an artifact without a (matching) sidecar, never a torn one that
validates. An artifact whose content no longer matches its sidecar's output
hash (hand-edited or torn) is treated as absent and recomputed (R1, R16).
"""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

META_SUFFIX = ".meta.json"
TMP_PREFIX = ".tmp."


def fingerprint(inputs: dict) -> str:
    """Stable hash of a JSON-serializable dict, independent of key order."""
    canon = json.dumps(inputs, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def video_fingerprint(path: str | Path) -> dict:
    """Cheap identity for multi-GB inputs: path + size + mtime, no content read."""
    p = Path(path).resolve()
    st = p.stat()
    return {"path": str(p), "size": st.st_size, "mtime": st.st_mtime}


def meta_path(artifact: Path) -> Path:
    return artifact.with_name(artifact.name + META_SUFFIX)


def read_meta(artifact: Path) -> dict | None:
    try:
        return json.loads(meta_path(artifact).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _stat_matches(meta: dict, path: Path) -> bool:
    try:
        st = path.stat()
    except OSError:
        return False
    return meta.get("size") == st.st_size and meta.get("mtime") == st.st_mtime


def is_valid(artifact: Path, inputs: dict) -> bool:
    if not artifact.exists():
        return False
    meta = read_meta(artifact)
    if meta is None:
        return False
    if meta.get("input_fingerprint") != fingerprint(inputs):
        return False
    # Fast path: unchanged size+mtime since the sidecar was written means the
    # content is what we hashed — skips re-reading multi-GB artifacts on every
    # rerun. Any edit touches mtime, falling through to the full content hash
    # (which is how hand-edited artifacts are detected and recomputed, R16).
    if _stat_matches(meta, artifact):
        return True
    return meta.get("output_sha256") == file_sha256(artifact)


def cached_file_sha256(path: str | Path) -> str:
    """Content hash of a file, served from its artifact sidecar when the
    size+mtime still match; falls back to hashing for non-artifacts."""
    p = Path(path)
    meta = read_meta(p)
    if meta and "output_sha256" in meta and _stat_matches(meta, p):
        return meta["output_sha256"]
    return file_sha256(p)


def _tmp_for(path: Path) -> Path:
    # Keep the real name (and so the extension) last: tools like ffmpeg infer
    # the output format from it.
    return path.with_name(f"{TMP_PREFIX}{uuid.uuid4().hex}.{path.name}")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = _tmp_for(path)
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_unfinalized(artifact: Path, data: bytes) -> None:
    """Write an artifact body WITHOUT a sidecar: the content is durable (a
    later compute may reuse parts of the body) but the artifact stays invalid,
    so the next run recomputes it. Used for translate batches that contain
    failed segments (R5b)."""
    artifact.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(artifact, data)
    meta_path(artifact).unlink(missing_ok=True)


def _write_sidecar(artifact: Path, inputs: dict, stage: str) -> None:
    st = artifact.stat()
    meta = {
        "input_fingerprint": fingerprint(inputs),
        "output_sha256": file_sha256(artifact),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "stage": stage,
        "written_at": time.time(),
    }
    atomic_write_bytes(meta_path(artifact), json.dumps(meta).encode("utf-8"))


def produce(artifact: Path, inputs: dict, stage: str, build: Callable[[Path], None]) -> bool:
    """Load-or-compute one artifact. Returns True when the existing artifact
    was valid (build not called), False when it was (re)computed.

    `build(tmp_path)` must write the artifact content to `tmp_path`; the
    harness renames it into place and writes the sidecar last.
    """
    if is_valid(artifact, inputs):
        return True
    artifact.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_for(artifact)
    try:
        build(tmp)
        os.replace(tmp, artifact)
    finally:
        tmp.unlink(missing_ok=True)
    _write_sidecar(artifact, inputs, stage)
    return False


def produce_json(artifact: Path, inputs: dict, stage: str, compute: Callable[[], Any]) -> Any:
    """Load-or-compute a JSON artifact; returns the parsed payload either way."""
    def build(tmp: Path) -> None:
        tmp.write_text(json.dumps(compute(), ensure_ascii=False, indent=1), encoding="utf-8")

    produce(artifact, inputs, stage, build)
    return json.loads(artifact.read_text(encoding="utf-8"))


class LockError(RuntimeError):
    pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OverflowError:
        return False
    return True


class WorkdirLock:
    """`run.lock` guards a workdir against concurrent invocations (R24).

    The lock records the owner pid; a lock whose pid is no longer alive is
    stale and silently broken.
    """

    def __init__(self, workdir: Path):
        self.path = Path(workdir) / "run.lock"
        self._owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):  # second pass after breaking a stale lock
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                pid = self._holder_pid()
                if pid is not None and _pid_alive(pid):
                    raise LockError(
                        f"workdir is in use by process {pid} (lock: {self.path}); "
                        "wait for the other run to finish, or delete the lock if you are sure it is dead"
                    )
                # Claim the stale lock atomically: exactly one contender wins
                # the rename; a plain unlink could delete a lock another
                # contender just re-created.
                claim = self.path.with_name(f".stale.{uuid.uuid4().hex}.lock")
                try:
                    os.rename(self.path, claim)
                except FileNotFoundError:
                    continue  # someone else claimed it; retry the O_EXCL open
                claim.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w") as f:
                json.dump({"pid": os.getpid(), "acquired_at": time.time()}, f)
            self._owned = True
            return
        raise LockError(f"could not acquire {self.path} after breaking stale lock")

    def _holder_pid(self) -> int | None:
        try:
            return int(json.loads(self.path.read_text())["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None  # corrupt lock -> treat as stale

    def release(self) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
