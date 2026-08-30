"""Locking and immutable-input fingerprints for the one-shot v4 test."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(payload: object) -> str:
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ExclusiveRunLock:
    """Fixed-path O_EXCL lock with auditable stale-lock recovery on resume."""

    def __init__(self, path: Path, *, resume: bool):
        self.path = path
        self.resume = bool(resume)
        self.inode: int | None = None
        self.stale_archive: str | None = None
        self.payload: dict[str, Any] | None = None

    def acquire(self) -> dict[str, Any]:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError:
            if not self.resume:
                raise RuntimeError(
                    f"formal test_v4 global lock already exists: {self.path}"
                )
            recovery = self.path.with_name(f"{self.path.name}.recovery")
            try:
                recovery_descriptor = os.open(recovery, flags, 0o600)
            except FileExistsError as error:
                raise RuntimeError("another process is auditing the stale formal lock") from error
            try:
                os.write(recovery_descriptor, f"{os.getpid()}\n".encode("ascii"))
                os.fsync(recovery_descriptor)
                initial_inode = self.path.stat().st_ino
                try:
                    prior = json.loads(self.path.read_text(encoding="utf-8"))
                    prior_pid = int(prior["pid"])
                except Exception as error:
                    raise RuntimeError(
                        f"cannot audit existing formal lock: {self.path}"
                    ) from error
                if _pid_alive(prior_pid):
                    raise RuntimeError(
                        f"another formal test_v4 process is active: pid={prior_pid}"
                    )
                if self.path.stat().st_ino != initial_inode:
                    raise RuntimeError("formal lock changed during stale recovery")
                archive = self.path.with_name(
                    f"{self.path.name}.stale.{prior_pid}.{os.getpid()}"
                )
                self.path.replace(archive)
                self.stale_archive = str(archive)
                descriptor = os.open(self.path, flags, 0o600)
            finally:
                os.close(recovery_descriptor)
                recovery.unlink(missing_ok=True)
        payload = {
            "protocol": "test_v4_locked_global_exclusive_lock",
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "created_utc": _utc(),
            "resume": self.resume,
            "stale_lock_archive": self.stale_archive,
            "recovered_lock_payload": prior if self.stale_archive else None,
        }
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        os.write(descriptor, encoded)
        os.fsync(descriptor)
        self.inode = os.fstat(descriptor).st_ino
        os.close(descriptor)
        self.payload = payload
        return payload

    def bind_control_plane(self, seal_sha256: str) -> None:
        """Persist the post-generation control seal in the live lock inode."""

        if self.inode is None or self.payload is None:
            raise RuntimeError("formal lock is not held")
        if self.path.stat().st_ino != self.inode:
            raise RuntimeError("formal lock inode changed before control-plane binding")
        self.payload["control_plane_seal_sha256"] = str(seal_sha256)
        encoded = (json.dumps(self.payload, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(self.path, os.O_WRONLY)
        try:
            if os.fstat(descriptor).st_ino != self.inode:
                raise RuntimeError("formal lock inode changed during control-plane binding")
            os.ftruncate(descriptor, 0)
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def release(self) -> None:
        if self.inode is None:
            return
        try:
            current = self.path.stat()
        except FileNotFoundError:
            return
        if current.st_ino != self.inode:
            raise RuntimeError("formal lock inode changed while the run was active")
        self.path.unlink()
        self.inode = None

    def __enter__(self) -> "ExclusiveRunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def git_tracked_files(workspace: Path, scope: Iterable[str]) -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", *scope],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    return [workspace / value.decode("utf-8") for value in completed.stdout.split(b"\0") if value]


def evidence_fingerprint(
    *,
    workspace: Path,
    source_scope: Iterable[str],
    asset_paths: Iterable[Path],
) -> dict[str, Any]:
    scope = list(source_scope)
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *scope,
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"formal evidence source scope is dirty:\n{status}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout.strip()
    sources = git_tracked_files(workspace, scope)
    unique_paths: set[Path] = set()
    for candidate in [*sources, *asset_paths]:
        if candidate.is_symlink():
            raise RuntimeError(f"formal evidence input cannot be a symlink: {candidate}")
        unique_paths.add(candidate.resolve())
    unique = sorted(unique_paths)
    files: dict[str, Any] = {}
    for path in unique:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"formal evidence input is missing or a symlink: {path}")
        try:
            key = str(path.relative_to(workspace.resolve()))
        except ValueError:
            key = str(path)
        files[key] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    payload = {
        "git_commit": commit,
        "git_tree": tree,
        "source_scope": scope,
        "source_scope_clean": True,
        "files": files,
    }
    payload["digest"] = json_digest(payload)
    return payload


def assert_evidence_fingerprint(
    expected: dict[str, Any],
    current: dict[str, Any],
    *,
    context: str,
) -> None:
    if expected != current:
        expected_files = expected.get("files", {})
        current_files = current.get("files", {})
        changed = sorted(
            key
            for key in set(expected_files) | set(current_files)
            if expected_files.get(key) != current_files.get(key)
        )
        raise RuntimeError(
            "formal source/assets fingerprint changed during the locked run; "
            f"context={context}, expected_digest={expected.get('digest')}, "
            f"current_digest={current.get('digest')}, changed_files={changed}"
        )


__all__ = [
    "ExclusiveRunLock",
    "assert_evidence_fingerprint",
    "evidence_fingerprint",
    "git_tracked_files",
    "json_digest",
    "sha256_file",
]
