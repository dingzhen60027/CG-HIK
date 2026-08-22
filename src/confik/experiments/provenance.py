from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import sklearn


def _canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def source_tree_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def ensure_protocol_manifest(
    path: str | Path,
    config: dict[str, Any],
    robot: str,
) -> dict[str, Any]:
    """Freeze code/config identity and refuse stale artifact reuse."""
    canonical = _canonical_config(config)
    payload = {
        "protocol_version": canonical.get("protocol_version"),
        "experiment_name": canonical.get("experiment_name"),
        "robot": robot,
        "config_sha256": _json_hash(canonical),
        "source_tree_sha256": source_tree_hash(),
    }
    payload["run_fingerprint"] = _json_hash(payload)
    manifest_path = Path(path)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in payload}
        if comparable != payload:
            raise RuntimeError(
                "experiment code/config changed after artifacts were created; "
                "use a new experiment_name instead of reusing the test directory"
            )
        return existing
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def environment_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import scipy

        payload["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover
        payload["scipy"] = None
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:  # pragma: no cover
        payload["torch"] = None
    try:
        import pinocchio

        payload["pinocchio"] = getattr(pinocchio, "__version__", "unknown")
    except ImportError:  # pragma: no cover
        payload["pinocchio"] = None
    return payload
