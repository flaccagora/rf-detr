# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Deployment-facing loader for locked lifecycle policy artifacts.

Every consumer of a locked policy -- ``TrackingSession``, chronological
validation, benchmark CLIs, and deployment entry points -- must deserialize
the on-disk artifact (e.g. ``tracking_policy.lock.json``) through
:func:`load_tracking_policy` into the same :class:`~rfdetr.config.TrackingPolicy`
type, so a tampered, mismatched, or malformed lock file fails closed
identically everywhere instead of each caller re-implementing its own
parsing and integrity checks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rfdetr.config import TRACKING_POLICY_LOCK_SCHEMA_VERSION, TrackingPolicy


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_of(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def load_tracking_policy(path: str | Path) -> TrackingPolicy:
    """Load, integrity-check, and deserialize a locked lifecycle policy artifact.

    Args:
        path: Path to a ``tracking_policy.lock.json``-shaped artifact.

    Returns:
        The locked :class:`~rfdetr.config.TrackingPolicy`.

    Raises:
        ValueError: If the artifact's schema version is unsupported, its
            fields do not validate as a ``TrackingPolicy``, its declared
            ``policy_hash`` does not match its own fields, or its
            ``lock_hash`` does not match the rest of the payload (indicating
            the artifact was modified after it was written).
    """
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != TRACKING_POLICY_LOCK_SCHEMA_VERSION:
        raise ValueError("unsupported tracking policy lock schema_version")
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("tracking policy lock is missing its fields")
    policy = TrackingPolicy.model_validate(fields)
    if payload.get("policy_hash") != policy.sha256():
        raise ValueError("tracking policy lock policy_hash does not match its fields")
    if "lock_hash" not in payload:
        raise ValueError("tracking policy lock is missing its lock_hash")
    expected_lock_hash = _sha256_of({key: value for key, value in payload.items() if key != "lock_hash"})
    if payload["lock_hash"] != expected_lock_hash:
        raise ValueError("tracking policy lock has been modified since it was written")
    return policy


__all__ = ["load_tracking_policy"]
