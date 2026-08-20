# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Checkpoint conversion utilities for the PTL training stack.

Provides :func:`convert_legacy_checkpoint` to convert RF-DETR ``*.pth`` checkpoints (produced by the pre-PTL
``engine.py`` training loop) into the ``*.ckpt`` format expected by ``pytorch_lightning.Trainer``.

Auto-detection of legacy format at load time is handled by
:meth:`rfdetr.training.module_model.RFDETRModelModule.on_load_checkpoint`.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
_DEPRECATED_TRAIN_ARCHITECTURE_FIELDS = frozenset({"group_detr", "ia_bce_loss", "num_select", "segmentation_head"})
_SOURCE_HASH_UNSET = object()

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "authoritative_checkpoint_metadata",
    "convert_legacy_checkpoint",
    "serialize_model_config",
    "serialize_train_config",
    "source_checkpoint_hash",
]


def serialize_model_config(model_config: object, state_dict: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Serialize architecture configuration, syncing fields encoded by the weights."""
    if isinstance(model_config, dict):
        return model_config
    else:
        model_dump = getattr(model_config, "model_dump", None)
        if not callable(model_dump):
            return None
        dumped = model_dump(mode="json")
        if not isinstance(dumped, dict):
            return None

    if state_dict is None:
        return dumped
    keypoint_mask = state_dict.get("_kp_active_mask")
    if isinstance(keypoint_mask, torch.Tensor) and keypoint_mask.ndim == 2 and "num_keypoints_per_class" in dumped:
        dumped["num_keypoints_per_class"] = [int(count) for count in keypoint_mask.sum(dim=1).tolist()]
    class_weight = state_dict.get("class_embed.weight")
    if isinstance(class_weight, torch.Tensor) and class_weight.ndim == 2 and "num_classes" in dumped:
        dumped["num_classes"] = class_weight.shape[0] - 1
    return dumped


def serialize_train_config(train_config: object) -> dict[str, Any]:
    """Return training metadata without deprecated architecture duplicates."""
    if isinstance(train_config, dict):
        dumped = dict(train_config)
    else:
        model_dump = getattr(train_config, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump(mode="json")
        else:
            try:
                dumped = dict(vars(train_config))
            except TypeError:
                dumped = {}
    return {key: value for key, value in dumped.items() if key not in _DEPRECATED_TRAIN_ARCHITECTURE_FIELDS}


def source_checkpoint_hash(model_config: object) -> str | None:
    """Hash the source weights once so derived checkpoints retain their lineage."""
    source = (
        model_config.get("pretrain_weights")
        if isinstance(model_config, dict)
        else getattr(model_config, "pretrain_weights", None)
    )
    if source is None:
        return None
    try:
        path = Path(os.fspath(source))
    except TypeError:
        return None
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authoritative_checkpoint_metadata(
    *,
    model_config: object,
    train_config: object,
    state_dict: dict[str, Any] | None,
    epoch: int,
    weight_flavor: str,
    source_checkpoint_hash_value: str | None | object = _SOURCE_HASH_UNSET,
) -> dict[str, Any]:
    """Build the single self-describing metadata contract used by every new checkpoint."""
    serialized_model = serialize_model_config(model_config, state_dict)
    serialized_train = serialize_train_config(train_config)
    class_schema = serialized_model.get("class_schema") if serialized_model is not None else None
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_config_type": type(model_config).__name__,
        "model_config": serialized_model,
        "class_schema": class_schema,
        "train_config": serialized_train,
        "args": serialized_train,
        "epoch": int(epoch),
        "weight_flavor": weight_flavor,
        "source_checkpoint_hash": (
            source_checkpoint_hash(model_config)
            if source_checkpoint_hash_value is _SOURCE_HASH_UNSET
            else source_checkpoint_hash_value
        ),
    }


def convert_legacy_checkpoint(old_path: str, new_path: str) -> None:
    """Convert a legacy RF-DETR ``.pth`` checkpoint to PTL ``.ckpt`` format.

    Loads a checkpoint saved by the pre-PTL ``engine.py`` training loop and rewrites it in the structure expected by
    ``pytorch_lightning.Trainer``:

    * ``state_dict`` keys are prefixed with ``"model."`` to match the
      attribute path inside :class:`~rfdetr.training.module_model.RFDETRModelModule`.
    * ``args`` (``argparse.Namespace`` or ``dict``) is normalised to a plain
      ``dict`` and stored as ``hyper_parameters``.
    * ``legacy_checkpoint_format: True`` is written so
      :meth:`~rfdetr.training.module_model.RFDETRModelModule.on_load_checkpoint` can distinguish converted files from
      native PTL checkpoints.
    * If an ``ema_model`` key is present it is preserved verbatim under
      ``legacy_ema_state_dict`` for optional EMA weight restoration.

    Args:
        old_path: Path to the source legacy ``.pth`` checkpoint.
        new_path: Destination path for the converted ``.ckpt`` file.
    """
    # trust=True: this function converts internally-produced legacy .pth files;
    # allow pickle fallback if safe deserialization fails due to non-tensor/custom objects.
    from rfdetr.util.io import _safe_torch_load

    old: dict[str, Any] = _safe_torch_load(old_path, trust=True)

    if "model" not in old:
        raise ValueError(
            f"The checkpoint at {old_path!r} does not contain a 'model' key."
            " Only RF-DETR legacy .pth files produced by engine.py are supported."
        )

    args_obj = old.get("args")
    if isinstance(args_obj, dict):
        hyper_parameters: dict[str, Any] = args_obj
    elif args_obj is None:
        hyper_parameters = {}
    else:
        try:
            hyper_parameters = vars(args_obj)
        except TypeError:
            logger.warning(
                "Cannot extract hyper_parameters from args of type %s; storing empty dict.",
                type(args_obj).__name__,
            )
            hyper_parameters = {}

    new: dict[str, Any] = {
        "state_dict": {"model." + k: v for k, v in old["model"].items()},
        "epoch": old.get("epoch", 0),
        "global_step": 0,
        "hyper_parameters": hyper_parameters,
        "legacy_checkpoint_format": True,
    }

    if "ema_model" in old:
        # Preserve EMA weights under a dedicated key.  Callback-specific state
        # keys are framework-internal and must not be written here.
        new["legacy_ema_state_dict"] = old["ema_model"]

    torch.save(new, new_path)
