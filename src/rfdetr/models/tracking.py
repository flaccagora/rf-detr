# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tensor contracts shared by persistent-query tracking components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _validate_boxes(boxes: Tensor, *, name: str) -> None:
    """Validate normalized ``cxcywh`` boxes at the tracking-state boundary.

    Args:
        boxes: Tensor whose final dimension contains normalized ``cxcywh`` boxes.
        name: Field name included in validation errors.

    Raises:
        ValueError: If boxes are non-finite or outside the normalized box domain.
    """
    if not torch.isfinite(boxes).all():
        raise ValueError(f"{name} must contain only finite values")
    if ((boxes < 0) | (boxes > 1)).any():
        raise ValueError(f"{name} must contain normalized cxcywh values in [0, 1]")


@dataclass(frozen=True)
class TrackQueryState:
    """Explicit neural state for a fixed set of decoder query slots.

    ``active_mask`` is the sole validity indicator. Feature and box values at
    inactive positions are padding and have no historical-state semantics.
    """

    query_features: Tensor
    reference_boxes: Tensor
    active_mask: Tensor

    def __post_init__(self) -> None:
        """Validate slot alignment, tensor semantics, device, and dtype."""
        if self.query_features.ndim != 3:
            raise ValueError("query_features must have shape [batch, queries, hidden_dim]")
        if self.reference_boxes.ndim != 3 or self.reference_boxes.shape[-1] != 4:
            raise ValueError("reference_boxes must have shape [batch, queries, 4]")
        if self.active_mask.ndim != 2:
            raise ValueError("active_mask must have shape [batch, queries]")

        slot_shape = self.query_features.shape[:2]
        if self.reference_boxes.shape[:2] != slot_shape or self.active_mask.shape != slot_shape:
            raise ValueError("query_features, reference_boxes, and active_mask must be slot-aligned")
        if not self.query_features.is_floating_point() or not self.reference_boxes.is_floating_point():
            raise TypeError("query_features and reference_boxes must use floating-point dtypes")
        if self.query_features.dtype != self.reference_boxes.dtype:
            raise TypeError("query_features and reference_boxes must have the same dtype")
        if self.active_mask.dtype != torch.bool:
            raise TypeError("active_mask must have dtype torch.bool")
        devices = {self.query_features.device, self.reference_boxes.device, self.active_mask.device}
        if len(devices) != 1:
            raise ValueError("query_features, reference_boxes, and active_mask must be on the same device")
        if not torch.isfinite(self.query_features).all():
            raise ValueError("query_features must contain only finite values")
        _validate_boxes(self.reference_boxes, name="reference_boxes")

    @classmethod
    def empty(
        cls,
        batch_size: int,
        num_queries: int,
        hidden_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> TrackQueryState:
        """Create an explicitly inactive, fixed-shape state.

        Args:
            batch_size: Number of independent streams in the batch.
            num_queries: Fixed decoder slot count.
            hidden_dim: Decoder query feature width.
            device: Device for all state tensors.
            dtype: Floating-point dtype for features and boxes.

        Returns:
            A validated state whose slots are all inactive.

        Raises:
            ValueError: If a requested dimension is not positive.
            TypeError: If ``dtype`` is not floating point.
        """
        dimensions = {
            "batch_size": batch_size,
            "num_queries": num_queries,
            "hidden_dim": hidden_dim,
        }
        for name, value in dimensions.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        resolved_dtype = dtype or torch.get_default_dtype()
        if not resolved_dtype.is_floating_point:
            raise TypeError("dtype must be floating point")
        return cls(
            query_features=torch.zeros(
                (batch_size, num_queries, hidden_dim),
                device=device,
                dtype=resolved_dtype,
            ),
            reference_boxes=torch.zeros(
                (batch_size, num_queries, 4),
                device=device,
                dtype=resolved_dtype,
            ),
            active_mask=torch.zeros(
                (batch_size, num_queries),
                device=device,
                dtype=torch.bool,
            ),
        )


@dataclass(frozen=True)
class TrackingFrameOutput:
    """Slot-aligned output of one low-level tracking forward pass."""

    pred_logits: Tensor
    pred_boxes: Tensor
    candidate_state: TrackQueryState
    input_active_mask: Tensor
    aux_outputs: tuple[dict[str, Tensor], ...] = ()
    enc_outputs: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate prediction and candidate-state slot alignment."""
        if self.pred_logits.ndim != 3:
            raise ValueError("pred_logits must have shape [batch, queries, classes]")
        if self.pred_boxes.ndim != 3 or self.pred_boxes.shape[-1] != 4:
            raise ValueError("pred_boxes must have shape [batch, queries, 4]")

        slot_shape = self.pred_logits.shape[:2]
        if self.pred_boxes.shape[:2] != slot_shape:
            raise ValueError("pred_logits and pred_boxes must be slot-aligned")
        if self.candidate_state.query_features.shape[:2] != slot_shape:
            raise ValueError("candidate_state must be slot-aligned with frame predictions")
        if self.input_active_mask.shape != slot_shape or self.input_active_mask.dtype != torch.bool:
            raise ValueError("input_active_mask must be a boolean [batch, queries] role map")

        tensors = (
            self.pred_logits,
            self.pred_boxes,
            self.candidate_state.query_features,
            self.input_active_mask,
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("frame predictions, candidate_state, and input_active_mask must share a device")
        if not self.pred_logits.is_floating_point() or not self.pred_boxes.is_floating_point():
            raise TypeError("pred_logits and pred_boxes must use floating-point dtypes")
        if self.pred_boxes.dtype != self.candidate_state.reference_boxes.dtype:
            raise TypeError("pred_boxes and candidate reference boxes must have the same dtype")
        if not torch.isfinite(self.pred_logits).all():
            raise ValueError("pred_logits must contain only finite values")
        if not torch.isfinite(self.pred_boxes).all():
            raise ValueError("pred_boxes must contain only finite values")


# Short form retained for callers that name the result after its frame scope.
TrackFrameOutput = TrackingFrameOutput

__all__ = ["TrackFrameOutput", "TrackQueryState", "TrackingFrameOutput"]
