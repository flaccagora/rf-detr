# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pure host-side transitions for persistent track-query slots."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor

from rfdetr.config import TrackingSessionConfig
from rfdetr.models.tracking import TrackingFrameOutput, TrackQueryState

TrackStatus = Literal["inactive", "active", "suspended"]
EventKind = Literal["activated", "suspended", "recovered", "terminated", "duplicate_suppressed", "capacity_suppressed"]


@dataclass(frozen=True)
class TrackSlot:
    """Lifecycle metadata for one fixed decoder query position."""

    track_id: int | None = None
    status: TrackStatus = "inactive"
    age: int = 0
    hits: int = 0
    missed_frames: int = 0
    last_reliable_frame: int | None = None
    confidence: float | None = None
    class_id: int | None = None

    def __post_init__(self) -> None:
        """Validate identity and status consistency."""
        if (self.status == "inactive") != (self.track_id is None):
            raise ValueError("inactive slots must have no track_id and tracked slots must have one")
        if self.status != "inactive" and self.last_reliable_frame is None:
            raise ValueError("tracked slots must record their last reliable frame")
        if min(self.age, self.hits, self.missed_frames) < 0:
            raise ValueError("slot counters cannot be negative")


@dataclass(frozen=True)
class TrackSlotTable:
    """Immutable lifecycle metadata aligned with fixed neural query slots."""

    slots: tuple[TrackSlot, ...]
    next_track_id: int = 0
    last_frame_index: int | None = None

    def __post_init__(self) -> None:
        """Validate table shape, identity uniqueness, and monotonic counter."""
        if not self.slots:
            raise ValueError("slot table must contain at least one query slot")
        if self.next_track_id < 0:
            raise ValueError("next_track_id cannot be negative")
        track_ids = [slot.track_id for slot in self.slots if slot.track_id is not None]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("track IDs must be unique within a slot table")
        if track_ids and self.next_track_id <= max(track_ids):
            raise ValueError("next_track_id must be greater than every assigned track ID")

    @classmethod
    def empty(cls, num_queries: int) -> TrackSlotTable:
        """Create an inactive table for a fixed decoder capacity."""
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        return cls(slots=tuple(TrackSlot() for _ in range(num_queries)))

    @property
    def active_mask(self) -> tuple[bool, ...]:
        """Return which slots must enter the decoder as persistent queries."""
        return tuple(slot.status != "inactive" for slot in self.slots)

    @property
    def slot_track_ids(self) -> tuple[int | None, ...]:
        """Return track identity aligned with decoder query positions."""
        return tuple(slot.track_id for slot in self.slots)


@dataclass(frozen=True)
class LifecycleEvent:
    """Diagnostic record emitted by one lifecycle transition."""

    kind: EventKind
    slot: int
    track_id: int | None
    frame_index: int


@dataclass(frozen=True)
class LifecycleTransition:
    """New host metadata and committed neural state from one pure update."""

    table: TrackSlotTable
    state: TrackQueryState
    events: tuple[LifecycleEvent, ...]


def _box_iou(box_a: Tensor, box_b: Tensor) -> float:
    """Compute IoU between two normalized ``cxcywh`` boxes."""
    a_xy1 = box_a[:2] - box_a[2:] / 2
    a_xy2 = box_a[:2] + box_a[2:] / 2
    b_xy1 = box_b[:2] - box_b[2:] / 2
    b_xy2 = box_b[:2] + box_b[2:] / 2
    intersection = (torch.minimum(a_xy2, b_xy2) - torch.maximum(a_xy1, b_xy1)).clamp(min=0).prod()
    union = box_a[2:].prod() + box_b[2:].prod() - intersection
    return float((intersection / union).item()) if union > 0 else 0.0


def _validate_transition_inputs(
    table: TrackSlotTable,
    state: TrackQueryState,
    frame: TrackingFrameOutput,
    max_active_tracks: int,
    frame_index: int,
) -> None:
    """Reject host/neural misalignment before applying policy."""
    num_queries = len(table.slots)
    if state.query_features.shape[0] != 1 or frame.pred_logits.shape[0] != 1:
        raise ValueError("lifecycle transitions currently require a single-stream batch")
    if state.query_features.shape[1] != num_queries or frame.pred_logits.shape[1] != num_queries:
        raise ValueError("slot table, prior state, and frame output must be query-aligned")
    if (
        state.query_features.device != frame.candidate_state.query_features.device
        or state.query_features.dtype != frame.candidate_state.query_features.dtype
    ):
        raise ValueError("prior and candidate neural state must have matching device and dtype")
    expected_mask = torch.tensor(table.active_mask, device=state.active_mask.device).unsqueeze(0)
    if not torch.equal(state.active_mask, expected_mask):
        raise ValueError("slot table and neural active mask must agree")
    if not torch.equal(frame.input_active_mask.to(expected_mask.device), expected_mask):
        raise ValueError("frame input role map must agree with the committed prior state")
    if not 0 <= max_active_tracks < num_queries:
        raise ValueError("max_active_tracks must be nonnegative and leave a discovery slot")
    if table.last_frame_index is not None and frame_index <= table.last_frame_index:
        raise ValueError("frame_index must increase monotonically")


def transition_lifecycle(
    table: TrackSlotTable,
    state: TrackQueryState,
    frame: TrackingFrameOutput,
    config: TrackingSessionConfig,
    *,
    max_active_tracks: int,
    frame_index: int,
) -> LifecycleTransition:
    """Apply deterministic activation, suspension, recovery, termination, and recycling.

    Args:
        table: Current host lifecycle metadata.
        state: Last committed trusted neural state.
        frame: Slot-aligned candidate output for the current frame.
        config: Inference lifecycle thresholds.
        max_active_tracks: Capacity reserved for active and suspended tracks.
        frame_index: Monotonically increasing source-frame index.

    Returns:
        A new immutable table, committed state, and ordered diagnostic events.
    """
    _validate_transition_inputs(table, state, frame, max_active_tracks, frame_index)
    scores, classes = frame.pred_logits[0].sigmoid().max(dim=-1)
    slots = list(table.slots)
    features = state.query_features.clone()
    boxes = state.reference_boxes.clone()
    active_mask = state.active_mask.clone()
    events: list[LifecycleEvent] = []

    for index, old_slot in enumerate(table.slots):
        if old_slot.status == "inactive":
            continue
        score = float(scores[index].item())
        if score >= config.continuation_threshold:
            kind: EventKind | None = "recovered" if old_slot.status == "suspended" else None
            slots[index] = replace(
                old_slot,
                status="active",
                age=old_slot.age + 1,
                hits=old_slot.hits + 1,
                missed_frames=0,
                last_reliable_frame=frame_index,
                confidence=score,
                class_id=int(classes[index].item()),
            )
            features[:, index] = frame.candidate_state.query_features[:, index]
            boxes[:, index] = frame.candidate_state.reference_boxes[:, index]
            if kind is not None:
                events.append(LifecycleEvent(kind, index, old_slot.track_id, frame_index))
            continue

        assert old_slot.last_reliable_frame is not None
        missed_frames = frame_index - old_slot.last_reliable_frame
        if missed_frames > config.max_missed_frames:
            slots[index] = TrackSlot()
            features[:, index] = 0
            boxes[:, index] = 0
            active_mask[:, index] = False
            events.append(LifecycleEvent("terminated", index, old_slot.track_id, frame_index))
        else:
            slots[index] = replace(
                old_slot,
                status="suspended",
                age=old_slot.age + 1,
                missed_frames=missed_frames,
                confidence=score,
            )
            if old_slot.status == "active":
                events.append(LifecycleEvent("suspended", index, old_slot.track_id, frame_index))

    tracked_count = sum(slot.status != "inactive" for slot in slots)
    next_track_id = table.next_track_id
    tracked_boxes = [(slot.class_id, boxes[0, index]) for index, slot in enumerate(slots) if slot.status != "inactive"]
    for index, old_slot in enumerate(table.slots):
        if old_slot.status != "inactive" or float(scores[index].item()) < config.activation_threshold:
            continue
        class_id = int(classes[index].item())
        candidate_box = frame.candidate_state.reference_boxes[0, index]
        if any(
            tracked_class == class_id and _box_iou(candidate_box, tracked_box) > config.duplicate_iou_threshold
            for tracked_class, tracked_box in tracked_boxes
        ):
            events.append(LifecycleEvent("duplicate_suppressed", index, None, frame_index))
            continue
        if tracked_count >= max_active_tracks:
            events.append(LifecycleEvent("capacity_suppressed", index, None, frame_index))
            continue
        confidence = float(scores[index].item())
        slots[index] = TrackSlot(
            track_id=next_track_id,
            status="active",
            age=1,
            hits=1,
            last_reliable_frame=frame_index,
            confidence=confidence,
            class_id=class_id,
        )
        features[:, index] = frame.candidate_state.query_features[:, index]
        boxes[:, index] = candidate_box
        active_mask[:, index] = True
        tracked_boxes.append((class_id, boxes[0, index]))
        events.append(LifecycleEvent("activated", index, next_track_id, frame_index))
        tracked_count += 1
        next_track_id += 1

    return LifecycleTransition(
        table=TrackSlotTable(tuple(slots), next_track_id, frame_index),
        state=TrackQueryState(features, boxes, active_mask),
        events=tuple(events),
    )


__all__ = [
    "LifecycleEvent",
    "LifecycleTransition",
    "TrackSlot",
    "TrackSlotTable",
    "transition_lifecycle",
]
