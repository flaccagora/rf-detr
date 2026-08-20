# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pure host-side transitions for persistent track-query slots."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
from torch import Tensor

from rfdetr.config import ClassSchema, TrackingSessionConfig
from rfdetr.models.tracking import TrackingFrameOutput, TrackQueryState

TrackStatus = Literal["inactive", "tentative", "active", "suspended"]
EventKind = Literal[
    "tentative_started",
    "confirmed",
    "tentative_cancelled",
    "activated",
    "suspended",
    "recovered",
    "reassociated",
    "terminated",
    "duplicate_suppressed",
    "capacity_suppressed",
]
SuppressionReason = Literal[
    "duplicate_overlap",
    "durable_capacity",
    "tentative_capacity",
    "discovery_candidate_limit",
]


@dataclass(frozen=True)
class TrackSlot:
    """Lifecycle metadata for one fixed decoder query position."""

    track_id: int | None = None
    status: TrackStatus = "inactive"
    age: int = 0
    hits: int = 0
    missed_frames: int = 0
    last_reliable_frame: int | None = None
    tentative_started_frame: int | None = None
    confidence: float | None = None
    class_id: int | None = None
    last_reliable_box: tuple[float, float, float, float] | None = None
    velocity: tuple[float, float, float, float] | None = None
    identity_memory: tuple[tuple[float, tuple[float, ...]], ...] = ()

    def __post_init__(self) -> None:
        """Validate identity and status consistency."""
        if self.status in {"inactive", "tentative"} and self.track_id is not None:
            raise ValueError("inactive and tentative slots must have no track_id")
        if self.status in {"active", "suspended"} and self.track_id is None:
            raise ValueError("active and suspended slots must have a track_id")
        if self.status != "inactive" and self.last_reliable_frame is None:
            raise ValueError("tracked slots must record their last reliable frame")
        if (self.status == "tentative") != (self.tentative_started_frame is not None):
            raise ValueError("only tentative slots must record their first-hit frame")
        if min(self.age, self.hits, self.missed_frames) < 0:
            raise ValueError("slot counters cannot be negative")
        if self.status == "inactive" and (self.last_reliable_box is not None or self.velocity is not None):
            raise ValueError("inactive slots must not retain motion-prediction state")
        if self.velocity is not None and self.last_reliable_box is None:
            raise ValueError("a slot cannot carry a velocity estimate without a last reliable box")
        if self.status == "inactive" and self.identity_memory:
            raise ValueError("inactive slots must not retain identity memory")


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
    """Diagnostic record emitted by one lifecycle transition.

    Suppressions additionally carry the evidence that rejected a discovery so
    that sweeps and failure slices can attribute every unborn candidate.

    Attributes:
        kind: Transition that produced this record.
        slot: Fixed decoder query position the record belongs to.
        track_id: Public identity when the slot owns one.
        frame_index: Source-frame index of the transition.
        reason: Why a candidate was suppressed; ``None`` for other kinds.
        score: Foreground score of the suppressed candidate.
        class_id: Declared foreground class of the suppressed candidate.
        compared_status: Lifecycle status of the track a duplicate matched,
            or ``"suspended"`` for a reassociation.
        overlap: IoU between a duplicate or reassociation candidate and the
            compared track.
    """

    kind: EventKind
    slot: int
    track_id: int | None
    frame_index: int
    reason: SuppressionReason | None = None
    score: float | None = None
    class_id: int | None = None
    compared_status: TrackStatus | None = None
    overlap: float | None = None

    def __post_init__(self) -> None:
        """Require complete, kind-appropriate suppression and reassociation diagnostics."""
        suppressed = self.kind in {"duplicate_suppressed", "capacity_suppressed"}
        if suppressed != (self.reason is not None):
            raise ValueError("suppression events must record a reason and other events must not")
        if suppressed and (self.score is None or self.class_id is None):
            raise ValueError("suppression events must record the candidate score and class")
        carries_overlap_evidence = self.reason == "duplicate_overlap" or self.kind == "reassociated"
        if carries_overlap_evidence != (self.compared_status is not None and self.overlap is not None):
            raise ValueError("duplicate suppressions and reassociations must record the compared status and overlap")


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


def _motion_observation(
    old_slot: TrackSlot,
    new_box: Tensor,
    frame_index: int,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float] | None]:
    """Fold one fresh reliable box into a per-frame constant-velocity estimate.

    Velocity is expressed per unit of elapsed source-frame index, so it is
    unaffected by nonuniform frame spacing: doubling both the frame-index
    step and the elapsed interval leaves the estimate unchanged (PRD US-025,
    Section 7.7's "timestamps and nonuniform frame indices").

    Returns:
        The new last-reliable box, and an updated velocity estimate (``None``
        until a second reliable observation is available).
    """
    new_box_tuple = tuple(float(value) for value in new_box.tolist())
    velocity = None
    if old_slot.last_reliable_box is not None and old_slot.last_reliable_frame is not None:
        elapsed = frame_index - old_slot.last_reliable_frame
        if elapsed > 0:
            velocity = tuple(
                (new_box_tuple[dim] - old_slot.last_reliable_box[dim]) / elapsed for dim in range(4)
            )
        else:
            velocity = old_slot.velocity
    return new_box_tuple, velocity


def _initial_motion_state(
    config: TrackingSessionConfig, box: Tensor
) -> tuple[float, float, float, float] | None:
    """Seed a freshly (re)activated slot's last-reliable box, with no velocity yet.

    A slot needs two reliable observations before :func:`_motion_observation` can produce a
    velocity estimate, so a track suspended immediately after birth predicts no drift (its
    reference box stays put, exactly like the disabled baseline) until it has recurred at least
    once while active.
    """
    if not config.motion_reference_prediction_enabled:
        return None
    return tuple(float(value) for value in box.tolist())


def _predict_motion_box(
    last_reliable_box: tuple[float, float, float, float],
    velocity: tuple[float, float, float, float],
    elapsed_frames: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Extrapolate a suspended track's stale reference box via constant per-frame velocity.

    Always anchored at the last *reliable* box rather than compounded from
    the previous frame's prediction, so repeated calls across a long
    occlusion do not accumulate rounding drift.
    """
    base = torch.tensor(last_reliable_box, device=device, dtype=dtype)
    drift = torch.tensor(velocity, device=device, dtype=dtype) * elapsed_frames
    predicted = base + drift
    predicted = torch.cat(
        [predicted[:2].clamp(min=0.0, max=1.0), predicted[2:].clamp(min=1e-4, max=1.0)]
    )
    return predicted


def _identity_memory_update(
    memory: tuple[tuple[float, tuple[float, ...]], ...],
    feature: Tensor,
    weight: float,
    capacity: int,
) -> tuple[tuple[float, tuple[float, ...]], ...]:
    """Fold one reliable observation into a quality-weighted bounded memory bank.

    The bank keeps the ``capacity`` highest-weight observations it has ever
    been offered, not simply the most recent ones, so a track's identity
    representation stays anchored to its most confident evidence instead of
    drifting toward whatever a crossing or partial occlusion most recently
    produced.
    """
    feature_tuple = tuple(float(value) for value in feature.tolist())
    updated = memory + ((weight, feature_tuple),)
    if len(updated) > capacity:
        updated = tuple(sorted(updated, key=lambda entry: entry[0], reverse=True)[:capacity])
    return updated


def _identity_memory_mean(
    memory: tuple[tuple[float, tuple[float, ...]], ...],
) -> tuple[float, ...] | None:
    """Return the weight-averaged feature of a memory bank, or ``None`` when empty."""
    if not memory:
        return None
    total_weight = sum(weight for weight, _ in memory)
    dims = len(memory[0][1])
    if total_weight <= 0:
        return tuple(sum(feature[dim] for _, feature in memory) / len(memory) for dim in range(dims))
    return tuple(sum(weight * feature[dim] for weight, feature in memory) / total_weight for dim in range(dims))


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _identity_margin(
    candidate_feature: Tensor,
    own_memory: tuple[tuple[float, tuple[float, ...]], ...],
    other_memories: list[tuple[tuple[float, tuple[float, ...]], ...]],
) -> float | None:
    """Own-identity cosine similarity minus the best-matching other slot's similarity.

    Returns ``None`` when this slot has no memory yet, or no other same-class
    occupied slot carries memory to disambiguate against -- there is no
    ambiguity to resolve, so the continuation is accepted exactly as it would
    be with identity memory disabled.
    """
    own_mean = _identity_memory_mean(own_memory)
    if own_mean is None:
        return None
    other_means = [mean for mean in (_identity_memory_mean(memory) for memory in other_memories) if mean is not None]
    if not other_means:
        return None
    candidate_tuple = tuple(float(value) for value in candidate_feature.tolist())
    own_similarity = _cosine_similarity(candidate_tuple, own_mean)
    best_other_similarity = max(_cosine_similarity(candidate_tuple, mean) for mean in other_means)
    return own_similarity - best_other_similarity


def _duplicate_conflict(
    candidate_box: Tensor,
    class_id: int,
    occupied: list[tuple[int, int | None, TrackStatus, Tensor]],
) -> tuple[float, TrackStatus, int] | None:
    """Return the strongest same-class overlap against every occupied slot.

    Active, suspended, and tentative slots all reserve their region, so a
    discovery may not duplicate any of them. Callers may pre-filter
    ``occupied`` (e.g. to suspended slots only) to answer a narrower
    question such as reassociation eligibility.
    """
    conflicts = [
        (_box_iou(candidate_box, occupied_box), status, index)
        for index, occupied_class, status, occupied_box in occupied
        if occupied_class == class_id
    ]
    return max(conflicts, key=lambda conflict: conflict[0], default=None)


def _suppression(
    slot: int,
    frame_index: int,
    reason: SuppressionReason,
    score: float,
    class_id: int,
    *,
    compared_status: TrackStatus | None = None,
    overlap: float | None = None,
) -> LifecycleEvent:
    """Build a fully attributed suppression diagnostic for one candidate."""
    return LifecycleEvent(
        kind="duplicate_suppressed" if reason == "duplicate_overlap" else "capacity_suppressed",
        slot=slot,
        track_id=None,
        frame_index=frame_index,
        reason=reason,
        score=score,
        class_id=class_id,
        compared_status=compared_status,
        overlap=overlap,
    )


def foreground_scores(pred_logits: Tensor, class_schema: ClassSchema) -> tuple[Tensor, Tensor]:
    """Return each query's best declared-foreground score and model class ID."""
    declared_indices = (*class_schema.foreground_class_ids, class_schema.background_logit_index)
    expected_logits = max(index for index in declared_indices if index is not None) + 1
    if pred_logits.shape[-1] != expected_logits:
        raise ValueError(
            f"pred_logits has {pred_logits.shape[-1]} classes but class_schema declares {expected_logits} logits"
        )
    probabilities = (
        pred_logits.sigmoid() if class_schema.logit_activation == "sigmoid_independent" else pred_logits.softmax(dim=-1)
    )
    foreground_ids = torch.tensor(class_schema.foreground_class_ids, device=pred_logits.device)
    scores, foreground_positions = probabilities.index_select(-1, foreground_ids).max(dim=-1)
    return scores, foreground_ids[foreground_positions]


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
    class_schema: ClassSchema,
    *,
    max_active_tracks: int,
    frame_index: int,
) -> LifecycleTransition:
    """Apply deterministic tentative birth, tracking, suspension, and recycling.

    Args:
        table: Current host lifecycle metadata.
        state: Last committed trusted neural state.
        frame: Slot-aligned candidate output for the current frame.
        config: Inference lifecycle thresholds.
        class_schema: Authoritative foreground and background logit roles.
        max_active_tracks: Capacity reserved for active and suspended tracks.
        frame_index: Monotonically increasing source-frame index.

    Returns:
        A new immutable table, committed state, and ordered diagnostic events.
    """
    _validate_transition_inputs(table, state, frame, max_active_tracks, frame_index)
    scores, classes = foreground_scores(frame.pred_logits[0], class_schema)
    slots = list(table.slots)
    features = state.query_features.clone()
    boxes = state.reference_boxes.clone()
    active_mask = state.active_mask.clone()
    events: list[LifecycleEvent] = []
    next_track_id = table.next_track_id

    for index, old_slot in enumerate(table.slots):
        if old_slot.status in {"inactive", "tentative"}:
            continue
        score = float(scores[index].item())
        accept_continuation = score >= config.continuation_threshold
        if accept_continuation and config.identity_memory_enabled:
            other_memories = [
                other_slot.identity_memory
                for other_index, other_slot in enumerate(table.slots)
                if other_index != index
                and other_slot.status in {"active", "suspended"}
                and other_slot.class_id == old_slot.class_id
            ]
            margin = _identity_margin(
                frame.candidate_state.query_features[0, index], old_slot.identity_memory, other_memories
            )
            if margin is not None and margin <= 0:
                accept_continuation = False
        if accept_continuation:
            kind: EventKind | None = "recovered" if old_slot.status == "suspended" else None
            new_last_reliable_box, new_velocity = (
                _motion_observation(old_slot, frame.candidate_state.reference_boxes[0, index], frame_index)
                if config.motion_reference_prediction_enabled
                else (None, None)
            )
            new_identity_memory = (
                _identity_memory_update(
                    old_slot.identity_memory,
                    frame.candidate_state.query_features[0, index],
                    score,
                    config.identity_memory_length,
                )
                if config.identity_memory_enabled and score >= config.identity_memory_reliable_threshold
                else old_slot.identity_memory
            )
            slots[index] = replace(
                old_slot,
                status="active",
                age=old_slot.age + 1,
                hits=old_slot.hits + 1,
                missed_frames=0,
                last_reliable_frame=frame_index,
                confidence=score,
                class_id=int(classes[index].item()),
                tentative_started_frame=None,
                last_reliable_box=new_last_reliable_box,
                velocity=new_velocity,
                identity_memory=new_identity_memory,
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
            if (
                config.motion_reference_prediction_enabled
                and old_slot.last_reliable_box is not None
                and old_slot.velocity is not None
            ):
                boxes[:, index] = _predict_motion_box(
                    old_slot.last_reliable_box,
                    old_slot.velocity,
                    missed_frames,
                    device=boxes.device,
                    dtype=boxes.dtype,
                )
            if old_slot.status == "active":
                events.append(LifecycleEvent("suspended", index, old_slot.track_id, frame_index))

    tracked_count = sum(slot.status in {"active", "suspended"} for slot in slots)
    previous_frame_index = table.last_frame_index
    for index, old_slot in enumerate(table.slots):
        if old_slot.status != "tentative":
            continue
        assert old_slot.tentative_started_frame is not None
        score = float(scores[index].item())
        window_expired = frame_index - old_slot.tentative_started_frame >= config.tentative_confirmation_window_frames
        elapsed_frames = frame_index - (previous_frame_index if previous_frame_index is not None else frame_index - 1)
        missed_since_update = elapsed_frames if score < config.activation_threshold else elapsed_frames - 1
        misses = old_slot.missed_frames + missed_since_update
        if window_expired or misses >= config.tentative_max_misses:
            slots[index] = TrackSlot()
            features[:, index] = 0
            boxes[:, index] = 0
            active_mask[:, index] = False
            events.append(LifecycleEvent("tentative_cancelled", index, None, frame_index))
            continue
        if score < config.activation_threshold:
            slots[index] = replace(old_slot, age=old_slot.age + elapsed_frames, missed_frames=misses, confidence=score)
            continue

        hits = old_slot.hits + 1
        features[:, index] = frame.candidate_state.query_features[:, index]
        boxes[:, index] = frame.candidate_state.reference_boxes[:, index]
        if hits >= config.tentative_confirmation_hits and tracked_count < max_active_tracks:
            slots[index] = replace(
                old_slot,
                track_id=next_track_id,
                status="active",
                age=old_slot.age + elapsed_frames,
                hits=hits,
                missed_frames=0,
                last_reliable_frame=frame_index,
                tentative_started_frame=None,
                confidence=score,
                class_id=int(classes[index].item()),
                last_reliable_box=_initial_motion_state(config, boxes[0, index]),
                velocity=None,
            )
            events.append(LifecycleEvent("confirmed", index, next_track_id, frame_index))
            tracked_count += 1
            next_track_id += 1
        else:
            slots[index] = replace(
                old_slot,
                age=old_slot.age + elapsed_frames,
                hits=hits,
                missed_frames=misses,
                last_reliable_frame=frame_index,
                confidence=score,
                class_id=int(classes[index].item()),
            )
            if hits >= config.tentative_confirmation_hits:
                events.append(_suppression(index, frame_index, "durable_capacity", score, int(classes[index].item())))

    occupied = [
        (index, slot.class_id, slot.status, boxes[0, index])
        for index, slot in enumerate(slots)
        if slot.status != "inactive"
    ]
    tentative_count = sum(slot.status == "tentative" for slot in slots)
    immediate_birth = config.tentative_confirmation_hits == 1
    candidates = sorted(
        (
            (-float(scores[index].item()), index)
            for index, old_slot in enumerate(table.slots)
            if old_slot.status == "inactive" and float(scores[index].item()) >= config.activation_threshold
        )
    )
    for rank, (negated_score, index) in enumerate(candidates):
        confidence = -negated_score
        class_id = int(classes[index].item())
        if rank >= config.max_discovery_candidates_per_frame:
            events.append(_suppression(index, frame_index, "discovery_candidate_limit", confidence, class_id))
            continue
        candidate_box = frame.candidate_state.reference_boxes[0, index]

        if config.reassociation_enabled:
            suspended_occupied = [entry for entry in occupied if entry[2] == "suspended"]
            reassociation = _duplicate_conflict(candidate_box, class_id, suspended_occupied)
            if reassociation is not None and reassociation[0] >= config.reassociation_iou_threshold:
                overlap, _, suspended_index = reassociation
                suspended_slot = slots[suspended_index]
                assert suspended_slot.track_id is not None
                occupied = [entry for entry in occupied if entry[0] != suspended_index]
                slots[suspended_index] = TrackSlot()
                features[:, suspended_index] = 0
                boxes[:, suspended_index] = 0
                active_mask[:, suspended_index] = False
                slots[index] = TrackSlot(
                    track_id=suspended_slot.track_id,
                    status="active",
                    age=suspended_slot.age + 1,
                    hits=suspended_slot.hits + 1,
                    last_reliable_frame=frame_index,
                    confidence=confidence,
                    class_id=class_id,
                    last_reliable_box=_initial_motion_state(config, candidate_box),
                    velocity=None,
                )
                features[:, index] = frame.candidate_state.query_features[:, index]
                boxes[:, index] = candidate_box
                active_mask[:, index] = True
                occupied.append((index, class_id, "active", boxes[0, index]))
                events.append(
                    LifecycleEvent(
                        "reassociated",
                        index,
                        suspended_slot.track_id,
                        frame_index,
                        compared_status="suspended",
                        overlap=overlap,
                    )
                )
                continue

        conflict = _duplicate_conflict(candidate_box, class_id, occupied)
        if conflict is not None and conflict[0] > config.duplicate_iou_threshold:
            overlap, compared_status, _ = conflict
            events.append(
                _suppression(
                    index,
                    frame_index,
                    "duplicate_overlap",
                    confidence,
                    class_id,
                    compared_status=compared_status,
                    overlap=overlap,
                )
            )
            continue
        if immediate_birth:
            if tracked_count >= max_active_tracks:
                events.append(_suppression(index, frame_index, "durable_capacity", confidence, class_id))
                continue
        elif tentative_count >= config.max_tentative_tracks:
            events.append(_suppression(index, frame_index, "tentative_capacity", confidence, class_id))
            continue
        slots[index] = TrackSlot(
            track_id=next_track_id if immediate_birth else None,
            status="active" if immediate_birth else "tentative",
            age=1,
            hits=1,
            last_reliable_frame=frame_index,
            tentative_started_frame=None if immediate_birth else frame_index,
            confidence=confidence,
            class_id=class_id,
            last_reliable_box=_initial_motion_state(config, candidate_box) if immediate_birth else None,
        )
        features[:, index] = frame.candidate_state.query_features[:, index]
        boxes[:, index] = candidate_box
        active_mask[:, index] = True
        occupied.append((index, class_id, slots[index].status, boxes[0, index]))
        events.append(
            LifecycleEvent(
                "activated" if immediate_birth else "tentative_started",
                index,
                next_track_id if immediate_birth else None,
                frame_index,
            )
        )
        if immediate_birth:
            tracked_count += 1
            next_track_id += 1
        else:
            tentative_count += 1

    return LifecycleTransition(
        table=TrackSlotTable(tuple(slots), next_track_id, frame_index),
        state=TrackQueryState(features, boxes, active_mask),
        events=tuple(events),
    )


__all__ = [
    "LifecycleEvent",
    "LifecycleTransition",
    "SuppressionReason",
    "TrackSlot",
    "TrackSlotTable",
    "foreground_scores",
    "transition_lifecycle",
]
