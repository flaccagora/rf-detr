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

TrackStatus = Literal["inactive", "active", "suspended"]
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
CollisionDecision = Literal["remains_active", "suspended", "terminated"]


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
    last_reliable_box: tuple[float, float, float, float] | None = None
    velocity: tuple[float, float, float, float] | None = None
    identity_memory: tuple[tuple[float, tuple[float, ...]], ...] = ()
    collision_streak: int = 0

    def __post_init__(self) -> None:
        """Validate identity and status consistency."""
        if self.status == "inactive" and self.track_id is not None:
            raise ValueError("inactive slots must have no track_id")
        if self.status in {"active", "suspended"} and self.track_id is None:
            raise ValueError("active and suspended slots must have a track_id")
        if self.status != "inactive" and self.last_reliable_frame is None:
            raise ValueError("tracked slots must record their last reliable frame")
        if min(self.age, self.hits, self.missed_frames, self.collision_streak) < 0:
            raise ValueError("slot counters cannot be negative")
        if self.status == "inactive" and (self.last_reliable_box is not None or self.velocity is not None):
            raise ValueError("inactive slots must not retain motion-prediction state")
        if self.velocity is not None and self.last_reliable_box is None:
            raise ValueError("a slot cannot carry a velocity estimate without a last reliable box")
        if self.status == "inactive" and self.identity_memory:
            raise ValueError("inactive slots must not retain identity memory")
        if self.status == "inactive" and self.collision_streak != 0:
            raise ValueError("inactive slots must not retain a collision streak")


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
class TentativeRecord:
    """Host-only candidate awaiting independent rediscovery before any public identity exists.

    A tentative record never occupies a fixed decoder query slot and never
    contributes to ``TrackQueryState``: it carries no feature, and its box is
    tracked purely as host metadata (PRD Section 5.1). It has no public
    track ID until a later, independently generated discovery confirms it.
    """

    tentative_id: int
    class_id: int
    first_frame: int
    last_frame: int
    hits: int
    misses: int
    score_history: tuple[float, ...]
    last_observed_box: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        """Validate counter and frame-ordering invariants."""
        if self.tentative_id < 0:
            raise ValueError("tentative_id cannot be negative")
        if min(self.hits, self.misses) < 0:
            raise ValueError("tentative hit/miss counters cannot be negative")
        if self.last_frame < self.first_frame:
            raise ValueError("a tentative record cannot end before it started")
        if not self.score_history:
            raise ValueError("a tentative record must retain at least one score observation")


@dataclass(frozen=True)
class TentativeTrackPool:
    """Immutable host-side registry of candidates awaiting independent confirmation.

    Capacity, monotonic identity, and lifetime here are independent of
    ``TrackSlotTable``: tentative candidates never reserve a decoder query
    slot or durable active/suspended capacity (PRD Section 5.1).
    """

    records: tuple[TentativeRecord, ...] = ()
    next_tentative_id: int = 0

    def __post_init__(self) -> None:
        """Validate identity uniqueness and the monotonic counter."""
        if self.next_tentative_id < 0:
            raise ValueError("next_tentative_id cannot be negative")
        tentative_ids = [record.tentative_id for record in self.records]
        if len(tentative_ids) != len(set(tentative_ids)):
            raise ValueError("tentative IDs must be unique within a pool")
        if tentative_ids and self.next_tentative_id <= max(tentative_ids):
            raise ValueError("next_tentative_id must be greater than every assigned tentative ID")

    @classmethod
    def empty(cls) -> TentativeTrackPool:
        """Create a pool with no candidates and a counter starting at zero."""
        return cls()


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
        tentative_id: Private host-pool identity a tentative-scoped event
            belongs to. Required for ``tentative_started``, ``confirmed``,
            and ``tentative_cancelled`` -- the only kinds that reference a
            ``TentativeTrackPool`` record -- since that record carries no
            decoder query slot of its own (PRD Section 5.1) and may
            therefore be the only stable identity a diagnostic consumer can
            use to correlate a tentative's later events.
    """

    kind: EventKind
    slot: int | None
    track_id: int | None
    frame_index: int
    reason: SuppressionReason | None = None
    score: float | None = None
    class_id: int | None = None
    compared_status: TrackStatus | None = None
    overlap: float | None = None
    tentative_id: int | None = None

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
        requires_tentative_id = self.kind in {"tentative_started", "confirmed", "tentative_cancelled"}
        if requires_tentative_id and self.tentative_id is None:
            raise ValueError("tentative-scoped events must record the tentative_id they belong to")


@dataclass(frozen=True)
class CollisionArbitrationEvent:
    """Diagnostic record for one active-track duplicate-collision decision.

    Emitted whenever two same-class ``"active"`` slots overlap above
    ``collision_iou_threshold`` after this frame's continuation, confirmation,
    and birth stages have all committed (PRD Section 5.2). The loser's
    observation is suppressed from this frame's emission regardless of
    ``decision`` -- suppression never silently deletes lifecycle state.
    ``decision`` instead reports what happens to the loser's recurrent state:
    ``"remains_active"`` while ``collision_streak`` is still below
    ``collision_persistence_frames``, or the configured
    ``collision_loser_outcome`` once persistence is reached.
    """

    frame_index: int
    winner_track_id: int
    loser_track_id: int
    winner_slot: int
    loser_slot: int
    winner_score: float
    loser_score: float
    winner_age: int
    loser_age: int
    class_id: int
    iou: float
    collision_streak: int
    decision: CollisionDecision

    def __post_init__(self) -> None:
        """Reject a collision arbitrated against itself or with invalid evidence."""
        if self.winner_track_id == self.loser_track_id:
            raise ValueError("a collision cannot arbitrate a track against itself")
        if self.winner_slot == self.loser_slot:
            raise ValueError("a collision cannot arbitrate a slot against itself")
        if self.collision_streak < 1:
            raise ValueError("a collision arbitration must record a positive streak")
        if not 0.0 <= self.iou <= 1.0:
            raise ValueError("collision IoU must lie in [0, 1]")


@dataclass(frozen=True)
class LifecycleTransition:
    """New host metadata and committed neural state from one pure update."""

    table: TrackSlotTable
    state: TrackQueryState
    tentative_pool: TentativeTrackPool
    events: tuple[LifecycleEvent, ...]
    collision_events: tuple[CollisionArbitrationEvent, ...] = ()
    emitted_slots: tuple[int, ...] = ()


class LifecycleInvariantError(RuntimeError):
    """Raised when a completed lifecycle transition breaks a documented invariant (PRD US-005)."""


def validate_lifecycle_invariants(
    transition: LifecycleTransition,
    class_schema: ClassSchema,
) -> None:
    """Reject any transition output that could reintroduce duplicate identity or self-confirmation.

    Checked directly against one :func:`transition_lifecycle` call's output, encoding the
    corrected PRD Section 5.1/5.2 theory as an executable regression guard rather than only an
    implicit property of the update loop:

    - A discovery that only starts a tentative this frame never becomes active or claims a public
      track ID in that same transition -- the failure mode responsible for the pre-fix
      "106-of-107 self-confirmations" (US-005 acceptance criterion 1; ``TentativeRecord`` also has
      no ``track_id``/active-bit field at all, so a confirmed public identity can never be *read*
      off a pooled candidate by construction).
    - Every active/suspended slot owns exactly one public track ID, and every public track ID is
      owned by at most one slot (criterion 2).
    - No tentative ID and no discovery slot is consumed by more than one confirmation in this
      frame (criterion 3).
    - No public track ID appears twice among this frame's emitted rows (criterion 4).
    - No emitted row's class ID falls outside the declared foreground schema (criterion 5).

    Args:
        transition: The completed transition to validate.
        class_schema: Authoritative foreground and background logit roles.

    Raises:
        LifecycleInvariantError: The first violated invariant, naming the specific slot,
            track, or tentative identity that broke it.
    """
    for event in transition.events:
        if event.kind != "tentative_started":
            continue
        if event.track_id is not None:
            raise LifecycleInvariantError(
                f"tentative_started event for tentative {event.tentative_id} carries a public track_id"
            )
        if event.slot is not None and bool(transition.state.active_mask[0, event.slot]):
            raise LifecycleInvariantError(
                f"discovery at slot {event.slot} became active while only starting "
                f"tentative {event.tentative_id} this frame"
            )
        if event.slot is not None and transition.table.slots[event.slot].track_id is not None:
            raise LifecycleInvariantError(
                f"discovery at slot {event.slot} claimed a public track ID while only starting "
                f"tentative {event.tentative_id} this frame"
            )

    confirmed_events = [event for event in transition.events if event.kind == "confirmed"]
    confirmed_tentative_ids = [event.tentative_id for event in confirmed_events]
    if len(confirmed_tentative_ids) != len(set(confirmed_tentative_ids)):
        raise LifecycleInvariantError("more than one confirmation consumed the same tentative this frame")
    confirmed_slots = [event.slot for event in confirmed_events]
    if len(confirmed_slots) != len(set(confirmed_slots)):
        raise LifecycleInvariantError("more than one confirmation consumed the same discovery slot this frame")
    live_tentative_ids = {record.tentative_id for record in transition.tentative_pool.records}
    for tentative_id in confirmed_tentative_ids:
        if tentative_id in live_tentative_ids:
            raise LifecycleInvariantError(
                f"tentative {tentative_id} was confirmed but its record still lives in the pool"
            )

    track_ids = [slot.track_id for slot in transition.table.slots if slot.track_id is not None]
    if len(track_ids) != len(set(track_ids)):
        raise LifecycleInvariantError("more than one slot owns the same public track ID")
    for index, slot in enumerate(transition.table.slots):
        owns_identity = slot.track_id is not None
        is_durable = slot.status in {"active", "suspended"}
        if owns_identity != is_durable:
            raise LifecycleInvariantError(f"slot {index} has status {slot.status!r} but track_id={slot.track_id!r}")

    emitted_track_ids = [transition.table.slots[index].track_id for index in transition.emitted_slots]
    if any(track_id is None for track_id in emitted_track_ids):
        raise LifecycleInvariantError("an emitted slot has no public track ID")
    if len(emitted_track_ids) != len(set(emitted_track_ids)):
        raise LifecycleInvariantError("the same public track ID was emitted more than once this frame")

    for index in transition.emitted_slots:
        class_id = transition.table.slots[index].class_id
        if class_id not in class_schema.foreground_class_ids:
            raise LifecycleInvariantError(
                f"emitted slot {index} carries class_id={class_id!r}, outside the declared foreground schema"
            )


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
            velocity = tuple((new_box_tuple[dim] - old_slot.last_reliable_box[dim]) / elapsed for dim in range(4))
        else:
            velocity = old_slot.velocity
    return new_box_tuple, velocity


def _initial_motion_state(config: TrackingSessionConfig, box: Tensor) -> tuple[float, float, float, float] | None:
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
    predicted = torch.cat([predicted[:2].clamp(min=0.0, max=1.0), predicted[2:].clamp(min=1e-4, max=1.0)])
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
    tentative_id: int | None = None,
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
        tentative_id=tentative_id,
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
    tentative_pool: TentativeTrackPool | None = None,
    *,
    max_active_tracks: int,
    frame_index: int,
) -> LifecycleTransition:
    """Apply deterministic tentative birth, confirmation, tracking, suspension, and recycling.

    Tentative candidates are tracked purely as host-side metadata in
    ``tentative_pool``: starting one never occupies a decoder query slot,
    never sets a neural ``active_mask`` bit, and never feeds its feature or
    box into the next frame's persistent-query inputs (PRD Section 5.1).
    Confirmation requires a fresh, independently generated discovery from a
    later frame: pooled candidates are associated with this frame's eligible
    inactive-slot discoveries via class-aware, IoU-gated, globally one-to-one
    matching (maximizing total IoU, with deterministic tie-breaking by
    discovery score, tentative start frame, tentative ID, and query index).
    A tentative can never confirm itself from its own first-hit slot, since
    that slot's feature/box are never fed back into subsequent frames. A
    discovery that already duplicates a reliable active-track observation is
    also excluded from confirmation eligibility, so it can never start or
    confirm a tentative on top of an already-represented object (PRD Section
    5.2). Finally, every same-class pair of ``"active"`` slots that still
    overlap above ``collision_iou_threshold`` after this frame's continuation,
    confirmation, and birth stages is resolved by deterministic duplicate-
    collision arbitration: the loser's observation is excluded from
    ``LifecycleTransition.emitted_slots`` for this frame (though never
    silently deleted from lifecycle state) and recorded in
    ``LifecycleTransition.collision_events``.

    Args:
        table: Current host lifecycle metadata.
        state: Last committed trusted neural state.
        frame: Slot-aligned candidate output for the current frame.
        config: Inference lifecycle thresholds.
        class_schema: Authoritative foreground and background logit roles.
        tentative_pool: Current host-side tentative candidate registry.
            Defaults to an empty pool.
        max_active_tracks: Capacity reserved for active and suspended tracks.
        frame_index: Monotonically increasing source-frame index.

    Returns:
        A new immutable table, committed state, tentative pool, ordered
        lifecycle diagnostic events, ordered collision-arbitration
        diagnostics, and the decoder slot indices that should be emitted as
        visible observations this frame.
    """
    _validate_transition_inputs(table, state, frame, max_active_tracks, frame_index)
    scores, classes = foreground_scores(frame.pred_logits[0], class_schema)
    slots = list(table.slots)
    features = state.query_features.clone()
    boxes = state.reference_boxes.clone()
    active_mask = state.active_mask.clone()
    events: list[LifecycleEvent] = []
    next_track_id = table.next_track_id
    pool = tentative_pool if tentative_pool is not None else TentativeTrackPool.empty()
    tentative_records = list(pool.records)
    next_tentative_id = pool.next_tentative_id

    for index, old_slot in enumerate(table.slots):
        if old_slot.status == "inactive":
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
    occupied = [
        (index, slot.class_id, slot.status, boxes[0, index])
        for index, slot in enumerate(slots)
        if slot.status != "inactive"
    ]
    immediate_birth = config.tentative_confirmation_hits == 1
    ranked_candidates = sorted(
        (
            (-float(scores[index].item()), index)
            for index, old_slot in enumerate(table.slots)
            if old_slot.status == "inactive" and float(scores[index].item()) >= config.activation_threshold
        )
    )

    consumed_discoveries: set[int] = set()
    if not immediate_birth:
        # Independent-rediscovery confirmation obeys the same per-frame candidate cap as birth
        # (PRD Section 5.1's "all capacity limits"); ``ranked_candidates`` is score-sorted, so
        # every later rank also exceeds the cap once one does.
        active_occupied = [entry for entry in occupied if entry[2] == "active"]
        eligible_for_association: list[tuple[float, int, int]] = []
        for rank, (negated_score, index) in enumerate(ranked_candidates):
            if rank >= config.max_discovery_candidates_per_frame:
                break
            class_id = int(classes[index].item())
            candidate_box = frame.candidate_state.reference_boxes[0, index]
            # PRD Section 5.2: a discovery already representing a reliable active-track
            # observation cannot start or confirm a tentative. Excluding it here only removes
            # it from confirmation eligibility -- the birth loop below still evaluates it
            # directly and records its own duplicate-suppression diagnostic against the full
            # (active + suspended) occupied set, so no event is lost or duplicated.
            conflict = _duplicate_conflict(candidate_box, class_id, active_occupied)
            if conflict is not None and conflict[0] > config.duplicate_iou_threshold:
                continue
            eligible_for_association.append((-negated_score, index, class_id))
        eligible_by_index = {index: (confidence, class_id) for confidence, index, class_id in eligible_for_association}

        surviving_records: list[TentativeRecord] = []
        for record in tentative_records:
            if frame_index - record.first_frame >= config.tentative_confirmation_window_frames:
                events.append(
                    LifecycleEvent("tentative_cancelled", None, None, frame_index, tentative_id=record.tentative_id)
                )
            else:
                surviving_records.append(record)

        # Class-aware, IoU-gated edges between every surviving tentative and this frame's
        # eligible discoveries. Sorting maximizes total IoU first, then applies the documented
        # deterministic tie-break (discovery score, tentative start, tentative ID, query index),
        # so greedy one-to-one consumption is independent of pool/candidate iteration order.
        edges: list[tuple[float, float, int, int, int]] = []
        for record in surviving_records:
            last_box = torch.tensor(record.last_observed_box, device=boxes.device, dtype=boxes.dtype)
            for confidence, index, class_id in eligible_for_association:
                if class_id != record.class_id:
                    continue
                iou = _box_iou(frame.candidate_state.reference_boxes[0, index], last_box)
                if iou < config.tentative_association_iou_threshold:
                    continue
                edges.append((iou, confidence, record.first_frame, record.tentative_id, index))
        edges.sort(key=lambda edge: (-edge[0], -edge[1], edge[2], edge[3], edge[4]))

        matched_by_tentative_id: dict[int, int] = {}
        for _, _, _, tentative_id, index in edges:
            if tentative_id in matched_by_tentative_id or index in consumed_discoveries:
                continue
            matched_by_tentative_id[tentative_id] = index
            consumed_discoveries.add(index)

        tentative_records = []
        for record in surviving_records:
            match_index = matched_by_tentative_id.get(record.tentative_id)
            if match_index is None:
                misses = frame_index - record.last_frame
                if misses >= config.tentative_max_misses:
                    events.append(
                        LifecycleEvent("tentative_cancelled", None, None, frame_index, tentative_id=record.tentative_id)
                    )
                else:
                    tentative_records.append(replace(record, misses=misses))
                continue

            confidence, class_id = eligible_by_index[match_index]
            candidate_box = frame.candidate_state.reference_boxes[0, match_index]
            new_hits = record.hits + 1
            box_tuple = tuple(float(value) for value in candidate_box.tolist())
            updated_record = replace(
                record,
                hits=new_hits,
                misses=0,
                last_frame=frame_index,
                score_history=record.score_history + (confidence,),
                last_observed_box=box_tuple,
            )
            if new_hits < config.tentative_confirmation_hits:
                tentative_records.append(updated_record)
                continue
            if tracked_count >= max_active_tracks:
                tentative_records.append(updated_record)
                events.append(
                    _suppression(
                        match_index,
                        frame_index,
                        "durable_capacity",
                        confidence,
                        class_id,
                        tentative_id=record.tentative_id,
                    )
                )
                continue

            # Confirmation: the confirming discovery's feature, box, and decoder slot seed the
            # new recurrent track. Earlier tentative observations are never emitted or backfilled
            # (PRD Section 5.1) -- ``updated_record`` above is discarded, not committed.
            slots[match_index] = TrackSlot(
                track_id=next_track_id,
                status="active",
                age=1,
                hits=1,
                last_reliable_frame=frame_index,
                confidence=confidence,
                class_id=class_id,
                last_reliable_box=_initial_motion_state(config, candidate_box),
            )
            features[:, match_index] = frame.candidate_state.query_features[:, match_index]
            boxes[:, match_index] = candidate_box
            active_mask[:, match_index] = True
            occupied.append((match_index, class_id, "active", boxes[0, match_index]))
            events.append(
                LifecycleEvent("confirmed", match_index, next_track_id, frame_index, tentative_id=record.tentative_id)
            )
            tracked_count += 1
            next_track_id += 1

    for rank, (negated_score, index) in enumerate(ranked_candidates):
        confidence = -negated_score
        class_id = int(classes[index].item())
        if rank >= config.max_discovery_candidates_per_frame:
            events.append(_suppression(index, frame_index, "discovery_candidate_limit", confidence, class_id))
            continue
        if index in consumed_discoveries:
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
            slots[index] = TrackSlot(
                track_id=next_track_id,
                status="active",
                age=1,
                hits=1,
                last_reliable_frame=frame_index,
                confidence=confidence,
                class_id=class_id,
                last_reliable_box=_initial_motion_state(config, candidate_box),
            )
            features[:, index] = frame.candidate_state.query_features[:, index]
            boxes[:, index] = candidate_box
            active_mask[:, index] = True
            occupied.append((index, class_id, slots[index].status, boxes[0, index]))
            events.append(LifecycleEvent("activated", index, next_track_id, frame_index))
            tracked_count += 1
            next_track_id += 1
        else:
            if len(tentative_records) >= config.max_tentative_tracks:
                events.append(_suppression(index, frame_index, "tentative_capacity", confidence, class_id))
                continue
            box_tuple = tuple(float(value) for value in candidate_box.tolist())
            tentative_records.append(
                TentativeRecord(
                    tentative_id=next_tentative_id,
                    class_id=class_id,
                    first_frame=frame_index,
                    last_frame=frame_index,
                    hits=1,
                    misses=0,
                    score_history=(confidence,),
                    last_observed_box=box_tuple,
                )
            )
            events.append(LifecycleEvent("tentative_started", index, None, frame_index, tentative_id=next_tentative_id))
            next_tentative_id += 1

    # Active-track duplicate-collision arbitration (PRD Section 5.2): every same-class pair of
    # "active" slots overlapping above collision_iou_threshold is resolved deterministically, so
    # this frame emits at most one observation per represented object hypothesis. Suppression
    # never deletes lifecycle state outright -- the loser's own recurrent continuation already
    # committed above and is left untouched until its collision_streak reaches
    # collision_persistence_frames, at which point the configured collision_loser_outcome policy
    # applies. Priority is a total order over globally-unique track IDs (confidence, then age,
    # then track_id), so arbitration is independent of slot iteration order.
    active_indices = [index for index, slot in enumerate(slots) if slot.status == "active"]
    priority_order = sorted(
        active_indices,
        key=lambda index: (
            -(slots[index].confidence if slots[index].confidence is not None else 0.0),
            -slots[index].age,
            slots[index].track_id,
        ),
    )
    kept_indices: list[int] = []
    suppressed_this_frame: set[int] = set()
    collision_events: list[CollisionArbitrationEvent] = []
    for index in priority_order:
        candidate_slot = slots[index]
        best_conflict: tuple[int, float] | None = None
        for kept_index in kept_indices:
            kept_slot = slots[kept_index]
            if kept_slot.class_id != candidate_slot.class_id:
                continue
            iou = _box_iou(boxes[0, index], boxes[0, kept_index])
            if iou > config.collision_iou_threshold and (best_conflict is None or iou > best_conflict[1]):
                best_conflict = (kept_index, iou)
        if best_conflict is None:
            kept_indices.append(index)
            continue
        winner_index, iou = best_conflict
        winner_slot = slots[winner_index]
        suppressed_this_frame.add(index)
        assert candidate_slot.track_id is not None and winner_slot.track_id is not None
        assert candidate_slot.confidence is not None and winner_slot.confidence is not None
        assert candidate_slot.class_id is not None
        new_streak = candidate_slot.collision_streak + 1
        if new_streak >= config.collision_persistence_frames:
            decision: CollisionDecision = config.collision_loser_outcome
            if decision == "terminated":
                slots[index] = TrackSlot()
                features[:, index] = 0
                boxes[:, index] = 0
                active_mask[:, index] = False
            elif decision == "suspended":
                slots[index] = replace(candidate_slot, status="suspended", collision_streak=0)
            else:
                slots[index] = replace(candidate_slot, collision_streak=new_streak)
        else:
            decision = "remains_active"
            slots[index] = replace(candidate_slot, collision_streak=new_streak)
        collision_events.append(
            CollisionArbitrationEvent(
                frame_index=frame_index,
                winner_track_id=winner_slot.track_id,
                loser_track_id=candidate_slot.track_id,
                winner_slot=winner_index,
                loser_slot=index,
                winner_score=winner_slot.confidence,
                loser_score=candidate_slot.confidence,
                winner_age=winner_slot.age,
                loser_age=candidate_slot.age,
                class_id=candidate_slot.class_id,
                iou=iou,
                collision_streak=new_streak,
                decision=decision,
            )
        )
    for index in kept_indices:
        if slots[index].collision_streak != 0:
            slots[index] = replace(slots[index], collision_streak=0)

    emitted_slots = tuple(sorted(set(active_indices) - suppressed_this_frame))

    return LifecycleTransition(
        table=TrackSlotTable(tuple(slots), next_track_id, frame_index),
        state=TrackQueryState(features, boxes, active_mask),
        tentative_pool=TentativeTrackPool(tuple(tentative_records), next_tentative_id),
        events=tuple(events),
        collision_events=tuple(collision_events),
        emitted_slots=emitted_slots,
    )


__all__ = [
    "CollisionArbitrationEvent",
    "CollisionDecision",
    "LifecycleEvent",
    "LifecycleInvariantError",
    "LifecycleTransition",
    "SuppressionReason",
    "TentativeRecord",
    "TentativeTrackPool",
    "TrackSlot",
    "TrackSlotTable",
    "foreground_scores",
    "transition_lifecycle",
    "validate_lifecycle_invariants",
]
