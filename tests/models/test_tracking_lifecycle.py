# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import random

import pytest
import torch

from rfdetr.config import ClassSchema, ForegroundClass, TrackingSessionConfig
from rfdetr.models.tracking import TrackingFrameOutput, TrackQueryState
from rfdetr.tracking.lifecycle import (
    CollisionArbitrationEvent,
    LifecycleEvent,
    LifecycleInvariantError,
    LifecycleTransition,
    TentativeRecord,
    TentativeTrackPool,
    TrackSlot,
    TrackSlotTable,
    transition_lifecycle,
    validate_lifecycle_invariants,
)

_PERSON_SCHEMA = ClassSchema(
    foreground_classes=(ForegroundClass(class_id=0, name="person", external_category_id=0),),
    background_logit_index=1,
)


def _immediate_config(**kwargs: object) -> TrackingSessionConfig:
    """Retain immediate-birth setup for tests unrelated to tentative state."""
    return TrackingSessionConfig(tentative_confirmation_hits=1, **kwargs)


def _frame(
    scores: list[float],
    *,
    boxes: torch.Tensor | None = None,
    features: torch.Tensor | None = None,
    input_active_mask: torch.Tensor | None = None,
) -> TrackingFrameOutput:
    """Build a single-class frame result with predictable sigmoid confidences."""
    foreground_logits = torch.logit(torch.tensor(scores)).view(1, -1, 1)
    logits = torch.cat((foreground_logits, torch.full_like(foreground_logits, -10.0)), dim=-1)
    num_queries = len(scores)
    candidate_boxes = boxes if boxes is not None else torch.zeros(1, num_queries, 4)
    candidate_features = (
        features
        if features is not None
        else torch.arange(
            num_queries * 2,
            dtype=torch.float32,
        ).view(1, num_queries, 2)
    )
    return TrackingFrameOutput(
        pred_logits=logits,
        pred_boxes=candidate_boxes,
        candidate_state=TrackQueryState(
            query_features=candidate_features,
            reference_boxes=candidate_boxes,
            active_mask=torch.ones(1, num_queries, dtype=torch.bool),
        ),
        input_active_mask=(
            input_active_mask if input_active_mask is not None else torch.zeros(1, num_queries, dtype=torch.bool)
        ),
    )


def _slot(**overrides: object) -> TrackSlot:
    """Build a ``TrackSlot`` for directly crafting a ``LifecycleTransition`` under test."""
    fields: dict[str, object] = {
        "track_id": None,
        "status": "inactive",
        "age": 0,
        "hits": 0,
        "missed_frames": 0,
        "last_reliable_frame": None,
        "confidence": None,
        "class_id": None,
        "last_reliable_box": None,
        "velocity": None,
        "identity_memory": (),
        "collision_streak": 0,
    }
    fields.update(overrides)
    return TrackSlot(**fields)


def _crafted_state(active_mask: list[bool]) -> TrackQueryState:
    """Build a zero-valued ``TrackQueryState`` with the given per-slot active mask."""
    num_queries = len(active_mask)
    return TrackQueryState(
        query_features=torch.zeros(1, num_queries, 2),
        reference_boxes=torch.zeros(1, num_queries, 4),
        active_mask=torch.tensor([active_mask], dtype=torch.bool),
    )


class TestLifecycleTransitions:
    """Pure lifecycle behavior for a single tracking stream."""

    def test_tentative_birth_creates_a_pool_entry_without_touching_the_slot_table(self) -> None:
        """A discovery below the immediate-birth setting starts a host-only tentative record."""
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.1]),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )

        assert first.table.slots[0].status == "inactive"
        assert first.table.slot_track_ids == (None, None)
        assert first.table.next_track_id == 0
        assert first.state.active_mask.tolist() == [[False, False]]
        assert not first.state.query_features[0, 0].any()
        assert not first.state.reference_boxes[0, 0].any()
        assert [(event.kind, event.slot, event.track_id) for event in first.events] == [("tentative_started", 0, None)]

        [record] = first.tentative_pool.records
        assert record.tentative_id == 0
        assert record.class_id == 0
        assert record.first_frame == record.last_frame == 0
        assert record.hits == 1
        assert record.misses == 0
        assert record.score_history == pytest.approx((0.9,), abs=1e-5)
        assert first.tentative_pool.next_tentative_id == 1

    def test_tentative_pool_id_counter_is_independent_of_the_public_track_id_counter(self) -> None:
        """Tentative IDs and public track IDs are separate monotonic sequences."""
        config = _immediate_config()
        table = TrackSlotTable.empty(3)
        state = TrackQueryState.empty(1, 3, 2)
        pool = TentativeTrackPool.empty()

        # An immediate-birth config always allocates a public track ID directly and never
        # touches the tentative pool, so it must not advance the tentative counter.
        activated = transition_lifecycle(
            table,
            state,
            _frame([0.9, 0.1, 0.1]),
            config,
            _PERSON_SCHEMA,
            pool,
            max_active_tracks=2,
            frame_index=0,
        )
        assert activated.table.next_track_id == 1
        assert activated.tentative_pool.next_tentative_id == 0

        non_immediate = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.1, 0.9, 0.1]),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            TentativeTrackPool.empty(),
            max_active_tracks=2,
            frame_index=0,
        )
        assert non_immediate.table.next_track_id == 0
        assert non_immediate.tentative_pool.next_tentative_id == 1

    def test_starting_a_tentative_leaves_every_previously_inactive_slot_mask_bit_false(self) -> None:
        """Two simultaneous discoveries under a non-immediate config never flip active_mask."""
        result = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.9, 0.1]),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )

        assert result.state.active_mask.tolist() == [[False, False, False]]
        assert not result.state.query_features.any()
        assert not result.state.reference_boxes.any()
        assert len(result.tentative_pool.records) == 2

    def test_high_score_in_the_original_discovery_slot_cannot_confirm_a_tentative(self) -> None:
        """Repeated high scores at the same query index never allocate a public track ID.

        This reproduces the corrected behavior for the PRD's documented failure mode: the old
        implementation copied a first-hit discovery into recurrent state and marked its slot
        active, so the *same* query then supplied its own second-frame confirmation evidence
        (106-of-107 self-confirmations observed with the old semantics). With tentative
        candidates held outside recurrence, the slot never becomes active, so a persistently
        high score at that index can only ever look like a brand new inactive-slot discovery --
        never a continuation of the earlier candidate -- and nothing in this story confirms it.
        """
        table = TrackSlotTable.empty(2)
        state = TrackQueryState.empty(1, 2, 2)
        pool = TentativeTrackPool.empty()

        for frame_index in range(10):
            transition = transition_lifecycle(
                table,
                state,
                _frame([0.99, 0.01], input_active_mask=state.active_mask),
                TrackingSessionConfig(),
                _PERSON_SCHEMA,
                pool,
                max_active_tracks=1,
                frame_index=frame_index,
            )
            table, state, pool = transition.table, transition.state, transition.tentative_pool

            assert table.slot_track_ids == (None, None)
            assert not any(event.kind == "confirmed" for event in transition.events)
            assert not state.active_mask[0, 0]

        assert table.next_track_id == 0

    def test_existing_active_track_lifecycle_is_unaffected_by_an_empty_tentative_pool(self) -> None:
        """Continuation, suspension, and termination are identical whether or not a pool is passed."""
        config = _immediate_config(max_missed_frames=0)
        birth = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.1]),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )

        with_default_pool = transition_lifecycle(
            birth.table,
            birth.state,
            _frame([0.1, 0.1], input_active_mask=birth.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=1,
        )
        with_explicit_empty_pool = transition_lifecycle(
            birth.table,
            birth.state,
            _frame([0.1, 0.1], input_active_mask=birth.state.active_mask),
            config,
            _PERSON_SCHEMA,
            TentativeTrackPool.empty(),
            max_active_tracks=1,
            frame_index=1,
        )

        assert with_default_pool.table == with_explicit_empty_pool.table
        assert torch.equal(with_default_pool.state.active_mask, with_explicit_empty_pool.state.active_mask)
        assert [event.kind for event in with_default_pool.events] == ["terminated"]
        assert [event.kind for event in with_explicit_empty_pool.events] == ["terminated"]

    def test_reliable_discovery_activates_with_monotonic_ids_and_commits_state(self) -> None:
        """Eligible discoveries activate up to capacity and carry candidate neural state."""
        table = TrackSlotTable.empty(num_queries=3)
        prior_state = TrackQueryState.empty(batch_size=1, num_queries=3, hidden_dim=2)

        result = transition_lifecycle(
            table,
            prior_state,
            _frame([0.9, 0.8, 0.1]),
            _immediate_config(activation_threshold=0.5),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )

        assert result.table.slot_track_ids == (0, 1, None)
        assert result.table.next_track_id == 2
        assert torch.equal(result.state.active_mask, torch.tensor([[True, True, False]]))
        expected_features = _frame([0.9, 0.8, 0.1]).candidate_state.query_features
        assert torch.equal(result.state.query_features[:, :2], expected_features[:, :2])
        assert [(event.kind, event.slot, event.track_id) for event in result.events] == [
            ("activated", 0, 0),
            ("activated", 1, 1),
        ]

    def test_miss_preserves_trusted_state_then_recovery_keeps_identity(self) -> None:
        """Suspension does not commit weak candidates, and recovery resumes the same track."""
        initial = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.1]),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=10,
        )
        trusted_features = initial.state.query_features.clone()
        trusted_boxes = initial.state.reference_boxes.clone()
        missed = transition_lifecycle(
            initial.table,
            initial.state,
            _frame(
                [0.1, 0.1],
                features=torch.full((1, 2, 2), 99.0),
                input_active_mask=initial.state.active_mask,
            ),
            _immediate_config(max_missed_frames=3),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=12,
        )

        assert missed.table.slots[0].status == "suspended"
        assert missed.table.slots[0].missed_frames == 2
        assert torch.equal(missed.state.query_features, trusted_features)
        assert torch.equal(missed.state.reference_boxes, trusted_boxes)
        assert [event.kind for event in missed.events] == ["suspended"]

        recovered = transition_lifecycle(
            missed.table,
            missed.state,
            _frame(
                [0.8, 0.1],
                features=torch.full((1, 2, 2), 7.0),
                input_active_mask=missed.state.active_mask,
            ),
            _immediate_config(max_missed_frames=3),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=13,
        )

        assert recovered.table.slots[0].status == "active"
        assert recovered.table.slot_track_ids == (0, None)
        assert torch.equal(recovered.state.query_features[:, 0], torch.full((1, 2), 7.0))
        assert [event.kind for event in recovered.events] == ["recovered"]

    def test_expired_track_clears_state_and_recycled_slot_gets_new_id_next_frame(self) -> None:
        """Termination clears both boundaries and recycling never reuses external identity."""
        activated = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.1]),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        terminated = transition_lifecycle(
            activated.table,
            activated.state,
            _frame([0.9, 0.1], input_active_mask=activated.state.active_mask),
            _immediate_config(continuation_threshold=0.95, max_missed_frames=0),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=1,
        )

        assert terminated.table.slot_track_ids == (None, None)
        assert not terminated.state.active_mask.any()
        assert not terminated.state.query_features.any()
        assert [event.kind for event in terminated.events] == ["terminated"]

        recycled = transition_lifecycle(
            terminated.table,
            terminated.state,
            _frame([0.9, 0.1]),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=2,
        )

        assert recycled.table.slot_track_ids == (1, None)
        assert recycled.table.next_track_id == 2

    def test_same_class_overlapping_discovery_is_suppressed_with_event(self) -> None:
        """Discovery cannot duplicate a tracked slot above the configured IoU."""
        boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1]]])
        result = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.8, 0.1], boxes=boxes),
            _immediate_config(duplicate_iou_threshold=0.5),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )

        assert result.table.slot_track_ids == (0, None, None)
        assert [event.kind for event in result.events] == ["activated", "duplicate_suppressed"]

    def test_host_and_neural_role_mismatch_is_rejected(self) -> None:
        """Lifecycle policy never implicitly repairs divergent host and neural state."""
        table = TrackSlotTable.empty(2)
        state = TrackQueryState(
            query_features=torch.zeros(1, 2, 2),
            reference_boxes=torch.zeros(1, 2, 4),
            active_mask=torch.tensor([[True, False]]),
        )

        with pytest.raises(ValueError, match="must agree"):
            transition_lifecycle(
                table,
                state,
                _frame([0.9, 0.1]),
                _immediate_config(),
                _PERSON_SCHEMA,
                max_active_tracks=1,
                frame_index=0,
            )

    def test_background_never_activates_or_continues_a_track(self) -> None:
        """A dominant no-object logit cannot create or preserve public identity."""
        background = torch.tensor([[[-5.0, 10.0], [-5.0, 10.0]]])
        frame = _frame([0.1, 0.1])
        frame = TrackingFrameOutput(
            pred_logits=background,
            pred_boxes=frame.pred_boxes,
            candidate_state=frame.candidate_state,
            input_active_mask=frame.input_active_mask,
        )
        inactive = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            frame,
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )

        assert inactive.table.slot_track_ids == (None, None)

        active = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.1]),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        continued_frame = TrackingFrameOutput(
            pred_logits=background,
            pred_boxes=frame.pred_boxes,
            candidate_state=frame.candidate_state,
            input_active_mask=active.state.active_mask,
        )
        suspended = transition_lifecycle(
            active.table,
            active.state,
            continued_frame,
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=1,
        )

        assert suspended.table.slots[0].status == "suspended"

    def test_multiclass_scoring_selects_only_declared_foreground_logits(self) -> None:
        """Foreground class identity controls duplicate checks despite a shared background winner."""
        schema = ClassSchema(
            foreground_classes=(
                ForegroundClass(class_id=0, name="cat", external_category_id=7),
                ForegroundClass(class_id=1, name="dog", external_category_id=12),
            ),
            background_logit_index=2,
        )
        boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1]]])
        base = _frame([0.1, 0.1, 0.1], boxes=boxes)
        frame = TrackingFrameOutput(
            pred_logits=torch.tensor([[[3.0, 0.0, 10.0], [0.0, 3.0, 10.0], [-5.0, -4.0, 10.0]]]),
            pred_boxes=base.pred_boxes,
            candidate_state=base.candidate_state,
            input_active_mask=base.input_active_mask,
        )

        result = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            frame,
            _immediate_config(activation_threshold=0.9, duplicate_iou_threshold=0.5),
            schema,
            max_active_tracks=2,
            frame_index=0,
        )

        assert [slot.class_id for slot in result.table.slots[:2]] == [0, 1]
        assert result.table.slot_track_ids == (0, 1, None)


class TestDiscoveryBounds:
    """Bounded discovery capacity and status-aware uniqueness."""

    def test_discovery_candidates_are_ranked_by_foreground_score(self) -> None:
        """Only the strongest candidates are considered when the per-frame cap binds."""
        boxes = torch.tensor(
            [[[0.1, 0.1, 0.05, 0.05], [0.3, 0.3, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], [0.7, 0.7, 0.05, 0.05]]]
        )
        result = transition_lifecycle(
            TrackSlotTable.empty(4),
            TrackQueryState.empty(1, 4, 2),
            _frame([0.6, 0.95, 0.7, 0.99], boxes=boxes),
            _immediate_config(max_discovery_candidates_per_frame=2),
            _PERSON_SCHEMA,
            max_active_tracks=3,
            frame_index=0,
        )

        assert result.table.slot_track_ids == (None, 1, None, 0)
        assert [(event.kind, event.slot, event.reason) for event in result.events] == [
            ("activated", 3, None),
            ("activated", 1, None),
            ("capacity_suppressed", 2, "discovery_candidate_limit"),
            ("capacity_suppressed", 0, "discovery_candidate_limit"),
        ]

    def test_tentative_capacity_is_separate_from_the_per_frame_candidate_limit(self) -> None:
        """Discovery consideration and coexisting tentative records are bounded independently."""
        num_queries = 40
        boxes = torch.stack(
            [torch.tensor([index / num_queries + 0.005, 0.5, 0.01, 0.01]) for index in range(num_queries)]
        ).unsqueeze(0)
        result = transition_lifecycle(
            TrackSlotTable.empty(num_queries),
            TrackQueryState.empty(1, num_queries, 2),
            _frame([0.9 - index * 0.01 for index in range(num_queries)], boxes=boxes),
            TrackingSessionConfig(max_discovery_candidates_per_frame=12, max_tentative_tracks=10),
            _PERSON_SCHEMA,
            max_active_tracks=num_queries - 1,
            frame_index=0,
        )
        reasons = [event.reason for event in result.events]

        assert len(result.tentative_pool.records) == 10
        assert sum(slot.status != "inactive" for slot in result.table.slots) == 0
        assert reasons.count("tentative_capacity") == 2
        assert reasons.count("discovery_candidate_limit") == num_queries - 12

    def test_long_false_positive_trajectory_cannot_exceed_tentative_capacity(self) -> None:
        """Sustained spurious discoveries never accumulate unbounded tentative records.

        Each frame's boxes jump to a disjoint region from the previous frame's (no query
        index's box ever overlaps its own prior-frame box), so independent-rediscovery
        association (PRD US-003) never finds an eligible edge and nothing is ever confirmed --
        this isolates the tentative-pool capacity bound from confirmation behavior, which is
        covered separately in ``TestIndependentRediscoveryConfirmation``.
        """
        num_queries = 8
        num_frames = 6
        total_slots = num_frames * num_queries

        def _boxes(frame_index: int) -> torch.Tensor:
            """Give every (frame, index) pair in this run its own disjoint, non-overlapping slot."""
            return torch.stack(
                [
                    torch.tensor(
                        [
                            ((frame_index * num_queries + index) % total_slots) / total_slots + 0.5 / total_slots,
                            0.5,
                            0.3 / total_slots,
                            0.02,
                        ]
                    )
                    for index in range(num_queries)
                ]
            ).unsqueeze(0)

        config = TrackingSessionConfig(max_tentative_tracks=2)
        table = TrackSlotTable.empty(num_queries)
        state = TrackQueryState.empty(1, num_queries, 2)
        pool = TentativeTrackPool.empty()

        for frame_index in range(num_frames):
            transition = transition_lifecycle(
                table,
                state,
                _frame([0.9] * num_queries, boxes=_boxes(frame_index), input_active_mask=state.active_mask),
                config,
                _PERSON_SCHEMA,
                pool,
                max_active_tracks=3,
                frame_index=frame_index,
            )
            table, state, pool = transition.table, transition.state, transition.tentative_pool
            assert len(pool.records) <= 2
            assert sum(slot.status in {"active", "suspended"} for slot in table.slots) <= 3

        # Every (frame, index) box occupies its own disjoint slot, so no independent rediscovery
        # ever occurs across the whole run.
        assert table.next_track_id == 0

    def test_discovery_overlapping_an_active_track_records_complete_evidence(self) -> None:
        """A duplicate of an emitted track reports its reason, score, class, status, and overlap."""
        boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.05, 0.05]]])
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.1, 0.1], boxes=boxes),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.9, 0.95, 0.1], boxes=boxes, input_active_mask=first.state.active_mask),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=1,
        )
        suppression = next(event for event in second.events if event.kind == "duplicate_suppressed")

        assert second.table.slot_track_ids == (0, None, None)
        assert (suppression.slot, suppression.reason, suppression.compared_status) == (
            1,
            "duplicate_overlap",
            "active",
        )
        assert (suppression.class_id, suppression.overlap) == (0, pytest.approx(1.0))
        assert suppression.score == pytest.approx(0.95, abs=1e-5)

    def test_discovery_overlapping_a_suspended_track_is_suppressed(self) -> None:
        """A suspended slot keeps reserving its stale region against new discoveries."""
        boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.05, 0.05]]])
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.1, 0.1], boxes=boxes),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.95, 0.1], boxes=boxes, input_active_mask=first.state.active_mask),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=1,
        )
        suppression = next(event for event in second.events if event.kind == "duplicate_suppressed")

        assert second.table.slots[0].status == "suspended"
        assert second.table.slot_track_ids == (0, None, None)
        assert (suppression.slot, suppression.compared_status, suppression.overlap) == (
            1,
            "suspended",
            pytest.approx(1.0),
        )

    def test_discovery_over_a_pending_tentatives_box_confirms_it_from_a_different_query_index(self) -> None:
        """A later frame's fresh discovery over a pending tentative's box confirms it (PRD US-003).

        Before independent-rediscovery confirmation existed (PRD US-002), this exact setup was
        documented as starting an unrelated second tentative, since a pending candidate reserved
        no decoder slot or ``occupied`` region to suppress a duplicate against. Now that
        confirmation is implemented, a later frame's fresh, different-slot discovery landing on
        the tentative's last observed box is precisely the independent evidence PRD Section 5.1
        requires, so it confirms the pending tentative instead of starting a second one.
        """
        boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.05, 0.05]]])
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.1, 0.1], boxes=boxes),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.95, 0.1], boxes=boxes, input_active_mask=first.state.active_mask),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=2,
            frame_index=1,
        )

        assert len(first.tentative_pool.records) == 1
        assert [event.kind for event in second.events] == ["confirmed"]
        assert second.tentative_pool.records == ()
        assert second.table.slot_track_ids == (None, 0, None)
        assert second.table.slots[1].status == "active"

    def test_non_suppression_events_cannot_claim_a_suppression_reason(self) -> None:
        """Diagnostic roles stay unambiguous for downstream failure slices."""
        with pytest.raises(ValueError, match="must record a reason"):
            LifecycleEvent("activated", slot=0, track_id=1, frame_index=0, reason="durable_capacity")


class TestReassociation:
    """Discovery-to-suspended reassociation experiment (PRD US-024)."""

    def _suspended_setup(self, **config_kwargs: object) -> tuple:
        """Produce one active-then-suspended slot 0 alongside a same-region discovery slot 1."""
        boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4]]])
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.1], boxes=boxes),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        suspended = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.1], boxes=boxes, input_active_mask=first.state.active_mask),
            _immediate_config(**config_kwargs),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=1,
        )
        assert suspended.table.slots[0].status == "suspended"
        return suspended, boxes

    def test_discovery_reassociates_with_a_matching_suspended_track(self) -> None:
        """A discovery overlapping a suspended track's stale box revives its identity elsewhere."""
        suspended, boxes = self._suspended_setup()
        reassociated = transition_lifecycle(
            suspended.table,
            suspended.state,
            _frame(
                [0.1, 0.95],
                boxes=boxes,
                features=torch.full((1, 2, 2), 7.0),
                input_active_mask=suspended.state.active_mask,
            ),
            _immediate_config(reassociation_enabled=True, reassociation_iou_threshold=0.5),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=2,
        )

        assert reassociated.table.slot_track_ids == (None, 0)
        assert reassociated.table.slots[0].status == "inactive"
        assert reassociated.table.slots[1].status == "active"
        assert reassociated.table.next_track_id == 1
        assert not reassociated.state.active_mask[0, 0]
        assert torch.equal(reassociated.state.query_features[:, 1], torch.full((1, 2), 7.0))
        events = [(event.kind, event.slot, event.track_id) for event in reassociated.events]
        assert events == [("reassociated", 1, 0)]
        [reassociation_event] = reassociated.events
        assert reassociation_event.compared_status == "suspended"
        assert reassociation_event.overlap == pytest.approx(1.0)

    def test_reassociation_disabled_by_default_keeps_duplicate_suppression(self) -> None:
        """Without opting in, a suspended slot's region is still only protected by suppression."""
        suspended, boxes = self._suspended_setup()
        result = transition_lifecycle(
            suspended.table,
            suspended.state,
            _frame([0.1, 0.95], boxes=boxes, input_active_mask=suspended.state.active_mask),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=2,
        )

        assert [event.kind for event in result.events] == ["duplicate_suppressed"]
        assert result.table.slot_track_ids == (0, None)

    def test_reassociation_ignores_active_and_tentative_tracks(self) -> None:
        """Reassociation compares discovery evidence against suspended tracks only."""
        boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.05, 0.05]]])
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.1, 0.1], boxes=boxes),
            _immediate_config(reassociation_enabled=True, reassociation_iou_threshold=0.5),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.9, 0.95, 0.1], boxes=boxes, input_active_mask=first.state.active_mask),
            _immediate_config(reassociation_enabled=True, reassociation_iou_threshold=0.5),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=1,
        )

        assert first.table.slots[0].status == "active"
        assert [event.kind for event in second.events] == ["duplicate_suppressed"]
        assert second.table.slot_track_ids == (0, None, None)

    def test_reassociation_requires_iou_at_or_above_the_configured_threshold(self) -> None:
        """A weak overlap with a suspended track falls through to ordinary discovery handling."""
        boxes = torch.tensor([[[0.2, 0.2, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2], [0.9, 0.9, 0.05, 0.05]]])
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.1, 0.1], boxes=boxes),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        suspended = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.1, 0.1], boxes=boxes, input_active_mask=first.state.active_mask),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=1,
        )
        assert suspended.table.slots[0].status == "suspended"

        result = transition_lifecycle(
            suspended.table,
            suspended.state,
            _frame([0.1, 0.95, 0.1], boxes=boxes, input_active_mask=suspended.state.active_mask),
            _immediate_config(reassociation_enabled=True, reassociation_iou_threshold=0.9, duplicate_iou_threshold=0.5),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=2,
        )

        assert [event.kind for event in result.events] == ["activated"]
        assert result.table.slot_track_ids == (0, 1, None)
        assert result.table.slots[0].status == "suspended"

    def test_reassociation_does_not_consume_a_new_track_id_or_durable_capacity(self) -> None:
        """A revived identity leaves the monotonic ID counter and tracked-slot count unchanged."""
        suspended, boxes = self._suspended_setup()

        reassociated = transition_lifecycle(
            suspended.table,
            suspended.state,
            _frame([0.1, 0.95], boxes=boxes, input_active_mask=suspended.state.active_mask),
            _immediate_config(reassociation_enabled=True, reassociation_iou_threshold=0.5),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=2,
        )

        assert reassociated.table.next_track_id == suspended.table.next_track_id
        assert sum(slot.status in {"active", "suspended"} for slot in reassociated.table.slots) == 1


class TestIndependentRediscoveryConfirmation:
    """Two-hit confirmation via later-frame, independently generated discoveries (PRD US-003)."""

    def _config(self, **kwargs: object) -> TrackingSessionConfig:
        """Retain a common non-immediate baseline so each test only overrides what it exercises."""
        fields: dict[str, object] = {
            "tentative_confirmation_hits": 2,
            "tentative_confirmation_window_frames": 3,
            "tentative_max_misses": 2,
            "tentative_association_iou_threshold": 0.5,
        }
        fields.update(kwargs)
        return TrackingSessionConfig(**fields)

    def test_confirmation_accepts_a_discovery_from_a_different_query_index_and_seeds_state_from_it(self) -> None:
        """The confirming discovery's own feature/box seed recurrence, not the first-hit slot's."""
        box = torch.tensor([0.3, 0.3, 0.1, 0.1])
        boxes0 = torch.zeros(1, 3, 4)
        boxes0[0, 0] = box
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.1, 0.1], boxes=boxes0),
            self._config(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        assert [event.kind for event in first.events] == ["tentative_started"]
        [record] = first.tentative_pool.records

        boxes1 = torch.zeros(1, 3, 4)
        boxes1[0, 2] = box
        confirming_features = torch.full((1, 3, 2), 5.0)
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame(
                [0.1, 0.1, 0.9],
                boxes=boxes1,
                features=confirming_features,
                input_active_mask=first.state.active_mask,
            ),
            self._config(),
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=2,
            frame_index=1,
        )

        assert [(event.kind, event.slot, event.track_id, event.tentative_id) for event in second.events] == [
            ("confirmed", 2, 0, record.tentative_id)
        ]
        assert second.table.slot_track_ids == (None, None, 0)
        assert torch.equal(second.state.query_features[:, 2], confirming_features[:, 2])
        assert torch.equal(second.state.reference_boxes[0, 2], box)
        assert second.tentative_pool.records == ()
        # No backfill: the original first-hit slot (index 0) was never touched.
        assert second.table.slots[0].status == "inactive"
        assert not second.state.reference_boxes[0, 0].any()
        assert not second.state.query_features[0, 0].any()

    def test_association_ignores_a_class_mismatched_discovery(self) -> None:
        """A same-box discovery of the wrong class cannot increment a tentative's hit count."""
        schema = ClassSchema(
            foreground_classes=(
                ForegroundClass(class_id=0, name="cat", external_category_id=7),
                ForegroundClass(class_id=1, name="dog", external_category_id=12),
            ),
            background_logit_index=2,
        )
        box = torch.tensor([0.3, 0.3, 0.1, 0.1])
        boxes0 = torch.zeros(1, 2, 4)
        boxes0[0, 0] = box
        frame0 = TrackingFrameOutput(
            pred_logits=torch.tensor([[[5.0, -5.0, -5.0], [-5.0, -5.0, 5.0]]]),
            pred_boxes=boxes0,
            candidate_state=TrackQueryState(
                query_features=torch.zeros(1, 2, 2),
                reference_boxes=boxes0,
                active_mask=torch.ones(1, 2, dtype=torch.bool),
            ),
            input_active_mask=torch.zeros(1, 2, dtype=torch.bool),
        )
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            frame0,
            self._config(),
            schema,
            max_active_tracks=1,
            frame_index=0,
        )
        [record] = first.tentative_pool.records
        assert record.class_id == 0

        boxes1 = torch.zeros(1, 2, 4)
        boxes1[0, 1] = box
        frame1 = TrackingFrameOutput(
            pred_logits=torch.tensor([[[-5.0, -5.0, 5.0], [-5.0, 5.0, -5.0]]]),
            pred_boxes=boxes1,
            candidate_state=TrackQueryState(
                query_features=torch.zeros(1, 2, 2),
                reference_boxes=boxes1,
                active_mask=torch.ones(1, 2, dtype=torch.bool),
            ),
            input_active_mask=first.state.active_mask,
        )
        second = transition_lifecycle(
            first.table,
            first.state,
            frame1,
            self._config(),
            schema,
            first.tentative_pool,
            max_active_tracks=1,
            frame_index=1,
        )

        # The wrong-class discovery cannot confirm the class-0 tentative; it only starts its
        # own, unrelated class-1 tentative via the ordinary birth path.
        assert [event.kind for event in second.events] == ["tentative_started"]
        records_by_class = {r.class_id: r for r in second.tentative_pool.records}
        assert records_by_class[0].hits == 1
        assert records_by_class[0].misses == 1
        assert records_by_class[1].hits == 1

    def test_association_ignores_a_below_gate_discovery(self) -> None:
        """A same-class discovery whose IoU falls below the association gate cannot confirm."""
        boxes0 = torch.zeros(1, 2, 4)
        boxes0[0, 0] = torch.tensor([0.2, 0.2, 0.1, 0.1])
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.1], boxes=boxes0),
            self._config(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        [record] = first.tentative_pool.records

        boxes1 = torch.zeros(1, 2, 4)
        boxes1[0, 1] = torch.tensor([0.8, 0.8, 0.1, 0.1])
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.9], boxes=boxes1, input_active_mask=first.state.active_mask),
            self._config(),
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=1,
            frame_index=1,
        )

        assert [event.kind for event in second.events] == ["tentative_started"]
        original = next(r for r in second.tentative_pool.records if r.tentative_id == record.tentative_id)
        assert original.hits == 1
        assert original.misses == 1

    def test_competing_tentatives_for_one_discovery_resolve_by_the_documented_tie_break(self) -> None:
        """Deterministic tie-break resolves competing tentatives tied on total IoU."""
        box = torch.tensor([0.5, 0.5, 0.2, 0.2])
        boxes0 = torch.zeros(1, 3, 4)
        boxes0[0, 0] = box
        boxes0[0, 1] = box
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.9, 0.1], boxes=boxes0),
            self._config(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        tentative_a, tentative_b = first.tentative_pool.records
        assert tentative_a.tentative_id < tentative_b.tentative_id

        boxes1 = torch.zeros(1, 3, 4)
        boxes1[0, 2] = box
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.1, 0.9], boxes=boxes1, input_active_mask=first.state.active_mask),
            self._config(),
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=2,
            frame_index=1,
        )

        assert [(event.kind, event.tentative_id) for event in second.events] == [
            ("confirmed", tentative_a.tentative_id)
        ]
        assert second.table.slot_track_ids == (None, None, 0)
        [surviving] = second.tentative_pool.records
        assert surviving.tentative_id == tentative_b.tentative_id
        assert surviving.hits == 1
        assert surviving.misses == 1

    def test_matching_is_deterministic_regardless_of_pool_record_order(self) -> None:
        """Reordering the incoming tentative pool tuple must not change the tie-break winner."""
        box = torch.tensor([0.5, 0.5, 0.2, 0.2])
        boxes0 = torch.zeros(1, 3, 4)
        boxes0[0, 0] = box
        boxes0[0, 1] = box
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.9, 0.1], boxes=boxes0),
            self._config(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        lower_id, higher_id = first.tentative_pool.records
        assert lower_id.tentative_id < higher_id.tentative_id
        reordered_pool = TentativeTrackPool((higher_id, lower_id), first.tentative_pool.next_tentative_id)

        boxes1 = torch.zeros(1, 3, 4)
        boxes1[0, 2] = box
        forward = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.1, 0.9], boxes=boxes1, input_active_mask=first.state.active_mask),
            self._config(),
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=2,
            frame_index=1,
        )
        reordered = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.1, 0.9], boxes=boxes1, input_active_mask=first.state.active_mask),
            self._config(),
            _PERSON_SCHEMA,
            reordered_pool,
            max_active_tracks=2,
            frame_index=1,
        )

        assert [(event.kind, event.tentative_id) for event in forward.events] == [("confirmed", lower_id.tentative_id)]
        assert [(event.kind, event.tentative_id) for event in reordered.events] == [
            ("confirmed", lower_id.tentative_id)
        ]

    def test_two_nearby_tentatives_confirm_from_their_own_matching_discovery(self) -> None:
        """Distinct nearby tentatives each confirm from the discovery that actually overlaps them."""
        box_left = torch.tensor([0.2, 0.5, 0.15, 0.3])
        box_right = torch.tensor([0.4, 0.5, 0.15, 0.3])
        boxes0 = torch.zeros(1, 3, 4)
        boxes0[0, 0] = box_left
        boxes0[0, 1] = box_right
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.9, 0.1], boxes=boxes0),
            self._config(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        left, right = first.tentative_pool.records

        # The pair reappears at swapped query indices next frame -- association must key off
        # box overlap, not which decoder position originally observed each candidate.
        boxes1 = torch.zeros(1, 3, 4)
        boxes1[0, 0] = box_right
        boxes1[0, 1] = box_left
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.9, 0.9, 0.1], boxes=boxes1, input_active_mask=first.state.active_mask),
            self._config(),
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=2,
            frame_index=1,
        )

        confirmed = {event.tentative_id: event.slot for event in second.events if event.kind == "confirmed"}
        assert confirmed == {left.tentative_id: 1, right.tentative_id: 0}

    def test_new_tentative_starts_remain_bounded_by_capacity_after_an_association_step(self) -> None:
        """Pending-tentative capacity is still enforced when nothing is freed by association."""
        box = torch.tensor([0.5, 0.5, 0.2, 0.2])
        far_box = torch.tensor([0.05, 0.05, 0.05, 0.05])
        boxes0 = torch.zeros(1, 3, 4)
        boxes0[0, 0] = box
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.1, 0.1], boxes=boxes0),
            self._config(max_tentative_tracks=1),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        [record] = first.tentative_pool.records

        boxes1 = torch.zeros(1, 3, 4)
        boxes1[0, 1] = far_box
        boxes1[0, 2] = torch.tensor([0.9, 0.9, 0.05, 0.05])
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.9, 0.9], boxes=boxes1, input_active_mask=first.state.active_mask),
            self._config(max_tentative_tracks=1),
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=2,
            frame_index=1,
        )

        assert [event.kind for event in second.events if event.kind == "confirmed"] == []
        reasons = [event.reason for event in second.events]
        assert reasons.count("tentative_capacity") == 2
        [surviving] = second.tentative_pool.records
        assert surviving.tentative_id == record.tentative_id
        assert surviving.misses == 1

    def test_tentative_is_cancelled_exactly_when_the_confirmation_window_elapses(self) -> None:
        """A pending tentative survives through its last eligible frame and is cancelled after."""
        box = torch.tensor([0.5, 0.5, 0.2, 0.2])
        config = self._config(tentative_confirmation_window_frames=3, tentative_max_misses=10)
        boxes0 = torch.zeros(1, 2, 4)
        boxes0[0, 0] = box
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.1], boxes=boxes0),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        [record] = first.tentative_pool.records

        # Frame index 2 == first_frame(0) + window(3) - 1: still inside the inclusive window.
        still_pending = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.1], input_active_mask=first.state.active_mask),
            config,
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=1,
            frame_index=2,
        )
        assert [event.kind for event in still_pending.events] == []
        [surviving] = still_pending.tentative_pool.records
        assert surviving.tentative_id == record.tentative_id
        assert surviving.misses == 2

        # Frame index 3 == first_frame(0) + window(3): the window has fully elapsed.
        cancelled = transition_lifecycle(
            still_pending.table,
            still_pending.state,
            _frame([0.1, 0.1], input_active_mask=still_pending.state.active_mask),
            config,
            _PERSON_SCHEMA,
            still_pending.tentative_pool,
            max_active_tracks=1,
            frame_index=3,
        )
        assert [(event.kind, event.tentative_id) for event in cancelled.events] == [
            ("tentative_cancelled", record.tentative_id)
        ]
        assert cancelled.tentative_pool.records == ()

    def test_tentative_is_cancelled_exactly_when_misses_reach_the_limit_across_a_frame_gap(self) -> None:
        """Miss expiry is a plain elapsed-frame gap, so it is exact even across skipped frames."""
        box = torch.tensor([0.5, 0.5, 0.2, 0.2])
        config = self._config(tentative_confirmation_window_frames=100, tentative_max_misses=3)
        boxes0 = torch.zeros(1, 2, 4)
        boxes0[0, 0] = box
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.1], boxes=boxes0),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        [record] = first.tentative_pool.records

        # One frame short of the limit survives.
        survived = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.1], input_active_mask=first.state.active_mask),
            config,
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=1,
            frame_index=2,
        )
        assert [event.kind for event in survived.events] == []
        [surviving] = survived.tentative_pool.records
        assert surviving.misses == 2

        # A three-frame gap (frame_index jumps from 0 straight to 3) lands exactly at the miss
        # limit in one call, without needing three separate unmatched updates.
        cancelled = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.1], input_active_mask=first.state.active_mask),
            config,
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=1,
            frame_index=3,
        )
        assert [(event.kind, event.tentative_id) for event in cancelled.events] == [
            ("tentative_cancelled", record.tentative_id)
        ]

    def test_confirmation_blocked_by_durable_capacity_keeps_the_tentative_pending(self) -> None:
        """A capacity-blocked confirmation leaves the tentative alive with its incremented hit count."""
        box_a = torch.tensor([0.2, 0.5, 0.1, 0.1])
        box_b = torch.tensor([0.8, 0.5, 0.1, 0.1])
        boxes0 = torch.zeros(1, 4, 4)
        boxes0[0, 0] = box_a
        boxes0[0, 1] = box_b
        config = self._config()
        first = transition_lifecycle(
            TrackSlotTable.empty(4),
            TrackQueryState.empty(1, 4, 2),
            _frame([0.9, 0.9, 0.1, 0.1], boxes=boxes0),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        tentative_a, tentative_b = first.tentative_pool.records
        assert tentative_a.tentative_id < tentative_b.tentative_id

        boxes1 = torch.zeros(1, 4, 4)
        boxes1[0, 2] = box_a
        boxes1[0, 3] = box_b
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.1, 0.9, 0.9], boxes=boxes1, input_active_mask=first.state.active_mask),
            config,
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=1,
            frame_index=1,
        )

        confirmed_event = next(event for event in second.events if event.kind == "confirmed")
        assert confirmed_event.tentative_id == tentative_a.tentative_id
        blocked_event = next(event for event in second.events if event.reason == "durable_capacity")
        assert blocked_event.tentative_id == tentative_b.tentative_id
        [surviving] = second.tentative_pool.records
        assert surviving.tentative_id == tentative_b.tentative_id
        assert surviving.hits == 2
        assert surviving.misses == 0


class TestMotionReferencePrediction:
    """Elapsed-time/velocity-aware suspended reference-box extrapolation (PRD US-025).

    Every fixture uses two query slots (a tracked slot 0 plus a permanently-quiet discovery
    slot 1) so ``max_active_tracks=1`` still leaves the required discovery slot free
    (``_validate_transition_inputs`` rejects ``max_active_tracks >= num_queries``).
    """

    _QUIET_BOX = [0.9, 0.9, 0.05, 0.05]

    def _boxes(self, tracked_box: list[float]) -> torch.Tensor:
        return torch.tensor([[tracked_box, self._QUIET_BOX]])

    def _two_hit_velocity_setup(self, **config_kwargs: object) -> tuple:
        """Produce one slot with two active hits (box moving +0.1 in x per frame)."""
        config = _immediate_config(motion_reference_prediction_enabled=True, **config_kwargs)
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.05], boxes=self._boxes([0.1, 0.5, 0.2, 0.2])),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame(
                [0.9, 0.05],
                boxes=self._boxes([0.2, 0.5, 0.2, 0.2]),
                input_active_mask=first.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=1,
        )
        assert second.table.slots[0].status == "active"
        assert second.table.slots[0].last_reliable_box == pytest.approx((0.2, 0.5, 0.2, 0.2))
        assert second.table.slots[0].velocity == pytest.approx((0.1, 0.0, 0.0, 0.0))
        return second, config

    def test_disabled_by_default_never_records_motion_state_or_moves_the_suspended_box(self) -> None:
        """Without opting in, a suspended slot's reference box stays exactly where it was."""
        config = _immediate_config()
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.05], boxes=self._boxes([0.1, 0.5, 0.2, 0.2])),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame(
                [0.9, 0.05],
                boxes=self._boxes([0.2, 0.5, 0.2, 0.2]),
                input_active_mask=first.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=1,
        )
        assert second.table.slots[0].last_reliable_box is None
        assert second.table.slots[0].velocity is None

        suspended = transition_lifecycle(
            second.table,
            second.state,
            _frame([0.1, 0.05], boxes=self._boxes([0.2, 0.5, 0.2, 0.2]), input_active_mask=second.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=2,
        )
        assert suspended.table.slots[0].status == "suspended"
        assert torch.equal(suspended.state.reference_boxes[0, 0], torch.tensor([0.2, 0.5, 0.2, 0.2]))

    def test_enabled_extrapolates_the_suspended_slot_reference_box(self) -> None:
        """A suspended slot's box drifts forward by its constant per-frame velocity estimate."""
        established, config = self._two_hit_velocity_setup()

        suspended = transition_lifecycle(
            established.table,
            established.state,
            _frame(
                [0.1, 0.05],
                boxes=self._boxes([0.2, 0.5, 0.2, 0.2]),
                input_active_mask=established.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=2,
        )

        assert suspended.table.slots[0].status == "suspended"
        # last_reliable_box=(0.2, 0.5, ...), velocity=(0.1, 0, 0, 0), one elapsed frame.
        assert suspended.state.reference_boxes[0, 0].tolist() == pytest.approx([0.3, 0.5, 0.2, 0.2])
        # Motion prediction never touches recurrent query features (lifecycle stays unchanged).
        assert torch.equal(suspended.state.query_features[:, 0], established.state.query_features[:, 0])
        assert [event.kind for event in suspended.events] == ["suspended"]

    def test_prediction_is_anchored_to_the_last_reliable_box_not_compounded(self) -> None:
        """Repeated suspended-frame calls recompute from the last reliable box, avoiding drift."""
        established, config = self._two_hit_velocity_setup()

        direct = transition_lifecycle(
            established.table,
            established.state,
            _frame(
                [0.1, 0.05],
                boxes=self._boxes([0.2, 0.5, 0.2, 0.2]),
                input_active_mask=established.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=8,
        )

        stepped_once = transition_lifecycle(
            established.table,
            established.state,
            _frame(
                [0.1, 0.05],
                boxes=self._boxes([0.2, 0.5, 0.2, 0.2]),
                input_active_mask=established.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=4,
        )
        stepped_twice = transition_lifecycle(
            stepped_once.table,
            stepped_once.state,
            _frame(
                [0.1, 0.05],
                boxes=self._boxes([0.2, 0.5, 0.2, 0.2]),
                input_active_mask=stepped_once.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=8,
        )

        # Elapsed 7 frames since the last reliable observation (frame 1) either way: same predicted box.
        assert direct.state.reference_boxes[0, 0].tolist() == pytest.approx(
            stepped_twice.state.reference_boxes[0, 0].tolist()
        )
        assert direct.state.reference_boxes[0, 0].tolist() == pytest.approx([0.9, 0.5, 0.2, 0.2])

    def test_variable_frame_interval_expiry_matches_regardless_of_intermediate_steps(self) -> None:
        """Termination depends only on elapsed source-frame index, not the number of poll calls."""
        config = _immediate_config(motion_reference_prediction_enabled=True, max_missed_frames=10)
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.05], boxes=self._boxes([0.5, 0.5, 0.2, 0.2])),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )

        direct_jump = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.05], boxes=self._boxes([0.5, 0.5, 0.2, 0.2]), input_active_mask=first.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=11,
        )
        assert direct_jump.table.slots[0].status == "inactive"
        assert [event.kind for event in direct_jump.events] == ["terminated"]

        intermediate = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.05], boxes=self._boxes([0.5, 0.5, 0.2, 0.2]), input_active_mask=first.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=5,
        )
        assert intermediate.table.slots[0].status == "suspended"
        stepped = transition_lifecycle(
            intermediate.table,
            intermediate.state,
            _frame(
                [0.1, 0.05],
                boxes=self._boxes([0.5, 0.5, 0.2, 0.2]),
                input_active_mask=intermediate.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=11,
        )
        assert stepped.table.slots[0].status == "inactive"
        assert [event.kind for event in stepped.events] == ["terminated"]

    def test_recovery_resumes_velocity_tracking_and_lifecycle_status_is_unaffected_by_motion(self) -> None:
        """Enabling motion prediction changes only the fed-forward box, never the status decision."""
        established, config = self._two_hit_velocity_setup()
        disabled_config = _immediate_config()

        suspended_enabled = transition_lifecycle(
            established.table,
            established.state,
            _frame(
                [0.1, 0.05],
                boxes=self._boxes([0.2, 0.5, 0.2, 0.2]),
                input_active_mask=established.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=2,
        )
        suspended_disabled = transition_lifecycle(
            established.table,
            established.state,
            _frame(
                [0.1, 0.05],
                boxes=self._boxes([0.2, 0.5, 0.2, 0.2]),
                input_active_mask=established.state.active_mask,
            ),
            disabled_config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=2,
        )

        assert suspended_enabled.table.slots[0].status == suspended_disabled.table.slots[0].status
        assert suspended_enabled.table.slots[0].track_id == suspended_disabled.table.slots[0].track_id
        assert [e.kind for e in suspended_enabled.events] == [e.kind for e in suspended_disabled.events]
        assert not torch.equal(
            suspended_enabled.state.reference_boxes[0, 0], suspended_disabled.state.reference_boxes[0, 0]
        )

        recovered = transition_lifecycle(
            suspended_enabled.table,
            suspended_enabled.state,
            _frame(
                [0.9, 0.05],
                boxes=self._boxes([0.5, 0.5, 0.2, 0.2]),
                input_active_mask=suspended_enabled.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=3,
        )
        assert recovered.table.slots[0].status == "active"
        assert [event.kind for event in recovered.events] == ["recovered"]
        # Velocity is recomputed from the *last reliable* box (established at frame 1, elapsed=2
        # frames to frame 3, delta=0.3), not from the suspended frame's motion-predicted box.
        assert recovered.table.slots[0].last_reliable_box == pytest.approx((0.5, 0.5, 0.2, 0.2))
        assert recovered.table.slots[0].velocity == pytest.approx((0.15, 0.0, 0.0, 0.0))


class TestIdentityMemory:
    """Bounded, quality-weighted identity-memory continuation veto (PRD US-026).

    Three query slots are used throughout: two candidate tracked identities
    (0 and 1) plus a permanently-quiet discovery slot (2), so
    ``max_active_tracks=2`` still leaves the required discovery slot free.
    Feature vectors are two-dimensional and hand-picked so cosine similarity
    is exactly computable: ``(1, 0)`` and ``(0, 1)`` are orthogonal.
    """

    _QUIET_BOX = [0.9, 0.9, 0.05, 0.05]
    _SLOT0_BOX = [0.1, 0.5, 0.2, 0.2]
    _SLOT1_BOX = [0.6, 0.5, 0.2, 0.2]

    def _boxes(self) -> torch.Tensor:
        return torch.tensor([[self._SLOT0_BOX, self._SLOT1_BOX, self._QUIET_BOX]])

    def _features(self, slot0: tuple[float, float], slot1: tuple[float, float]) -> torch.Tensor:
        return torch.tensor([[list(slot0), list(slot1), [0.0, 0.0]]])

    def _two_identity_setup(self, **config_kwargs: object) -> tuple:
        """Birth both tracked slots, then let one reliable continuation seed their memory."""
        config = _immediate_config(
            identity_memory_enabled=True, identity_memory_reliable_threshold=0.0, **config_kwargs
        )
        birth = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.9, 0.05], boxes=self._boxes(), features=self._features((1.0, 0.0), (0.0, 1.0))),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        assert birth.table.slots[0].track_id == 0
        assert birth.table.slots[1].track_id == 1
        assert birth.table.slots[0].identity_memory == ()
        assert birth.table.slots[1].identity_memory == ()

        seeded = transition_lifecycle(
            birth.table,
            birth.state,
            _frame(
                [0.9, 0.9, 0.05],
                boxes=self._boxes(),
                features=self._features((1.0, 0.0), (0.0, 1.0)),
                input_active_mask=birth.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=1,
        )
        assert seeded.table.slots[0].status == "active"
        assert seeded.table.slots[1].status == "active"
        assert len(seeded.table.slots[0].identity_memory) == 1
        assert seeded.table.slots[0].identity_memory[0][0] == pytest.approx(0.9, abs=1e-3)
        assert seeded.table.slots[0].identity_memory[0][1] == pytest.approx((1.0, 0.0))
        assert seeded.table.slots[1].identity_memory[0][1] == pytest.approx((0.0, 1.0))
        return seeded, config

    def test_disabled_by_default_never_records_memory_or_vetoes_continuation(self) -> None:
        """Without opting in, no slot records identity memory and continuation is score-only."""
        config = _immediate_config()
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.9, 0.05], boxes=self._boxes(), features=self._features((1.0, 0.0), (0.0, 1.0))),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        # Slot 0's next candidate feature exactly matches slot 1's identity -- with memory
        # disabled this must have no effect on the continuation decision.
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame(
                [0.9, 0.9, 0.05],
                boxes=self._boxes(),
                features=self._features((0.0, 1.0), (0.0, 1.0)),
                input_active_mask=first.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=1,
        )
        assert second.table.slots[0].status == "active"
        assert second.table.slots[0].identity_memory == ()
        assert second.table.slots[1].identity_memory == ()

    def test_nonpositive_identity_margin_vetoes_continuation_and_suspends_the_slot(self) -> None:
        """A candidate feature that better matches another slot's memory does not continue this slot."""
        seeded, config = self._two_identity_setup()

        # Slot 0's incoming feature now resembles slot 1's memory (a simulated crossing
        # hijack); slot 1's own feature stays consistent with its own memory.
        hijacked = transition_lifecycle(
            seeded.table,
            seeded.state,
            _frame(
                [0.9, 0.9, 0.05],
                boxes=self._boxes(),
                features=self._features((0.0, 1.0), (0.0, 1.0)),
                input_active_mask=seeded.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=2,
        )

        assert hijacked.table.slots[0].status == "suspended"
        assert hijacked.table.slots[0].track_id == 0
        assert hijacked.table.slots[0].identity_memory == seeded.table.slots[0].identity_memory
        slot0_events = [event for event in hijacked.events if event.slot == 0]
        assert [event.kind for event in slot0_events] == ["suspended"]

        assert hijacked.table.slots[1].status == "active"
        assert hijacked.table.slots[1].track_id == 1
        assert len(hijacked.table.slots[1].identity_memory) == 2

    def test_positive_margin_accepts_continuation_normally(self) -> None:
        """A candidate feature that still matches its own slot's memory continues as usual."""
        seeded, config = self._two_identity_setup()

        matched = transition_lifecycle(
            seeded.table,
            seeded.state,
            _frame(
                [0.9, 0.9, 0.05],
                boxes=self._boxes(),
                features=self._features((1.0, 0.0), (0.0, 1.0)),
                input_active_mask=seeded.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=2,
        )

        assert matched.table.slots[0].status == "active"
        assert [event.kind for event in matched.events if event.slot == 0] == []
        assert len(matched.table.slots[0].identity_memory) == 2

    def test_no_ambiguity_without_another_same_class_memory_accepts_continuation(self) -> None:
        """A lone tracked slot is never vetoed -- there is no other memory to disagree with."""
        config = _immediate_config(identity_memory_enabled=True, identity_memory_reliable_threshold=0.0)
        boxes = torch.tensor([[self._SLOT0_BOX, self._QUIET_BOX]])
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.05], boxes=boxes, features=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame(
                [0.9, 0.05],
                boxes=boxes,
                features=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
                input_active_mask=first.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=1,
        )
        assert len(second.table.slots[0].identity_memory) == 1

        # A wildly different feature (opposite direction) still continues: no other same-class
        # slot carries memory, so there is nothing to be more similar to.
        third = transition_lifecycle(
            second.table,
            second.state,
            _frame(
                [0.9, 0.05],
                boxes=boxes,
                features=torch.tensor([[[-1.0, 0.0], [0.0, 0.0]]]),
                input_active_mask=second.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=2,
        )
        assert third.table.slots[0].status == "active"
        assert [event.kind for event in third.events] == []

    def test_memory_updates_only_on_reliable_continuations(self) -> None:
        """A continuation below the reliable-memory threshold does not grow the bank."""
        config = _immediate_config(identity_memory_enabled=True, identity_memory_reliable_threshold=0.8)
        boxes = torch.tensor([[self._SLOT0_BOX, self._QUIET_BOX]])
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.05], boxes=boxes, features=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        weak = transition_lifecycle(
            first.table,
            first.state,
            _frame(
                [0.6, 0.05],
                boxes=boxes,
                features=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
                input_active_mask=first.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=1,
        )
        assert weak.table.slots[0].status == "active"
        assert weak.table.slots[0].identity_memory == ()

        strong = transition_lifecycle(
            weak.table,
            weak.state,
            _frame(
                [0.9, 0.05],
                boxes=boxes,
                features=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
                input_active_mask=weak.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=2,
        )
        assert len(strong.table.slots[0].identity_memory) == 1

    def test_memory_bank_is_quality_weighted_not_recency_bounded(self) -> None:
        """A low-weight early observation is evicted in favor of later higher-weight ones."""
        config = _immediate_config(
            identity_memory_enabled=True, identity_memory_reliable_threshold=0.0, identity_memory_length=2
        )
        boxes = torch.tensor([[self._SLOT0_BOX, self._QUIET_BOX]])
        state = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.6, 0.05], boxes=boxes, features=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        assert state.table.slots[0].identity_memory == ()  # birth never seeds memory

        for frame_index, score in enumerate([0.6, 0.95, 0.7], start=1):
            state = transition_lifecycle(
                state.table,
                state.state,
                _frame(
                    [score, 0.05],
                    boxes=boxes,
                    features=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
                    input_active_mask=state.state.active_mask,
                ),
                config,
                _PERSON_SCHEMA,
                max_active_tracks=1,
                frame_index=frame_index,
            )

        memory = state.table.slots[0].identity_memory
        assert len(memory) == 2
        weights = sorted(entry[0] for entry in memory)
        assert weights[0] == pytest.approx(0.7, abs=1e-3)
        assert weights[1] == pytest.approx(0.95, abs=1e-3)

    def test_termination_clears_identity_memory(self) -> None:
        """A terminated slot's identity memory is discarded, matching motion-state clearing."""
        seeded, config = self._two_identity_setup(max_missed_frames=0)

        terminated = transition_lifecycle(
            seeded.table,
            seeded.state,
            _frame(
                [0.05, 0.9, 0.05],
                boxes=self._boxes(),
                features=self._features((0.0, 0.0), (0.0, 1.0)),
                input_active_mask=seeded.state.active_mask,
            ),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=2,
        )
        assert terminated.table.slots[0].status == "inactive"
        assert terminated.table.slots[0].identity_memory == ()
        assert [event.kind for event in terminated.events if event.slot == 0] == ["terminated"]


class TestTentativeTrackPool:
    """Structural invariants of the host-only tentative candidate registry (PRD US-002)."""

    def _record(self, **overrides: object) -> TentativeRecord:
        fields: dict[str, object] = {
            "tentative_id": 0,
            "class_id": 0,
            "first_frame": 0,
            "last_frame": 0,
            "hits": 1,
            "misses": 0,
            "score_history": (0.9,),
            "last_observed_box": (0.5, 0.5, 0.1, 0.1),
        }
        fields.update(overrides)
        return TentativeRecord(**fields)

    def test_empty_pool_has_no_records_and_a_zeroed_counter(self) -> None:
        pool = TentativeTrackPool.empty()

        assert pool.records == ()
        assert pool.next_tentative_id == 0

    def test_pool_and_record_are_immutable(self) -> None:
        record = self._record()
        pool = TentativeTrackPool((record,), 1)

        with pytest.raises(AttributeError):
            record.hits = 2  # type: ignore[misc]
        with pytest.raises(AttributeError):
            pool.records = ()  # type: ignore[misc]

    def test_duplicate_tentative_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            TentativeTrackPool((self._record(tentative_id=0), self._record(tentative_id=0)), 1)

    def test_next_id_must_exceed_every_assigned_tentative_id(self) -> None:
        with pytest.raises(ValueError, match="greater than every assigned"):
            TentativeTrackPool((self._record(tentative_id=5),), 5)

    def test_negative_next_tentative_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            TentativeTrackPool((), -1)

    def test_record_requires_nonnegative_hit_and_miss_counters(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            self._record(misses=-1)

    def test_record_cannot_end_before_it_started(self) -> None:
        with pytest.raises(ValueError, match="cannot end before it started"):
            self._record(first_frame=5, last_frame=4)

    def test_record_requires_at_least_one_score_observation(self) -> None:
        with pytest.raises(ValueError, match="at least one score observation"):
            self._record(score_history=())


class TestConfirmationBlockedByActiveDuplicate:
    """A discovery already representing a reliable active track (PRD US-004, Section 5.2)."""

    def test_discovery_duplicating_an_active_track_cannot_start_a_tentative(self) -> None:
        """A discovery landing on an already-active track is suppressed, not pooled."""
        boxes = torch.tensor([[[0.1, 0.1, 0.05, 0.05], [0.1, 0.1, 0.05, 0.05], [0.9, 0.9, 0.05, 0.05]]])
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.05, 0.05], boxes=boxes),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.9, 0.9, 0.05], boxes=boxes, input_active_mask=first.state.active_mask),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=2,
            frame_index=1,
        )

        assert second.tentative_pool.records == ()
        assert [event.kind for event in second.events] == ["duplicate_suppressed"]
        suppression = second.events[0]
        assert (suppression.slot, suppression.reason, suppression.compared_status) == (1, "duplicate_overlap", "active")

    def test_discovery_duplicating_an_active_track_cannot_confirm_a_pending_tentative(self) -> None:
        """A drifted active track now overlapping a pending tentative blocks its confirmation.

        The tentative starts far from any active track (a legitimate new-object candidate).
        By the confirming frame, the existing active track has moved into that same region
        (simulating drift) -- so the fresh discovery that would otherwise confirm the tentative
        also duplicates the active track. PRD Section 5.2 requires this discovery be excluded
        from confirmation eligibility entirely; it still reaches the birth loop and is suppressed
        there against the active track exactly like an ordinary duplicate discovery.
        """
        far_box = [0.1, 0.1, 0.05, 0.05]
        near_box = [0.5, 0.5, 0.05, 0.05]
        boxes0 = torch.tensor([[far_box, [0.9, 0.05, 0.02, 0.02], [0.9, 0.9, 0.05, 0.05]]])
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.05, 0.05], boxes=boxes0),
            _immediate_config(),
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        assert first.table.slot_track_ids == (0, None, None)

        boxes1 = torch.tensor([[far_box, [0.9, 0.05, 0.02, 0.02], near_box]])
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.9, 0.05, 0.9], boxes=boxes1, input_active_mask=first.state.active_mask),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=2,
            frame_index=1,
        )
        assert [event.kind for event in second.events] == ["tentative_started"]
        assert len(second.tentative_pool.records) == 1

        boxes2 = torch.tensor([[near_box, [0.9, 0.05, 0.02, 0.02], near_box]])
        third = transition_lifecycle(
            second.table,
            second.state,
            _frame([0.9, 0.05, 0.9], boxes=boxes2, input_active_mask=second.state.active_mask),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            second.tentative_pool,
            max_active_tracks=2,
            frame_index=2,
        )

        assert "confirmed" not in [event.kind for event in third.events]
        suppression = next(event for event in third.events if event.kind == "duplicate_suppressed")
        assert (suppression.slot, suppression.reason, suppression.compared_status) == (2, "duplicate_overlap", "active")
        # The tentative survives unconfirmed: it was never offered as an eligible discovery, so it
        # is simply unmatched this frame rather than cancelled (one miss, well under the default
        # tentative_max_misses).
        assert len(third.tentative_pool.records) == 1
        assert third.tentative_pool.records[0].misses == 1


class TestActiveTrackDuplicateCollisionArbitration:
    """Deterministic arbitration between already-active duplicate tracks (PRD US-004, Section 5.2).

    Every scenario uses three decoder slots: two that become active tracks under test, and a
    third, permanently below-threshold "padding" slot -- ``max_active_tracks`` must leave at least
    one discovery slot free (PRD's fixed-capacity invariant), so two simultaneously active tracks
    require three total query slots.
    """

    def _birth_two_disjoint_tracks(self, *, scores: list[float], config: TrackingSessionConfig) -> LifecycleTransition:
        boxes = torch.tensor([[[0.1, 0.1, 0.05, 0.05], [0.9, 0.9, 0.05, 0.05], [0.5, 0.9, 0.02, 0.02]]])
        return transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([*scores, 0.05], boxes=boxes),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )

    def test_sustained_duplicate_pair_suppresses_the_losers_emission_and_records_full_diagnostics(self) -> None:
        """Two independently active tracks that converge emit at most one observation per frame."""
        config = TrackingSessionConfig(collision_persistence_frames=2, collision_loser_outcome="suspended")
        first = self._birth_two_disjoint_tracks(scores=[0.9, 0.85], config=_immediate_config())
        assert first.table.slot_track_ids == (0, 1, None)

        collided_boxes = torch.tensor([[[0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], [0.5, 0.9, 0.02, 0.02]]])
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.9, 0.85, 0.05], boxes=collided_boxes, input_active_mask=first.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=1,
        )

        assert second.emitted_slots == (0,)
        assert len(second.collision_events) == 1
        event = second.collision_events[0]
        assert isinstance(event, CollisionArbitrationEvent)
        assert (event.winner_track_id, event.loser_track_id) == (0, 1)
        assert (event.winner_slot, event.loser_slot) == (0, 1)
        assert event.winner_score == pytest.approx(0.9, abs=1e-5)
        assert event.loser_score == pytest.approx(0.85, abs=1e-5)
        assert (event.winner_age, event.loser_age) == (2, 2)
        assert event.class_id == 0
        assert event.iou == pytest.approx(1.0, abs=1e-5)
        assert event.collision_streak == 1
        assert event.decision == "remains_active"
        # Not yet at collision_persistence_frames=2: the loser's own lifecycle state is untouched.
        assert second.table.slots[1].status == "active"
        assert second.table.slot_track_ids == (0, 1, None)

        third = transition_lifecycle(
            second.table,
            second.state,
            _frame([0.9, 0.85, 0.05], boxes=collided_boxes, input_active_mask=second.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=2,
        )

        assert third.emitted_slots == (0,)
        assert len(third.collision_events) == 1
        assert third.collision_events[0].decision == "suspended"
        assert third.collision_events[0].collision_streak == 2
        # Suppression never silently deletes lifecycle state: the loser keeps its track ID.
        assert third.table.slots[1].status == "suspended"
        assert third.table.slot_track_ids == (0, 1, None)
        assert third.table.slots[1].collision_streak == 0

    def test_collision_persistence_terminates_the_loser_when_configured(self) -> None:
        """The ``"terminated"`` outcome recycles the loser's slot once persistence is reached."""
        config = TrackingSessionConfig(collision_persistence_frames=1, collision_loser_outcome="terminated")
        first = self._birth_two_disjoint_tracks(scores=[0.9, 0.85], config=_immediate_config())

        collided_boxes = torch.tensor([[[0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], [0.5, 0.9, 0.02, 0.02]]])
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.9, 0.85, 0.05], boxes=collided_boxes, input_active_mask=first.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=1,
        )

        assert second.emitted_slots == (0,)
        assert second.collision_events[0].decision == "terminated"
        assert second.table.slots[1].status == "inactive"
        assert second.table.slot_track_ids == (0, None, None)

    def test_single_frame_crossing_does_not_permanently_merge_either_identity(self) -> None:
        """A one-frame overlap between two genuine objects never merges or terminates identity."""
        config = TrackingSessionConfig(collision_persistence_frames=3)
        first = self._birth_two_disjoint_tracks(scores=[0.9, 0.85], config=_immediate_config())

        crossing_boxes = torch.tensor([[[0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], [0.5, 0.9, 0.02, 0.02]]])
        crossing = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.9, 0.85, 0.05], boxes=crossing_boxes, input_active_mask=first.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=1,
        )
        assert crossing.emitted_slots == (0,)
        assert crossing.table.slot_track_ids == (0, 1, None)
        assert crossing.table.slots[1].status == "active"

        separated_boxes = torch.tensor([[[0.1, 0.1, 0.05, 0.05], [0.9, 0.9, 0.05, 0.05], [0.5, 0.9, 0.02, 0.02]]])
        separated = transition_lifecycle(
            crossing.table,
            crossing.state,
            _frame([0.9, 0.85, 0.05], boxes=separated_boxes, input_active_mask=crossing.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=2,
        )

        assert separated.emitted_slots == (0, 1)
        assert separated.table.slot_track_ids == (0, 1, None)
        assert separated.table.slots[1].collision_streak == 0

    def test_arbitration_winner_is_determined_by_score_not_slot_index(self) -> None:
        """The higher-confidence slot wins even when it is not the lower decoder index."""
        config = TrackingSessionConfig()
        first = self._birth_two_disjoint_tracks(scores=[0.85, 0.9], config=_immediate_config())
        # Higher score (slot 1, 0.9) is processed first by the birth loop and claims track_id 0.
        assert first.table.slot_track_ids == (1, 0, None)

        collided_boxes = torch.tensor([[[0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.05, 0.05], [0.5, 0.9, 0.02, 0.02]]])
        result = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.85, 0.9, 0.05], boxes=collided_boxes, input_active_mask=first.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=1,
        )

        assert result.emitted_slots == (1,)
        event = result.collision_events[0]
        assert (event.winner_slot, event.loser_slot) == (1, 0)
        assert (event.winner_track_id, event.loser_track_id) == (0, 1)

    def test_different_class_overlap_is_not_a_collision(self) -> None:
        """Class-aware arbitration never suppresses a same-box detection of a different class."""
        schema = ClassSchema(
            foreground_classes=(
                ForegroundClass(class_id=0, name="cat", external_category_id=7),
                ForegroundClass(class_id=1, name="dog", external_category_id=12),
            ),
            background_logit_index=2,
        )
        shared_box = [0.5, 0.5, 0.4, 0.4]
        padding_box = [0.02, 0.98, 0.02, 0.02]
        boxes = torch.tensor([[shared_box, shared_box, padding_box]])
        base = _frame([0.1, 0.1, 0.1], boxes=boxes)
        birth_frame = TrackingFrameOutput(
            pred_logits=torch.tensor([[[6.0, -5.0, -10.0], [-5.0, 6.0, -10.0], [-5.0, -5.0, 10.0]]]),
            pred_boxes=base.pred_boxes,
            candidate_state=base.candidate_state,
            input_active_mask=base.input_active_mask,
        )
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            birth_frame,
            _immediate_config(),
            schema,
            max_active_tracks=2,
            frame_index=0,
        )
        assert first.table.slot_track_ids == (0, 1, None)
        assert [slot.class_id for slot in first.table.slots[:2]] == [0, 1]

        continuation_frame = TrackingFrameOutput(
            pred_logits=birth_frame.pred_logits,
            pred_boxes=base.pred_boxes,
            candidate_state=base.candidate_state,
            input_active_mask=first.state.active_mask,
        )
        result = transition_lifecycle(
            first.table,
            first.state,
            continuation_frame,
            _immediate_config(),
            schema,
            max_active_tracks=2,
            frame_index=1,
        )

        assert result.emitted_slots == (0, 1)
        assert result.collision_events == ()


class TestLifecycleInvariantValidation:
    """Direct unit tests for ``validate_lifecycle_invariants`` (PRD US-005).

    Each violation test crafts a ``LifecycleTransition`` whose ``events``/``emitted_slots`` lie
    about what the committed ``table``/``state`` actually contain. Every dataclass in
    ``lifecycle.py`` already refuses to *construct* a genuinely inconsistent
    ``TrackSlotTable``/``TrackQueryState`` pair (duplicate track IDs, an inactive slot owning an
    identity, and so on), so these tests exercise the one place such a lie could still slip
    through undetected: a diagnostic event stream that no longer matches the state it describes.
    """

    def test_a_genuine_confirmation_from_the_production_code_path_passes_validation(self) -> None:
        """Sanity check: two-hit independent-rediscovery confirmation never raises."""
        config = TrackingSessionConfig(
            tentative_confirmation_hits=2,
            tentative_confirmation_window_frames=3,
            tentative_association_iou_threshold=0.5,
        )
        box = torch.tensor([0.3, 0.3, 0.1, 0.1])
        boxes0 = torch.zeros(1, 3, 4)
        boxes0[0, 0] = box
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.1, 0.1], boxes=boxes0),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=2,
            frame_index=0,
        )
        validate_lifecycle_invariants(first, _PERSON_SCHEMA)

        boxes1 = torch.zeros(1, 3, 4)
        boxes1[0, 2] = box
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.1, 0.9], boxes=boxes1, input_active_mask=first.state.active_mask),
            config,
            _PERSON_SCHEMA,
            first.tentative_pool,
            max_active_tracks=2,
            frame_index=1,
        )

        assert any(event.kind == "confirmed" for event in second.events)
        validate_lifecycle_invariants(second, _PERSON_SCHEMA)

    def test_a_tentative_started_event_that_also_carries_a_track_id_is_rejected(self) -> None:
        """Criterion 1: a tentative-scoped event can never claim a public identity."""
        table = TrackSlotTable((_slot(),), next_track_id=0, last_frame_index=0)
        transition = LifecycleTransition(
            table=table,
            state=_crafted_state([False]),
            tentative_pool=TentativeTrackPool((TentativeRecord(0, 0, 0, 0, 1, 0, (0.9,), (0.5, 0.5, 0.1, 0.1)),), 1),
            events=(LifecycleEvent("tentative_started", 0, track_id=5, frame_index=0, tentative_id=0),),
        )

        with pytest.raises(LifecycleInvariantError, match="carries a public track_id"):
            validate_lifecycle_invariants(transition, _PERSON_SCHEMA)

    def test_a_tentative_started_event_whose_own_discovery_slot_went_active_is_rejected(self) -> None:
        """Criterion 1: reproduces the exact pre-fix self-confirmation shape at the validator level.

        The old semantics copied a first-hit discovery straight into ``active_mask``; this test
        proves ``validate_lifecycle_invariants`` catches that shape even if some future change to
        the production code accidentally reintroduced it, without requiring a live model run.
        """
        table = TrackSlotTable((_slot(track_id=0, status="active", age=1, hits=1, last_reliable_frame=0),), 1, 0)
        transition = LifecycleTransition(
            table=table,
            state=_crafted_state([True]),
            tentative_pool=TentativeTrackPool((TentativeRecord(0, 0, 0, 0, 1, 0, (0.9,), (0.5, 0.5, 0.1, 0.1)),), 1),
            events=(LifecycleEvent("tentative_started", 0, track_id=None, frame_index=0, tentative_id=0),),
        )

        with pytest.raises(LifecycleInvariantError, match="became active"):
            validate_lifecycle_invariants(transition, _PERSON_SCHEMA)

    def test_two_confirmations_of_the_same_tentative_in_one_frame_are_rejected(self) -> None:
        """Criterion 3: a tentative can be consumed by at most one confirmation per frame."""
        table = TrackSlotTable(
            (
                _slot(track_id=0, status="active", age=1, hits=1, last_reliable_frame=0, class_id=0),
                _slot(track_id=1, status="active", age=1, hits=1, last_reliable_frame=0, class_id=0),
            ),
            2,
            0,
        )
        transition = LifecycleTransition(
            table=table,
            state=_crafted_state([True, True]),
            tentative_pool=TentativeTrackPool.empty(),
            events=(
                LifecycleEvent("confirmed", 0, track_id=0, frame_index=0, tentative_id=7),
                LifecycleEvent("confirmed", 1, track_id=1, frame_index=0, tentative_id=7),
            ),
            emitted_slots=(0, 1),
        )

        with pytest.raises(LifecycleInvariantError, match="same tentative"):
            validate_lifecycle_invariants(transition, _PERSON_SCHEMA)

    def test_two_confirmations_consuming_the_same_discovery_slot_are_rejected(self) -> None:
        """Criterion 3: a discovery can confirm at most one tentative per frame."""
        table = TrackSlotTable(
            (_slot(track_id=0, status="active", age=1, hits=1, last_reliable_frame=0, class_id=0),), 1, 0
        )
        transition = LifecycleTransition(
            table=table,
            state=_crafted_state([True]),
            tentative_pool=TentativeTrackPool.empty(),
            events=(
                LifecycleEvent("confirmed", 0, track_id=0, frame_index=0, tentative_id=1),
                LifecycleEvent("confirmed", 0, track_id=0, frame_index=0, tentative_id=2),
            ),
            emitted_slots=(0,),
        )

        with pytest.raises(LifecycleInvariantError, match="same discovery slot"):
            validate_lifecycle_invariants(transition, _PERSON_SCHEMA)

    def test_a_confirmed_tentative_still_present_in_the_pool_is_rejected(self) -> None:
        """Criterion 1/3: confirmation must discard the tentative record, never backfill it."""
        table = TrackSlotTable(
            (_slot(track_id=0, status="active", age=1, hits=1, last_reliable_frame=0, class_id=0),), 1, 0
        )
        record = TentativeRecord(3, 0, 0, 0, 2, 0, (0.9, 0.9), (0.5, 0.5, 0.1, 0.1))
        transition = LifecycleTransition(
            table=table,
            state=_crafted_state([True]),
            tentative_pool=TentativeTrackPool((record,), 4),
            events=(LifecycleEvent("confirmed", 0, track_id=0, frame_index=0, tentative_id=3),),
            emitted_slots=(0,),
        )

        with pytest.raises(LifecycleInvariantError, match="still lives in the pool"):
            validate_lifecycle_invariants(transition, _PERSON_SCHEMA)

    def test_the_same_public_track_id_emitted_twice_in_one_frame_is_rejected(self) -> None:
        """Criterion 4: no public track may emit more than one row per source frame."""
        table = TrackSlotTable(
            (_slot(track_id=0, status="active", age=1, hits=1, last_reliable_frame=0, class_id=0),), 1, 0
        )
        transition = LifecycleTransition(
            table=table,
            state=_crafted_state([True]),
            tentative_pool=TentativeTrackPool.empty(),
            events=(),
            emitted_slots=(0, 0),
        )

        with pytest.raises(LifecycleInvariantError, match="emitted more than once"):
            validate_lifecycle_invariants(transition, _PERSON_SCHEMA)

    def test_an_emitted_row_outside_the_foreground_schema_is_rejected(self) -> None:
        """Criterion 5: no exported row may carry a class ID the schema never declared."""
        table = TrackSlotTable(
            (_slot(track_id=0, status="active", age=1, hits=1, last_reliable_frame=0, class_id=99),), 1, 0
        )
        transition = LifecycleTransition(
            table=table,
            state=_crafted_state([True]),
            tentative_pool=TentativeTrackPool.empty(),
            events=(),
            emitted_slots=(0,),
        )

        with pytest.raises(LifecycleInvariantError, match="outside the declared foreground schema"):
            validate_lifecycle_invariants(transition, _PERSON_SCHEMA)


class TestLifecycleInvariantPropertySequences:
    """Property-style regression coverage for ``validate_lifecycle_invariants`` (PRD US-005).

    Drives many independent chronological streams -- each with its own randomized capacity,
    lifecycle-policy knobs, per-frame scores, boxes, and non-unit frame gaps -- through
    ``transition_lifecycle`` under a fixed seed, validating every produced transition. A fixed
    seed keeps any future counterexample immediately reproducible.
    """

    _SEED = 20260826
    _TRIALS = 40
    _STEPS_PER_TRIAL = 25

    def test_random_sequences_never_violate_lifecycle_invariants(self) -> None:
        rng = random.Random(self._SEED)
        for _ in range(self._TRIALS):
            num_queries = rng.randint(3, 6)
            max_active_tracks = rng.randint(1, num_queries - 1)
            window_frames = rng.choice([2, 3, 4])
            config = TrackingSessionConfig(
                activation_threshold=rng.uniform(0.3, 0.7),
                continuation_threshold=rng.uniform(0.1, 0.5),
                duplicate_iou_threshold=rng.uniform(0.4, 0.8),
                tentative_confirmation_hits=rng.randint(1, window_frames),
                tentative_confirmation_window_frames=window_frames,
                tentative_max_misses=rng.choice([1, 2, 3]),
                tentative_association_iou_threshold=rng.uniform(0.1, 0.5),
                max_discovery_candidates_per_frame=rng.choice([2, 4, 10]),
                max_tentative_tracks=rng.choice([1, 2, 5]),
                collision_iou_threshold=rng.uniform(0.4, 0.8),
                collision_persistence_frames=rng.choice([1, 2, 3]),
                collision_loser_outcome=rng.choice(["remains_active", "suspended", "terminated"]),
                reassociation_enabled=rng.choice([True, False]),
                reassociation_iou_threshold=rng.uniform(0.3, 0.7),
                motion_reference_prediction_enabled=rng.choice([True, False]),
                identity_memory_enabled=rng.choice([True, False]),
            )
            table = TrackSlotTable.empty(num_queries)
            state = TrackQueryState.empty(1, num_queries, 2)
            pool = TentativeTrackPool.empty()
            frame_index = -1
            for _ in range(self._STEPS_PER_TRIAL):
                frame_index += rng.choice([1, 1, 1, 2, 3])
                scores = [rng.uniform(0.01, 0.99) for _ in range(num_queries)]
                boxes = torch.zeros(1, num_queries, 4)
                for index in range(num_queries):
                    boxes[0, index] = torch.tensor(
                        [
                            rng.uniform(0.1, 0.9),
                            rng.uniform(0.1, 0.9),
                            rng.uniform(0.05, 0.3),
                            rng.uniform(0.05, 0.3),
                        ]
                    )
                features = torch.rand(1, num_queries, 2)
                frame = _frame(scores, boxes=boxes, features=features, input_active_mask=state.active_mask)

                transition = transition_lifecycle(
                    table,
                    state,
                    frame,
                    config,
                    _PERSON_SCHEMA,
                    pool,
                    max_active_tracks=max_active_tracks,
                    frame_index=frame_index,
                )

                validate_lifecycle_invariants(transition, _PERSON_SCHEMA)

                table, state, pool = transition.table, transition.state, transition.tentative_pool


class TestSelfConfirmationRegressionPattern:
    """Reproduces the PRD's documented "106-of-107 self-confirmations" failure (PRD US-005).

    ``_old_semantics_self_confirms`` is a minimal standalone re-creation of the pre-fix behavior
    described in PRD Section 1: a first discovery was copied straight into recurrent query state
    and marked ``active_mask=True``, so *that same slot* could supply its own next-frame
    confirmation evidence merely by staying above the continuation threshold. It exists only in
    this test, deliberately isolated from production code, to give the "106-of-107" claim and its
    corrected-semantics counter-claim an executable, side-by-side proof.
    """

    def _old_semantics_self_confirms(self, second_frame_score: float, *, continuation_threshold: float) -> bool:
        """Mirror of the pre-fix rule: a first-hit slot confirms itself once it re-clears threshold."""
        return second_frame_score >= continuation_threshold

    def test_old_semantics_self_confirms_106_of_107_candidates(self) -> None:
        """106 candidates persist above threshold at their own first-hit slot; one drifts below it."""
        continuation_threshold = 0.3
        second_frame_scores = [0.9] * 106 + [0.1]

        self_confirmed = sum(
            self._old_semantics_self_confirms(score, continuation_threshold=continuation_threshold)
            for score in second_frame_scores
        )

        assert self_confirmed == 106
        assert len(second_frame_scores) == 107

    def test_corrected_semantics_self_confirms_none_of_the_same_107_candidates(self) -> None:
        """The identical 107 same-slot score trajectories confirm zero tracks under the fix.

        Each candidate is an independent two-frame stream: a tentative starts at frame 0, and the
        *same query index* echoes the documented second-frame score at frame 1 -- exactly the
        input that self-confirmed under old semantics above. Because a tentative never occupies a
        decoder slot or feeds its own evidence forward (PRD Section 5.1), and confirmation
        requires an independent later-frame discovery from an *inactive* slot associating with the
        pool, that discovery is available at every other query index but never at the one that
        merely started the tentative -- so none of these candidates ever produce a ``"confirmed"``
        event or an active track.
        """
        continuation_threshold = 0.3
        second_frame_scores = [0.9] * 106 + [0.1]
        config = TrackingSessionConfig(continuation_threshold=continuation_threshold)

        confirmed_count = 0
        for second_frame_score in second_frame_scores:
            table = TrackSlotTable.empty(2)
            state = TrackQueryState.empty(1, 2, 2)
            first = transition_lifecycle(
                table,
                state,
                _frame([0.9, 0.05]),
                config,
                _PERSON_SCHEMA,
                max_active_tracks=1,
                frame_index=0,
            )
            assert not first.state.active_mask[0, 0]
            validate_lifecycle_invariants(first, _PERSON_SCHEMA)

            second = transition_lifecycle(
                first.table,
                first.state,
                _frame([second_frame_score, 0.05], input_active_mask=first.state.active_mask),
                config,
                _PERSON_SCHEMA,
                first.tentative_pool,
                max_active_tracks=1,
                frame_index=1,
            )
            validate_lifecycle_invariants(second, _PERSON_SCHEMA)
            if any(event.kind == "confirmed" for event in second.events):
                confirmed_count += 1
            assert not second.state.active_mask[0, 0]
            assert second.table.slot_track_ids == (None, None)

        assert confirmed_count == 0
        assert len(second_frame_scores) == 107
