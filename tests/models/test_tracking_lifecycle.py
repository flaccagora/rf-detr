# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.config import ClassSchema, ForegroundClass, TrackingSessionConfig
from rfdetr.models.tracking import TrackingFrameOutput, TrackQueryState
from rfdetr.tracking.lifecycle import LifecycleEvent, TrackSlotTable, transition_lifecycle

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


class TestLifecycleTransitions:
    """Pure lifecycle behavior for a single tracking stream."""

    def test_tentative_birth_confirms_on_second_hit_without_early_identity(self) -> None:
        """A discovery persists privately and emits only after confirmation."""
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.1]),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )

        assert first.table.slots[0].status == "tentative"
        assert first.table.slot_track_ids == (None, None)
        assert first.table.next_track_id == 0
        assert first.state.active_mask.tolist() == [[True, False]]
        assert [(event.kind, event.track_id) for event in first.events] == [("tentative_started", None)]

        confirmed = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.8, 0.1], input_active_mask=first.state.active_mask),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=1,
        )

        assert confirmed.table.slots[0].status == "active"
        assert confirmed.table.slot_track_ids == (0, None)
        assert confirmed.table.next_track_id == 1
        assert [(event.kind, event.track_id) for event in confirmed.events] == [("confirmed", 0)]

    def test_nonconsecutive_tentative_hits_confirm_within_three_frame_window(self) -> None:
        """One intervening miss does not prevent a two-of-three confirmation."""
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.1]),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=4,
        )
        trusted_features = first.state.query_features.clone()
        missed = transition_lifecycle(
            first.table,
            first.state,
            _frame(
                [0.1, 0.1],
                features=torch.full((1, 2, 2), 99.0),
                input_active_mask=first.state.active_mask,
            ),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=5,
        )

        assert missed.table.slots[0].status == "tentative"
        assert missed.table.slots[0].missed_frames == 1
        assert torch.equal(missed.state.query_features, trusted_features)

        confirmed = transition_lifecycle(
            missed.table,
            missed.state,
            _frame([0.8, 0.1], input_active_mask=missed.state.active_mask),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=6,
        )

        assert confirmed.table.slot_track_ids == (0, None)
        assert [event.kind for event in confirmed.events] == ["confirmed"]

    @pytest.mark.parametrize("second_frame,second_score", [(2, 0.1), (3, 0.9)])
    def test_tentative_cancellation_clears_state_on_second_miss_or_expiry(
        self, second_frame: int, second_score: float
    ) -> None:
        """Cancellation is immediate for two misses or an expired birth window."""
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.1]),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        cancelled = transition_lifecycle(
            first.table,
            first.state,
            _frame([second_score, 0.1], input_active_mask=first.state.active_mask),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=second_frame,
        )

        assert cancelled.table.slots[0].status == "inactive"
        assert cancelled.table.next_track_id == 0
        assert not cancelled.state.active_mask[0, 0]
        assert not cancelled.state.query_features[0, 0].any()
        assert not cancelled.state.reference_boxes[0, 0].any()
        assert [event.kind for event in cancelled.events] == ["tentative_cancelled"]

    def test_cancelled_slot_recycles_without_consuming_an_identity(self) -> None:
        """A cancelled host/neural slot can be reused and receives an ID only on confirmation."""
        config = TrackingSessionConfig()
        first = transition_lifecycle(
            TrackSlotTable.empty(2),
            TrackQueryState.empty(1, 2, 2),
            _frame([0.9, 0.1]),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        cancelled = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.1, 0.1], input_active_mask=first.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=2,
        )
        recycled = transition_lifecycle(
            cancelled.table,
            cancelled.state,
            _frame([0.9, 0.1]),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=3,
        )
        confirmed = transition_lifecycle(
            recycled.table,
            recycled.state,
            _frame([0.9, 0.1], input_active_mask=recycled.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=4,
        )

        assert recycled.table.slots[0].status == "tentative"
        assert confirmed.table.slot_track_ids == (0, None)

    def test_confirmation_respects_durable_capacity(self) -> None:
        """Tentatives remain ID-less when durable capacity is full and confirm when it opens."""
        config = TrackingSessionConfig(max_missed_frames=0)
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.8, 0.1]),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=0,
        )
        pressured = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.9, 0.8, 0.1], input_active_mask=first.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=1,
        )

        assert pressured.table.slot_track_ids == (0, None, None)
        assert pressured.table.slots[1].status == "tentative"
        assert [event.kind for event in pressured.events] == ["confirmed", "capacity_suppressed"]

        available = transition_lifecycle(
            pressured.table,
            pressured.state,
            _frame([0.1, 0.8, 0.1], input_active_mask=pressured.state.active_mask),
            config,
            _PERSON_SCHEMA,
            max_active_tracks=1,
            frame_index=2,
        )

        assert available.table.slot_track_ids == (None, 1, None)
        assert [event.kind for event in available.events] == ["terminated", "confirmed"]

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
        """Discovery consideration and coexisting tentative slots are bounded independently."""
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

        assert sum(slot.status == "tentative" for slot in result.table.slots) == 10
        assert reasons.count("tentative_capacity") == 2
        assert reasons.count("discovery_candidate_limit") == num_queries - 12

    def test_long_false_positive_trajectory_cannot_exceed_tentative_capacity(self) -> None:
        """Sustained spurious discoveries never accumulate unbounded tentative slots."""
        num_queries = 8
        boxes = torch.stack(
            [torch.tensor([index / num_queries + 0.02, 0.5, 0.02, 0.02]) for index in range(num_queries)]
        ).unsqueeze(0)
        config = TrackingSessionConfig(max_tentative_tracks=2)
        table = TrackSlotTable.empty(num_queries)
        state = TrackQueryState.empty(1, num_queries, 2)

        for frame_index in range(6):
            transition = transition_lifecycle(
                table,
                state,
                _frame([0.9] * num_queries, boxes=boxes, input_active_mask=state.active_mask),
                config,
                _PERSON_SCHEMA,
                max_active_tracks=3,
                frame_index=frame_index,
            )
            table, state = transition.table, transition.state
            assert sum(slot.status == "tentative" for slot in table.slots) <= 2
            assert sum(slot.status in {"active", "suspended"} for slot in table.slots) <= 3

        assert table.next_track_id <= 3

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

    def test_discovery_overlapping_a_tentative_track_is_suppressed(self) -> None:
        """An unconfirmed tentative slot cannot be duplicated by a stronger discovery."""
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
            max_active_tracks=2,
            frame_index=1,
        )
        suppression = next(event for event in second.events if event.kind == "duplicate_suppressed")

        assert first.table.slots[0].status == "tentative"
        assert second.table.slots[1].status == "inactive"
        assert (suppression.slot, suppression.compared_status, suppression.overlap) == (
            1,
            "tentative",
            pytest.approx(1.0),
        )

    def test_confirmation_blocked_by_durable_capacity_reports_its_reason(self) -> None:
        """Capacity suppressions are distinguishable from duplicate suppressions."""
        boxes = torch.tensor([[[0.1, 0.1, 0.05, 0.05], [0.6, 0.6, 0.05, 0.05], [0.9, 0.9, 0.05, 0.05]]])
        first = transition_lifecycle(
            TrackSlotTable.empty(3),
            TrackQueryState.empty(1, 3, 2),
            _frame([0.9, 0.1, 0.1], boxes=boxes),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            max_active_tracks=0,
            frame_index=0,
        )
        second = transition_lifecycle(
            first.table,
            first.state,
            _frame([0.9, 0.1, 0.1], boxes=boxes, input_active_mask=first.state.active_mask),
            TrackingSessionConfig(),
            _PERSON_SCHEMA,
            max_active_tracks=0,
            frame_index=1,
        )
        suppression = next(event for event in second.events if event.kind == "capacity_suppressed")

        assert second.table.slot_track_ids == (None, None, None)
        assert (suppression.slot, suppression.reason, suppression.compared_status) == (0, "durable_capacity", None)
        assert suppression.class_id == 0

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
            _immediate_config(
                reassociation_enabled=True, reassociation_iou_threshold=0.9, duplicate_iou_threshold=0.5
            ),
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
            _frame(
                [0.1, 0.05], boxes=self._boxes([0.2, 0.5, 0.2, 0.2]), input_active_mask=second.state.active_mask
            ),
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
            _frame(
                [0.1, 0.05], boxes=self._boxes([0.5, 0.5, 0.2, 0.2]), input_active_mask=first.state.active_mask
            ),
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
            _frame(
                [0.1, 0.05], boxes=self._boxes([0.5, 0.5, 0.2, 0.2]), input_active_mask=first.state.active_mask
            ),
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
