# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.config import TrackingSessionConfig
from rfdetr.models.tracking import TrackingFrameOutput, TrackQueryState
from rfdetr.tracking.lifecycle import TrackSlotTable, transition_lifecycle


def _frame(
    scores: list[float],
    *,
    boxes: torch.Tensor | None = None,
    features: torch.Tensor | None = None,
    input_active_mask: torch.Tensor | None = None,
) -> TrackingFrameOutput:
    """Build a single-class frame result with predictable sigmoid confidences."""
    logits = torch.logit(torch.tensor(scores)).view(1, -1, 1)
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

    def test_reliable_discovery_activates_with_monotonic_ids_and_commits_state(self) -> None:
        """Eligible discoveries activate up to capacity and carry candidate neural state."""
        table = TrackSlotTable.empty(num_queries=3)
        prior_state = TrackQueryState.empty(batch_size=1, num_queries=3, hidden_dim=2)

        result = transition_lifecycle(
            table,
            prior_state,
            _frame([0.9, 0.8, 0.1]),
            TrackingSessionConfig(activation_threshold=0.5),
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
            TrackingSessionConfig(),
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
            TrackingSessionConfig(max_missed_frames=3),
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
            TrackingSessionConfig(max_missed_frames=3),
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
            TrackingSessionConfig(),
            max_active_tracks=1,
            frame_index=0,
        )
        terminated = transition_lifecycle(
            activated.table,
            activated.state,
            _frame([0.9, 0.1], input_active_mask=activated.state.active_mask),
            TrackingSessionConfig(continuation_threshold=0.95, max_missed_frames=0),
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
            TrackingSessionConfig(),
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
            TrackingSessionConfig(duplicate_iou_threshold=0.5),
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
                TrackingSessionConfig(),
                max_active_tracks=1,
                frame_index=0,
            )
