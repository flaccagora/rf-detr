# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from dataclasses import FrozenInstanceError

import pytest
import torch

from rfdetr.models.tracking import TrackingFrameOutput, TrackQueryState


class TestTrackQueryState:
    """Behavioral tests for the neural persistent-query state contract."""

    def test_empty_state_is_explicitly_inactive_and_immutable_style(self) -> None:
        """The empty-state factory creates aligned tensors without marking zero values as historical state."""
        state = TrackQueryState.empty(batch_size=2, num_queries=3, hidden_dim=4)

        assert state.query_features.shape == (2, 3, 4)
        assert state.reference_boxes.shape == (2, 3, 4)
        assert state.active_mask.shape == (2, 3)
        assert not state.active_mask.any()
        with pytest.raises(FrozenInstanceError):
            state.active_mask = torch.ones_like(state.active_mask)  # type: ignore[misc]

    def test_state_rejects_misaligned_slots(self) -> None:
        """Every tensor uses the same batch and query-slot axes."""
        with pytest.raises(ValueError, match="slot-aligned"):
            TrackQueryState(
                query_features=torch.zeros(2, 3, 4),
                reference_boxes=torch.zeros(2, 2, 4),
                active_mask=torch.zeros(2, 3, dtype=torch.bool),
            )

    @pytest.mark.parametrize(
        ("reference_boxes", "message"),
        [
            pytest.param(
                torch.tensor([[[float("nan"), 0.0, 0.0, 0.0]]]),
                "finite",
                id="non-finite",
            ),
            pytest.param(
                torch.tensor([[[0.5, 0.5, 1.1, 0.2]]]),
                "normalized",
                id="outside-normalized-domain",
            ),
        ],
    )
    def test_state_rejects_invalid_normalized_boxes(self, reference_boxes: torch.Tensor, message: str) -> None:
        """State boxes are finite normalized cxcywh values."""
        with pytest.raises(ValueError, match=message):
            TrackQueryState(
                query_features=torch.zeros(1, 1, 4),
                reference_boxes=reference_boxes,
                active_mask=torch.zeros(1, 1, dtype=torch.bool),
            )

    def test_empty_state_respects_requested_device_and_dtype(self) -> None:
        """All empty-state tensors are colocated and floating tensors share a dtype."""
        state = TrackQueryState.empty(
            batch_size=1,
            num_queries=2,
            hidden_dim=3,
            device="cpu",
            dtype=torch.float64,
        )

        assert state.query_features.dtype == torch.float64
        assert state.reference_boxes.dtype == torch.float64
        assert state.active_mask.dtype == torch.bool
        assert {tensor.device for tensor in (state.query_features, state.reference_boxes, state.active_mask)} == {
            torch.device("cpu")
        }


class TestTrackingFrameOutput:
    """Behavioral tests for the slot-aligned frame result."""

    def test_frame_output_keeps_predictions_state_and_input_roles_aligned(self) -> None:
        """A frame result carries predictions, candidate recurrence, and the original role map."""
        candidate_state = TrackQueryState.empty(batch_size=2, num_queries=3, hidden_dim=4)
        input_active_mask = torch.tensor(
            [[True, False, False], [False, True, False]],
            dtype=torch.bool,
        )

        output = TrackingFrameOutput(
            pred_logits=torch.zeros(2, 3, 5),
            pred_boxes=torch.zeros(2, 3, 4),
            candidate_state=candidate_state,
            input_active_mask=input_active_mask,
            aux_outputs=({"pred_logits": torch.zeros(2, 3, 5)},),
            enc_outputs={"pred_boxes": torch.zeros(2, 3, 4)},
        )

        assert output.candidate_state is candidate_state
        assert torch.equal(output.input_active_mask, input_active_mask)
        assert len(output.aux_outputs) == 1

    def test_frame_output_rejects_candidate_state_with_different_slots(self) -> None:
        """Candidate recurrence tensors cannot silently drift from prediction slot order."""
        with pytest.raises(ValueError, match="candidate_state must be slot-aligned"):
            TrackingFrameOutput(
                pred_logits=torch.zeros(1, 3, 5),
                pred_boxes=torch.zeros(1, 3, 4),
                candidate_state=TrackQueryState.empty(batch_size=1, num_queries=2, hidden_dim=4),
                input_active_mask=torch.zeros(1, 3, dtype=torch.bool),
            )
