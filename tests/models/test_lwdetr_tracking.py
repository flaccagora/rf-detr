# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Behavioral tests for the low-level LWDETR tracking forward path."""

import copy
from unittest.mock import MagicMock

import torch

from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.tracking import TrackingFrameOutput, TrackQueryState
from rfdetr.utilities.tensors import NestedTensor


def _make_tracking_model(*, aux_loss: bool = False) -> tuple[LWDETR, MagicMock]:
    """Create a minimal two-stage detection model with deterministic decoder outputs."""
    batch_size = 2
    num_queries = 3
    hidden_dim = 4
    features = [
        NestedTensor(
            torch.zeros(batch_size, hidden_dim, 2, 2),
            torch.zeros(batch_size, 2, 2, dtype=torch.bool),
        )
    ]
    backbone = MagicMock(return_value=(features, [torch.zeros_like(features[0].tensors)], None))

    transformer = MagicMock()
    transformer.d_model = hidden_dim
    transformer.decoder = MagicMock()
    transformer.decoder.bbox_embed = None
    transformer_outputs = (
        torch.full((2, batch_size, num_queries, hidden_dim), 0.25),
        torch.zeros(2, batch_size, num_queries, 4),
        torch.full((batch_size, num_queries, hidden_dim), 0.5),
        torch.full((batch_size, num_queries, 4), 0.5),
    )
    transformer.return_value = transformer_outputs

    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=5,
        num_queries=num_queries,
        aux_loss=aux_loss,
        group_detr=1,
        two_stage=True,
        bbox_reparam=False,
    )
    return model, transformer


def test_forward_tracking_returns_aligned_predictions_and_candidate_state() -> None:
    """Tracking forward exposes final decoder state and preserves input slot roles."""
    model, transformer = _make_tracking_model(aux_loss=True)
    model.eval()
    prior_state = TrackQueryState(
        query_features=torch.zeros(2, 3, 4),
        reference_boxes=torch.full((2, 3, 4), 0.5),
        active_mask=torch.tensor([[True, False, False], [False, True, False]]),
    )

    output = model.forward_tracking(torch.ones(2, 3, 8, 8), prior_state)

    assert isinstance(output, TrackingFrameOutput)
    assert output.pred_logits.shape == (2, 3, 5)
    assert output.pred_boxes.shape == (2, 3, 4)
    assert torch.equal(output.candidate_state.query_features, torch.full((2, 3, 4), 0.25))
    assert torch.equal(output.candidate_state.reference_boxes, output.pred_boxes)
    assert output.candidate_state.active_mask.all()
    assert torch.equal(output.input_active_mask, prior_state.active_mask)
    assert len(output.aux_outputs) == 1
    assert output.enc_outputs is not None
    assert transformer.call_args.kwargs["prior_state"] is prior_state


def test_empty_tracking_state_matches_stateless_evaluation_predictions() -> None:
    """An omitted prior state preserves ordinary evaluation predictions."""
    model, transformer = _make_tracking_model()
    model.eval()
    samples = torch.ones(2, 3, 8, 8)

    expected = model(samples)
    actual = model.forward_tracking(samples, None)

    torch.testing.assert_close(actual.pred_logits, expected["pred_logits"])
    torch.testing.assert_close(actual.pred_boxes, expected["pred_boxes"])
    supplied_state = transformer.call_args.kwargs["prior_state"]
    assert isinstance(supplied_state, TrackQueryState)
    assert not supplied_state.active_mask.any()


def test_detection_and_tracking_checkpoints_share_one_strict_state_dict() -> None:
    """Current image weights initialize tracking, and tracking weights retain image prediction."""
    image_model, _ = _make_tracking_model()
    tracking_model, _ = _make_tracking_model()
    image_model.eval()
    tracking_model.eval()
    samples = torch.ones(2, 3, 8, 8)

    detection_checkpoint = copy.deepcopy(image_model.state_dict())
    tracking_model.load_state_dict(detection_checkpoint, strict=True)
    tracking_output = tracking_model.forward_tracking(samples)

    tracking_checkpoint = copy.deepcopy(tracking_model.state_dict())
    reloaded_image_model, _ = _make_tracking_model()
    reloaded_image_model.eval()
    reloaded_image_model.load_state_dict(tracking_checkpoint, strict=True)
    image_output = reloaded_image_model(samples)

    torch.testing.assert_close(image_output["pred_logits"], tracking_output.pred_logits)
    torch.testing.assert_close(image_output["pred_boxes"], tracking_output.pred_boxes)
