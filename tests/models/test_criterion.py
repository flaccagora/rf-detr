# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for SetCriterion edge paths: _output_device and num_boxes_for_targets."""

import pytest
import torch

from rfdetr.models.criterion import SetCriterion, TrackingSetCriterion
from rfdetr.models.matcher import SequenceAssignment


class _MatcherStub:
    """Minimal matcher that returns identity indices for every target in the batch."""

    def __call__(self, outputs, targets, group_detr=1):
        return [(torch.arange(len(t["labels"])), torch.arange(len(t["labels"]))) for t in targets]


class _RecordingMatcher(_MatcherStub):
    """Matcher stub that records the output dictionaries it receives."""

    def __init__(self) -> None:
        """Initialize an empty call log."""
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(self, outputs, targets, group_detr=1):
        self.calls.append(outputs)
        return super().__call__(outputs, targets, group_detr)


def _sequence_assignment() -> SequenceAssignment:
    """Return one assignment with a continuation and a newborn discovery."""
    return SequenceAssignment(
        continuing_indices=(torch.tensor([0]), torch.tensor([1])),
        discovery_indices=(torch.tensor([2]), torch.tensor([0])),
        absent_query_indices=torch.tensor([1]),
        slot_track_ids=(20, 10, 30),
    )


def _bare_criterion() -> SetCriterion:
    """Return a SetCriterion with no losses so forward() is a no-op."""
    criterion = SetCriterion.__new__(SetCriterion)
    criterion.training = True
    criterion.group_detr = 1
    criterion.sum_group_losses = False
    criterion.losses = []
    criterion.weight_dict = {}
    criterion.matcher = _MatcherStub()
    criterion.num_keypoints_per_class = []
    return criterion


class TestOutputDevice:
    """Tests for SetCriterion._output_device — probes top-level tensor values only."""

    def test_returns_device_of_first_tensor(self):
        """Device inferred from the first tensor value in outputs."""
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}

        device = SetCriterion._output_device(outputs)

        assert device == torch.device("cpu")

    def test_raises_when_no_tensor_present(self):
        """ValueError raised when no top-level value is a tensor."""
        outputs = {"meta": "string_value", "count": 42}

        with pytest.raises(ValueError, match="at least one tensor"):
            SetCriterion._output_device(outputs)

    def test_skips_non_tensor_values(self):
        """Non-tensor entries at the top level are skipped; first tensor wins."""
        outputs = {"meta": "ignored", "pred_logits": torch.zeros(1, 1, 1)}

        device = SetCriterion._output_device(outputs)

        assert device == torch.device("cpu")


class TestNumBoxesForTargets:
    """Tests for SetCriterion.num_boxes_for_targets — clamp and empty-target edge cases."""

    def test_returns_tensor_gte_one(self):
        """Result must be clamped to >= 1.0 to prevent division by zero."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = [{"labels": torch.tensor([0, 1])}]

        result = criterion.num_boxes_for_targets(outputs, targets)

        assert result.item() >= 1.0

    def test_clamps_zero_box_count_to_one(self):
        """Empty targets (no labels) must clamp to 1.0 to avoid zero denominator."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = [{"labels": torch.zeros(0, dtype=torch.int64)}]

        result = criterion.num_boxes_for_targets(outputs, targets)

        assert result.item() == pytest.approx(1.0)

    def test_clamps_empty_target_list(self):
        """Empty target list (batch_size=0 edge case) must also clamp to 1.0."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = []

        result = criterion.num_boxes_for_targets(outputs, targets)

        assert result.item() == pytest.approx(1.0)

    def test_counts_labels_correctly(self):
        """Box count equals total number of labels across all targets in the batch."""
        criterion = _bare_criterion()
        outputs = {"pred_logits": torch.zeros(1, 1, 1)}
        targets = [
            {"labels": torch.tensor([0, 1])},
            {"labels": torch.tensor([0])},
        ]

        result = criterion.num_boxes_for_targets(outputs, targets)

        # 2 + 1 = 3 boxes; single-process so no all-reduce
        assert result.item() == pytest.approx(3.0)


class TestLossMasksEmptyMatch:
    """Tests for the dict-path zero-GT branch of SetCriterion.loss_masks."""

    def test_dict_path_zero_gt_stays_connected_to_graph(self):
        """Zero-match dict path returns a loss that back-propagates to every segmentation-head output."""
        criterion = _bare_criterion()
        spatial_features = torch.randn(1, 4, 8, 8, requires_grad=True)
        query_features = torch.randn(1, 5, 4, requires_grad=True)
        bias = torch.randn(1, requires_grad=True)
        outputs = {
            "pred_masks": {
                "spatial_features": spatial_features,
                "query_features": query_features,
                "bias": bias,
            }
        }
        empty = torch.empty(0, dtype=torch.long)
        indices = [(empty, empty)]

        losses = criterion.loss_masks(outputs, targets=[{}], indices=indices, num_boxes=1)

        assert losses["loss_mask_ce"].requires_grad
        (losses["loss_mask_ce"] + losses["loss_mask_dice"]).backward()
        assert spatial_features.grad is not None
        assert query_features.grad is not None
        assert bias.grad is not None


class TestTrackingSetCriterion:
    """Behavioral coverage for loss evaluation with sequence assignments."""

    def test_decoder_layers_reuse_precomputed_sequence_assignment(self) -> None:
        """Final and auxiliary decoder losses use fixed sequence matches without rematching."""
        matcher = _MatcherStub()
        criterion = TrackingSetCriterion(
            num_classes=2,
            matcher=matcher,
            weight_dict={},
            focal_alpha=0.25,
            losses=["boxes"],
        )
        outputs = {
            "pred_logits": torch.zeros(1, 3, 2),
            "pred_boxes": torch.tensor([[[0.8, 0.5, 0.1, 0.1], [0.4, 0.5, 0.1, 0.1], [0.2, 0.5, 0.1, 0.1]]]),
            "aux_outputs": [
                {
                    "pred_logits": torch.zeros(1, 3, 2),
                    "pred_boxes": torch.tensor([[[0.7, 0.5, 0.1, 0.1], [0.4, 0.5, 0.1, 0.1], [0.3, 0.5, 0.1, 0.1]]]),
                }
            ],
        }
        targets = [
            {
                "labels": torch.tensor([0, 1]),
                "boxes": torch.tensor([[0.2, 0.5, 0.1, 0.1], [0.8, 0.5, 0.1, 0.1]]),
            }
        ]

        losses = criterion(outputs, targets, assignments=[_sequence_assignment()])

        assert losses["loss_bbox"].item() == pytest.approx(0.0)
        assert losses["loss_bbox_0"].item() == pytest.approx(0.1)

    def test_encoder_losses_keep_ordinary_frame_matching(self) -> None:
        """Only encoder proposals invoke the ordinary matcher."""
        matcher = _RecordingMatcher()
        criterion = TrackingSetCriterion(
            num_classes=2,
            matcher=matcher,
            weight_dict={},
            focal_alpha=0.25,
            losses=["boxes"],
        )
        encoder_outputs = {
            "pred_logits": torch.zeros(1, 2, 2),
            "pred_boxes": torch.tensor([[[0.2, 0.5, 0.1, 0.1], [0.8, 0.5, 0.1, 0.1]]]),
        }
        outputs = {
            "pred_logits": torch.zeros(1, 3, 2),
            "pred_boxes": torch.zeros(1, 3, 4),
            "enc_outputs": encoder_outputs,
        }
        targets = [
            {
                "labels": torch.tensor([0, 1]),
                "boxes": torch.tensor([[0.2, 0.5, 0.1, 0.1], [0.8, 0.5, 0.1, 0.1]]),
            }
        ]

        losses = criterion(outputs, targets, assignments=[_sequence_assignment()])

        assert matcher.calls == [encoder_outputs]
        assert losses["loss_bbox_enc"].item() == pytest.approx(0.0)

    def test_empty_frame_box_losses_are_finite(self) -> None:
        """Empty frames clamp normalization and regress no absent slots."""
        criterion = TrackingSetCriterion(
            num_classes=2,
            matcher=_MatcherStub(),
            weight_dict={},
            focal_alpha=0.25,
            losses=["boxes"],
        )
        empty = torch.empty(0, dtype=torch.int64)
        assignment = SequenceAssignment(
            continuing_indices=(empty, empty),
            discovery_indices=(empty, empty),
            absent_query_indices=torch.tensor([0]),
            slot_track_ids=(10,),
        )
        outputs = {
            "pred_logits": torch.zeros(1, 1, 2),
            "pred_boxes": torch.zeros(1, 1, 4),
        }
        targets = [{"labels": empty, "boxes": torch.empty(0, 4)}]

        losses = criterion(outputs, targets, assignments=[assignment])

        assert losses["loss_bbox"].item() == pytest.approx(0.0)
        assert losses["loss_giou"].item() == pytest.approx(0.0)

    def test_absent_slot_is_classified_as_negative_without_box_target(self) -> None:
        """An absent persistent slot gets no-object evidence but no box regression."""
        criterion = TrackingSetCriterion(
            num_classes=1,
            matcher=_MatcherStub(),
            weight_dict={},
            focal_alpha=0.25,
            losses=["labels", "boxes"],
        )
        empty = torch.empty(0, dtype=torch.int64)
        assignment = SequenceAssignment(
            continuing_indices=(torch.tensor([0]), torch.tensor([0])),
            discovery_indices=(empty, empty),
            absent_query_indices=torch.tensor([1]),
            slot_track_ids=(10, 20),
        )
        targets = [{"labels": torch.tensor([0]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]

        def evaluate(absent_logit: float) -> dict[str, torch.Tensor]:
            outputs = {
                "pred_logits": torch.tensor([[[10.0], [absent_logit]]]),
                "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.0, 0.0, 1.0, 1.0]]]),
            }
            return criterion(outputs, targets, assignments=[assignment])

        negative_absent = evaluate(-10.0)
        positive_absent = evaluate(10.0)

        assert negative_absent["loss_ce"] < positive_absent["loss_ce"]
        assert positive_absent["loss_bbox"].item() == pytest.approx(0.0)
        assert positive_absent["loss_giou"].item() == pytest.approx(0.0)
