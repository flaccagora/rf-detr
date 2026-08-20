# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Smoke tests: Trainer(fast_dev_run=2).fit(module, datamodule) — T7.

Verifies that the PTL training loop runs end-to-end without error for both detection and segmentation configurations.
All heavy operations (build_model, build_criterion_and_postprocessors, build_dataset, get_param_dict) are patched so no
real dataset or GPU is required.

Chapter 1 gate: these must pass before Chapter 2 begins.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from pytorch_lightning import Trainer

from rfdetr.config import RFDETRBaseConfig, SegmentationTrainConfig, TrainConfig
from rfdetr.detr import RFDETR
from rfdetr.models.tracking import TrackingFrameOutput, TrackQueryState
from rfdetr.training import build_trainer
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule

from .helpers import (
    _fake_postprocess,
    _FakeCriterion,
    _FakeDataset,
    _FakeDatasetWithMasks,
    _FakePostProcess,
    _make_param_dicts,
    _TinyModel,
)

# ---------------------------------------------------------------------------
# Private helpers unique to smoke tests
# ---------------------------------------------------------------------------


def _make_trainer() -> Trainer:
    """Create a Trainer configured for minimal smoke-test runs."""
    return Trainer(
        fast_dev_run=2,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )


class _TinyRecurrentModel(_TinyModel):
    """Tiny trainable tracking model that records every recurrent input."""

    def __init__(self, model_config: RFDETRBaseConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.prior_states: list[TrackQueryState] = []

    def forward_tracking(self, samples, prior_state=None):
        """Return candidate state derived from the tiny trainable parameter."""
        self.prior_states.append(prior_state)
        batch_size = samples.tensors.shape[0]
        features = self.dummy.expand(batch_size, self.model_config.num_queries, self.model_config.hidden_dim)
        boxes = (self.dummy + 0.5).expand(batch_size, self.model_config.num_queries, 4)
        logits = self.dummy.expand(
            batch_size,
            self.model_config.num_queries,
            self.model_config.num_classes + 1,
        )
        return TrackingFrameOutput(
            pred_logits=logits,
            pred_boxes=boxes,
            candidate_state=TrackQueryState(features, boxes, torch.ones_like(prior_state.active_mask)),
            input_active_mask=prior_state.active_mask,
        )


class _TinyTrackingCriterion(_FakeCriterion):
    """Criterion supporting identity-aware recurrent assignment."""

    matcher = staticmethod(lambda outputs, targets: [(torch.tensor([0]), torch.tensor([0]))])

    def __call__(self, outputs, targets, assignments=None, num_boxes=None):
        """Return a differentiable scalar loss for one recurrent frame."""
        return {"loss_ce": outputs["pred_logits"].mean()}


def _write_video_split(root: Path, split: str) -> None:
    """Write one two-frame COCO-video clip for a public training smoke test."""
    split_dir = root / split
    split_dir.mkdir(parents=True)
    images = []
    annotations = []
    for frame_index in range(2):
        file_name = f"frame-{frame_index}.png"
        Image.new("RGB", (16, 16), color=(frame_index * 20, 0, 0)).save(split_dir / file_name)
        images.append(
            {
                "id": frame_index,
                "file_name": file_name,
                "width": 16,
                "height": 16,
                "sequence_id": "tiny",
                "frame_index": frame_index,
            }
        )
        annotations.append(
            {
                "id": frame_index,
                "image_id": frame_index,
                "category_id": 0,
                "bbox": [2, 2, 4, 4],
                "track_id": 7,
                "identity_provenance": "human",
            }
        )
    (split_dir / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 0, "name": "object"}]}),
        encoding="utf-8",
    )


def test_public_train_runs_one_recurrent_train_and_validation_batch(tmp_path: Path) -> None:
    """``RFDETR.train(dataset_file='video')`` should exercise a complete recurrent fit."""
    dataset_root = tmp_path / "video"
    _write_video_split(dataset_root, "train")
    _write_video_split(dataset_root, "val")
    output_dir = tmp_path / "output"
    model_config = RFDETRBaseConfig(
        pretrain_weights=str(tmp_path / "image-checkpoint.pth"),
        device="cpu",
        num_classes=1,
        group_detr=1,
        tracking={"enabled": True},
        class_schema={
            "foreground_classes": [{"class_id": 0, "name": "object", "external_category_id": 0}],
            "background_logit_index": 1,
        },
    )
    train_config = TrainConfig(
        dataset_file="video",
        dataset_dir=str(dataset_root),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=1,
        grad_accum_steps=1,
        num_workers=0,
        multi_scale=False,
        expanded_scales=False,
        tensorboard=False,
        use_ema=False,
        tracking={"clip_length": 2, "clip_stride": 1},
    )
    public_model = MagicMock()
    public_model.model_config = model_config
    public_model.model = SimpleNamespace(model=object(), class_names=["object"], args=SimpleNamespace())
    public_model.get_train_config.return_value = train_config
    tiny_model = _TinyRecurrentModel(model_config)
    criterion = _TinyTrackingCriterion()

    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model),
        patch("rfdetr.training.module_model.load_pretrain_weights") as load_pretrain_weights,
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(criterion, _FakePostProcess()),
        ),
        patch("rfdetr.training.module_model.get_param_dict", side_effect=lambda args, model: _make_param_dicts(model)),
        patch("rfdetr.training.build_trainer", return_value=_make_trainer()),
    ):
        RFDETR.train(public_model, dataset_file="video")

    load_pretrain_weights.assert_called_once_with(tiny_model, model_config)
    assert len(tiny_model.prior_states) >= 4
    assert all(not state.active_mask.any() for state in tiny_model.prior_states[::2])
    assert all(state.active_mask[0, 0] for state in tiny_model.prior_states[1::2])
    saved = json.loads((output_dir / "training_config.json").read_text(encoding="utf-8"))
    assert saved["train_config"]["dataset_file"] == "video"
    assert saved["train_config"]["tracking"]["clip_length"] == 2


# ---------------------------------------------------------------------------
# Smoke test classes
# ---------------------------------------------------------------------------


class TestDetectionSmoke:
    """Trainer(fast_dev_run=2).fit() must complete without error for detection."""

    def test_fit_runs_without_error(self, base_model_config, base_train_config):
        """Full PTL fit loop runs 2 train + 2 val batches without raising."""
        mc = base_model_config()
        tc = base_train_config()

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            _make_trainer().fit(module, datamodule)

    def test_training_step_called_expected_times(self, base_model_config, base_train_config):
        """fast_dev_run=2 must run exactly 2 training steps."""
        mc = base_model_config()
        tc = base_train_config()

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)

            original_training_step = module.training_step
            call_count = []

            def _counting_training_step(batch, batch_idx):
                call_count.append(1)
                return original_training_step(batch, batch_idx)

            module.training_step = _counting_training_step
            _make_trainer().fit(module, datamodule)

        assert sum(call_count) == 2

    def test_val_step_called_expected_times(self, base_model_config, base_train_config):
        """fast_dev_run=2 must run exactly 2 validation steps."""
        mc = base_model_config()
        tc = base_train_config()

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)

            original_validation_step = module.validation_step
            call_count = []

            def _counting_val_step(batch, batch_idx):
                call_count.append(1)
                return original_validation_step(batch, batch_idx)

            module.validation_step = _counting_val_step
            _make_trainer().fit(module, datamodule)

        assert sum(call_count) == 2

    def test_loss_decreases_or_is_finite(self, base_model_config, base_train_config):
        """Training loss must be finite (not NaN/inf) for the run to be valid."""
        mc = base_model_config()
        tc = base_train_config()

        tiny_model = _TinyModel()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        losses = []

        def _recording_criterion(outputs, targets, num_boxes=None):
            dummy = outputs.get("dummy", torch.zeros(1))
            denominator = fake_criterion.num_boxes_for_targets(outputs, targets) if num_boxes is None else num_boxes
            loss = dummy.mean() / denominator
            losses.append(loss.detach().item())
            return {"loss_ce": loss}

        fake_criterion = MagicMock(side_effect=_recording_criterion)
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        fake_criterion.num_boxes_for_targets.return_value = torch.tensor(1.0)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            _make_trainer().fit(module, datamodule)

        assert len(losses) > 0
        assert all(torch.isfinite(torch.tensor(v)) for v in losses)


class TestSegmentationSmoke:
    """Trainer(fast_dev_run=2).fit() must complete without error for segmentation."""

    def test_fit_runs_without_error(self, base_model_config, seg_train_config):
        """Full PTL fit loop runs 2 train + 2 val batches without raising."""
        mc = base_model_config(segmentation_head=True)
        tc = seg_train_config()

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_fake_postprocess)
        fake_dataset = _FakeDatasetWithMasks(length=20)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            _make_trainer().fit(module, datamodule)

    def test_segmentation_config_accepted(self, base_model_config, seg_train_config):
        """SegmentationTrainConfig must be accepted by both module and datamodule."""
        mc = base_model_config(segmentation_head=True)
        tc = seg_train_config()

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(_FakeCriterion(), MagicMock(side_effect=_fake_postprocess)),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDatasetWithMasks()),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)

            assert isinstance(module.train_config, SegmentationTrainConfig)
            assert isinstance(datamodule.train_config, SegmentationTrainConfig)


class TestBuildTrainerSmoke:
    """Smoke tests for the ``build_trainer()`` public factory.

    Verifies that the full callback stack wired by ``build_trainer`` runs end-to-end with ``fast_dev_run``, using mocked
    internals so no real dataset or GPU is required.
    """

    def test_fit_via_build_trainer(self, base_model_config, base_train_config):
        """build_trainer() + trainer.fit(module, datamodule=datamodule) must not raise."""
        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(_FakeCriterion(), MagicMock(side_effect=_fake_postprocess)),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=20)),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            trainer = build_trainer(tc, mc, accelerator="cpu", fast_dev_run=2)
            trainer.fit(module, datamodule=datamodule)


class _DDPModule(RFDETRModelModule):
    """RFDETRModelModule subclass for ddp_spawn smoke tests.

    Overrides ``configure_optimizers`` so ``get_param_dict`` is never called in child processes.  ``ddp_spawn`` forks
    child processes that unpack a pickled copy of this module; patches applied in the parent process are not visible in
    children, so the real ``get_param_dict`` would be called and would fail on ``_TinyModel`` (no ``.backbone``
    attribute).

    Must be defined at module level so ``pickle`` can look up the class by qualified name when deserialising in the
    child process.
    """

    def configure_optimizers(self):
        """Minimal single-group AdamW — bypasses get_param_dict."""
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


class _MultiScaleCheckDDPModule(RFDETRModelModule):
    """DDP-safe module that asserts on_train_batch_start mutation reaches training_step.

    With multi_scale=True and _FakeDataset's 32×32 images, on_train_batch_start interpolates samples.tensors to a multi-
    scale resolution (≥392 for RFDETRBaseConfig resolution=560).  This module raises AssertionError in training_step if
    the tensor height is still 32, meaning the in-place NestedTensor mutation did not propagate through the PTL batch-
    hook chain.

    Must be defined at module level so pickle can look up the class by qualified name when ddp_spawn deserialises it in
    the child process.

    Regression guard for issue #952.
    """

    def configure_optimizers(self):
        """Minimal single-group AdamW — bypasses get_param_dict."""
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

    def training_step(self, batch, batch_idx):
        """Assert resize from on_train_batch_start propagated before calling super."""
        samples, _ = batch
        h = samples.tensors.shape[2]
        if h == 32:
            raise AssertionError(
                f"training_step received images at original 32-px height (h={h}). "
                "on_train_batch_start's in-place NestedTensor mutation did not "
                "propagate through the PTL hook chain. "
                "Regression of issue #952: resize bypass in DDP batch-hook chain."
            )
        return super().training_step(batch, batch_idx)


# ---------------------------------------------------------------------------
# Multi-scale hook propagation tests (issue #952 regression)
# ---------------------------------------------------------------------------


class TestMultiScaleHookPropagation:
    """on_train_batch_start resize must propagate to training_step via NestedTensor mutation.

    _FakeDataset emits 32×32 images.  With multi_scale=True and RFDETRBaseConfig(resolution=560, patch_size=14,
    num_windows=4) the computed scales start at 392, so none equal 32.  _MultiScaleCheckDDPModule raises AssertionError
    in training_step if h==32, making trainer.fit() fail when the in-place mutation does not propagate.
    """

    def test_mutation_persists_to_training_step(self, base_model_config, base_train_config):
        """Single-process: training_step must see resized tensors, not original 32×32."""
        mc = base_model_config()
        tc = base_train_config(multi_scale=True, use_ema=False, run_test=False)
        fake_dataset = _FakeDataset(length=20)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(_FakeCriterion(), _FakePostProcess()),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
        ):
            module = _MultiScaleCheckDDPModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            trainer = build_trainer(tc, mc, accelerator="cpu", fast_dev_run=2)
            trainer.fit(module, datamodule=datamodule)


# Windows CI currently cannot run this smoke test because gloo DDP spawn fails
# with makeDeviceForHostname unsupported-device errors.
@pytest.mark.skipif(sys.platform == "win32", reason="gloo DDP spawn unsupported on Windows CI")
def test_ddp_spawn_fit_runs_without_error(base_model_config, base_train_config):
    """ddp_spawn with 2 CPU workers must run fast_dev_run=2 without error.

    ``ddp_spawn`` forks child processes, so all objects passed to ``trainer.fit()`` must be picklable.  ``MagicMock`` is
    NOT picklable; this test uses ``_FakePostProcess``, plain dataset instances, and ``_DDPModule`` (module-level class)
    instead.
    """
    mc = base_model_config()
    tc = base_train_config(use_ema=False, run_test=False, devices=2, strategy="ddp_spawn")

    fake_dataset = _FakeDataset(length=20)

    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), _FakePostProcess()),
        ),
    ):
        module = _DDPModule(mc, tc)

    datamodule = RFDETRDataModule(mc, tc)
    # Pre-set datasets: build_dataset mock doesn't survive the spawn boundary.
    datamodule._dataset_train = fake_dataset
    datamodule._dataset_val = fake_dataset

    trainer = build_trainer(tc, mc, accelerator="cpu", fast_dev_run=2)
    trainer.fit(module, datamodule=datamodule)


@pytest.mark.skipif(sys.platform == "win32", reason="gloo DDP spawn unsupported on Windows CI")
def test_ddp_spawn_multi_scale_mutation_propagates(base_model_config, base_train_config):
    """ddp_spawn with multi_scale=True must propagate on_train_batch_start resize to training_step.

    _MultiScaleCheckDDPModule raises AssertionError in training_step when the NestedTensor height is still 32 (original
    _FakeDataset size).  If trainer.fit() completes without error the PTL batch-hook reference chain is intact in DDP,
    i.e. the in-place mutation in on_train_batch_start is visible in training_step on both workers.

    Regression test for issue #952 on CPU DDP (non-Windows): confirms the transforms/resize propagation is not a
    Windows-only concern.
    """
    mc = base_model_config()
    tc = base_train_config(multi_scale=True, use_ema=False, run_test=False, devices=2, strategy="ddp_spawn")

    fake_dataset = _FakeDataset(length=20)

    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), _FakePostProcess()),
        ),
    ):
        module = _MultiScaleCheckDDPModule(mc, tc)

    datamodule = RFDETRDataModule(mc, tc)
    # Pre-set datasets: build_dataset mock doesn't survive the spawn boundary.
    datamodule._dataset_train = fake_dataset
    datamodule._dataset_val = fake_dataset

    trainer = build_trainer(tc, mc, accelerator="cpu", fast_dev_run=2)
    trainer.fit(module, datamodule=datamodule)
