# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Comprehensive unit tests for RFDETRModelModule (LightningModule wrapper)."""

import random
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from torch import nn

from rfdetr.config import RFDETRBaseConfig, TrackingSessionConfig, TrainConfig
from rfdetr.models.matcher import SequenceAssignment
from rfdetr.models.tracking import TrackingFrameOutput, TrackQueryState
from rfdetr.models.weights import apply_lora, load_pretrain_weights
from rfdetr.tracking.lifecycle import TrackSlotTable
from rfdetr.training.checkpoint import source_checkpoint_hash
from rfdetr.utilities.tensors import NestedTensor

# ---------------------------------------------------------------------------
# Private helpers — used by both module-level fixtures and class-level _setup_*
# methods (which cannot inject pytest fixtures directly).
# Only define a private helper when it is called from more than one site;
# single-use logic belongs directly in the fixture body.
# ---------------------------------------------------------------------------


def _base_model_config(**overrides):
    """Return a minimal RFDETRBaseConfig with pretrain_weights disabled."""
    defaults = dict(pretrain_weights=None, device="cpu", num_classes=5)
    defaults.update(overrides)
    tracking = defaults.get("tracking")
    tracking_enabled = tracking.get("enabled") if isinstance(tracking, dict) else getattr(tracking, "enabled", False)
    if tracking_enabled:
        defaults["class_schema"] = {
            "foreground_classes": [
                {"class_id": class_id, "name": f"class-{class_id}", "external_category_id": class_id}
                for class_id in range(5)
            ],
            "background_logit_index": 5,
        }
    return RFDETRBaseConfig(**defaults)


def _base_train_config(tmp_path=None, **overrides):
    """Return a minimal TrainConfig suitable for unit tests."""
    dataset_dir = str(tmp_path / "dataset") if tmp_path else "/nonexistent/dataset"
    output_dir = str(tmp_path / "output") if tmp_path else "/nonexistent/output"
    defaults = dict(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=10,
        lr=1e-4,
        lr_encoder=1.5e-4,
        batch_size=2,
        weight_decay=1e-4,
        lr_drop=8,
        warmup_epochs=1.0,
        drop_path=0.0,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        grad_accum_steps=1,
        tensorboard=False,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _fake_model():
    """Return a MagicMock that behaves enough like an LWDETR model."""
    model = MagicMock(spec=nn.Module)
    real_param = nn.Parameter(torch.randn(4, 4))
    model.parameters.return_value = iter([real_param])
    model.named_parameters.return_value = iter([("weight", real_param)])
    model.update_drop_path = MagicMock()
    model.update_dropout = MagicMock()
    model.reinitialize_detection_head = MagicMock()
    return model


def _fake_criterion():
    """Return a MagicMock criterion with a realistic weight_dict."""
    criterion = MagicMock()
    criterion.weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion.num_boxes_for_targets.return_value = torch.tensor(1.0)
    return criterion


def _fake_postprocess():
    """Return a callable MagicMock for postprocess."""
    return MagicMock(return_value=[{"boxes": torch.zeros(1, 4), "scores": torch.ones(1), "labels": torch.zeros(1)}])


def _build_module(model_config=None, train_config=None, tmp_path=None):
    """Construct RFDETRModelModule with build_model_from_config and build_criterion_from_config mocked."""
    mc = model_config or _base_model_config()
    tc = train_config or _base_train_config(tmp_path)
    fake_model = _fake_model()
    fake_criterion = _fake_criterion()
    fake_postprocess = _fake_postprocess()
    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=fake_model),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(fake_criterion, fake_postprocess),
        ),
    ):
        from rfdetr.training.module_model import RFDETRModelModule

        module = RFDETRModelModule(mc, tc)
    return module, fake_model, fake_criterion, fake_postprocess


def test_keypoint_training_resets_gaussian_parameters_after_pretrained_load(tmp_path) -> None:
    """Keypoint finetuning should reset pretrained Gaussian precision rows after loading weights."""
    mc = _base_model_config(
        pretrain_weights="/fake/keypoint.pth",
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
    )
    tc = _base_train_config(tmp_path)
    fake_model = _fake_model()
    fake_model.reset_keypoint_gaussian_parameters = MagicMock()
    events: list[str] = []

    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=fake_model),
        patch("rfdetr.training.module_model.load_pretrain_weights") as mock_load_pretrain_weights,
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_fake_criterion(), _fake_postprocess()),
        ),
    ):
        mock_load_pretrain_weights.side_effect = lambda *_args, **_kwargs: events.append("load")
        fake_model.reset_keypoint_gaussian_parameters.side_effect = lambda: events.append("reset")

        from rfdetr.training.module_model import RFDETRModelModule

        RFDETRModelModule(mc, tc)

    mock_load_pretrain_weights.assert_called_once_with(fake_model, mc)
    fake_model.reset_keypoint_gaussian_parameters.assert_called_once_with()
    assert events == ["load", "reset"]


def _empty_assignment(num_queries: int) -> SequenceAssignment:
    """Return a ``SequenceAssignment`` with no continuing or discovery correspondences."""
    return SequenceAssignment(
        continuing_indices=(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        discovery_indices=(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
        absent_query_indices=torch.empty(0, dtype=torch.long),
        slot_track_ids=tuple(None for _ in range(num_queries)),
    )


def _make_batch(batch_size=2, channels=3, h=16, w=16):
    """Build a (NestedTensor, targets) tuple for testing."""
    tensors = torch.randn(batch_size, channels, h, w)
    mask = torch.zeros(batch_size, h, w, dtype=torch.bool)
    samples = NestedTensor(tensors, mask)
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor(i),
            "orig_size": torch.tensor([h, w]),
        }
        for i in range(batch_size)
    ]
    return samples, targets


class TestMultiScaleBatchStart:
    """on_train_batch_start multi-scale resize picks a step-deterministic scale without clobbering global RNG."""

    def _build_multi_scale_module(self, tmp_path, global_step):
        """Return a module configured for multi-scale with a stubbed trainer at the given global step."""
        tc = _base_train_config(tmp_path, multi_scale=True, do_random_resize_via_padding=False)
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        module.trainer = SimpleNamespace(global_step=global_step)
        return module

    def test_scale_choice_is_deterministic_per_step(self, tmp_path):
        """The same global step must resize the batch to the same scale regardless of batch contents."""
        module = self._build_multi_scale_module(tmp_path, global_step=7)

        batch_a = _make_batch(h=64, w=64)
        module.on_train_batch_start(batch_a, 0)
        size_a = tuple(batch_a[0].tensors.shape[-2:])

        batch_b = _make_batch(h=64, w=64)
        module.on_train_batch_start(batch_b, 0)
        size_b = tuple(batch_b[0].tensors.shape[-2:])

        assert size_a == size_b

    def test_does_not_perturb_global_rng(self, tmp_path):
        """Scale selection must use a step-local generator and leave the process-global RNG untouched."""
        module = self._build_multi_scale_module(tmp_path, global_step=3)

        random.seed(42)
        expected = [random.random() for _ in range(3)]

        random.seed(42)
        module.on_train_batch_start(_make_batch(h=64, w=64), 0)
        actual = [random.random() for _ in range(3)]

        assert actual == expected


class _ScalarLossModel(nn.Module):
    """Tiny model exposing one scalar parameter for gradient-scaling tests."""

    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))

    def forward(self, samples, targets=None):
        return {"loss_scale": self.value}


class _BoxNormalizedCriterion:
    """Criterion with controllable per-target loss numerators and box counts."""

    weight_dict = {"loss_ce": 1.0}
    supports_loss_normalizer_override: bool = True

    def num_boxes_for_targets(self, outputs, targets):
        return torch.as_tensor(
            sum(int(target["labels"].numel()) for target in targets),
            dtype=torch.float32,
            device=outputs["loss_scale"].device,
        ).clamp(min=1.0)

    def __call__(self, outputs, targets, num_boxes=None):
        denominator = self.num_boxes_for_targets(outputs, targets) if num_boxes is None else num_boxes
        numerator = outputs["loss_scale"] * sum(target["loss_numerator"] for target in targets)
        return {"loss_ce": numerator / denominator}


# ---------------------------------------------------------------------------
# Fixtures — inject common test infrastructure; prefer these over private
# helpers in test methods.  Class-level _setup_* helpers still use the private
# functions directly (they cannot inject fixtures themselves).
# ---------------------------------------------------------------------------


@pytest.fixture
def build_module(tmp_path):
    """Factory fixture — returns (module, fake_model, fake_criterion, fake_postprocess).

    build_model and build_criterion_and_postprocessors are mocked automatically. tmp_path is injected automatically so
    test methods do not need to declare it.
    """
    return lambda model_config=None, train_config=None: _build_module(model_config, train_config, tmp_path)


@pytest.fixture
def make_batch():
    """Factory fixture — call with optional batch_size/channels/h/w."""
    return _make_batch


class TestInit:
    """Tests for RFDETRModelModule.__init__ — covers attribute assignment and delegation to build_model() /
    build_criterion_and_postprocessors() when pretrain_weights is None."""

    @pytest.mark.parametrize(
        "model_config,expected_manual",
        [
            pytest.param(_base_model_config(use_grouppose_keypoints=False), False, id="detection"),
            pytest.param(_base_model_config(segmentation_head=True), False, id="segmentation"),
            pytest.param(
                _base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
                True,
                id="keypoints",
            ),
        ],
    )
    def test_optimization_mode_per_model_type(self, build_module, model_config, expected_manual):
        """Only keypoint models need manual optimization for box-normalized accumulation; detection and segmentation
        keep Lightning's automatic optimization path."""
        module, _, _, _ = build_module(model_config=model_config)

        assert module._use_manual_optimization is expected_manual
        assert module.automatic_optimization is (not expected_manual)

    def test_model_is_set(self, build_module):
        """__init__ must assign the built model to module.model."""
        module, fake_model, _, _ = build_module()
        assert module.model is fake_model

    def test_criterion_is_set(self, build_module):
        """__init__ must assign the built criterion to module.criterion."""
        module, _, fake_criterion, _ = build_module()
        assert module.criterion is fake_criterion

    def test_postprocess_is_set(self, build_module):
        """__init__ must assign the built postprocessor to module.postprocess."""
        module, _, _, fake_pp = build_module()
        assert module.postprocess is fake_pp

    def test_configs_stored(self, base_model_config, base_train_config, build_module):
        """Both model and train configs must be stored for later access."""
        mc = base_model_config()
        tc = base_train_config()
        module, _, _, _ = build_module(model_config=mc, train_config=tc)
        assert module.model_config is mc
        assert module.train_config is tc

    def test_compile_disabled_when_multi_scale_enabled(self, tmp_path):
        """torch.compile is skipped when multi_scale=True (dynamic shapes)."""
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=True)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("rfdetr.training.module_model.torch.compile") as mock_compile,
        ):
            _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        mock_compile.assert_not_called()

    def test_compile_runs_when_enabled_and_static_shapes(self, tmp_path):
        """torch.compile runs when compile=True and multi_scale=False on CUDA."""
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=False)
        with (
            patch("rfdetr.config.DEVICE", "cuda"),
            patch("rfdetr.training.module_model.torch.compile", side_effect=lambda m, **_: m) as mock_compile,
        ):
            _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        mock_compile.assert_called_once()

    @patch("rfdetr.training.module_model.torch.compile")
    @patch("rfdetr.config.DEVICE", "cuda")
    def test_compile_disabled_when_train_accelerator_is_cpu(self, _mock_compile: MagicMock, tmp_path):
        """Compile stays disabled when training is explicitly forced to CPU."""
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=False, accelerator="cpu")
        _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        _mock_compile.assert_not_called()


class TestLoadPretrainWeights:
    """Tests for _load_pretrain_weights() — covers checkpoint validation, detection-head reinitialization on class-count
    mismatch, query-embedding trimming, re-download on corruption, and class-name extraction from checkpoint
    metadata."""

    def _make_checkpoint(self, num_classes_in_ckpt=91, num_queries=300, group_detr=13):
        """Build a fake checkpoint dict."""
        total_queries = num_queries * group_detr
        return {
            "model": {
                "class_embed.weight": torch.randn(num_classes_in_ckpt, 256),
                "class_embed.bias": torch.randn(num_classes_in_ckpt),
                "refpoint_embed.weight": torch.randn(total_queries, 4),
                "query_feat.weight": torch.randn(total_queries, 256),
                "other_layer.weight": torch.randn(10, 10),
            }
        }

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_loads_checkpoint_successfully(self, mock_validate, mock_torch_load, base_model_config, build_module):
        """A valid checkpoint must be validated, loaded, and applied to the model."""
        mc = base_model_config(num_classes=90)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})
        load_pretrain_weights(module.model, module.model_config)

        mock_validate.assert_called_once_with("/fake/weights.pth", strict=False)
        module.model.load_state_dict.assert_called_once()

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_class_count_mismatch_triggers_reinitialize(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Detection head is expanded to checkpoint size, then trimmed back to config size."""
        mc = base_model_config(num_classes=5)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        mock_torch_load.return_value = checkpoint

        module, fake_model, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})
        load_pretrain_weights(module.model, module.model_config)

        # First call: expand to checkpoint size so load_state_dict shapes match.
        # Second call: trim back to configured num_classes + 1 (background class).
        from unittest.mock import call

        fake_model.reinitialize_detection_head.assert_has_calls([call(91), call(6)])
        assert fake_model.reinitialize_detection_head.call_count == 2

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_class_count_match_does_not_reinitialize(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Detection head must NOT be reinitialized when class counts match."""
        mc = base_model_config(num_classes=5)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=6)
        mock_torch_load.return_value = checkpoint

        module, fake_model, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})
        load_pretrain_weights(module.model, module.model_config)

        fake_model.reinitialize_detection_head.assert_not_called()

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_query_embedding_trimmed_to_configured_count(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Oversized query embeddings in checkpoint must be trimmed to match config."""
        mc = base_model_config(num_classes=90)
        module, _, _, _ = build_module(model_config=mc)

        num_queries = getattr(module.model_config, "num_queries", 300)
        group_detr = getattr(module.model_config, "group_detr", 13)
        desired = num_queries * group_detr

        large_total = desired + 500
        checkpoint = {
            "model": {
                "class_embed.weight": torch.randn(91, 256),
                "class_embed.bias": torch.randn(91),
                "refpoint_embed.weight": torch.randn(large_total, 4),
                "query_feat.weight": torch.randn(large_total, 256),
            }
        }
        mock_torch_load.return_value = checkpoint

        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})
        load_pretrain_weights(module.model, module.model_config)

        assert checkpoint["model"]["refpoint_embed.weight"].shape[0] == desired
        assert checkpoint["model"]["query_feat.weight"].shape[0] == desired

    @patch("rfdetr.models.weights.os.path.isfile", return_value=True)
    @patch("rfdetr.models.weights.download_pretrain_weights")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_redownloads_on_load_failure(
        self, mock_validate, mock_download, mock_isfile, base_model_config, build_module
    ):
        """A corrupted checkpoint must trigger re-download and a second load attempt."""
        mc = base_model_config(num_classes=90)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})

        load_calls = [0]

        def fake_safe_load(*args, **kwargs):
            load_calls[0] += 1
            if load_calls[0] == 1:
                raise RuntimeError("corrupted file")
            return checkpoint

        # Patch at the definition site in util.io (_safe_torch_load is a deferred import in
        # weights.py so it is not a module-level name there). MD5 validation is intentionally
        # kept on the retry (validate_md5=False was removed in favour of rejecting
        # hash-mismatched files rather than silently accepting them).
        with patch("rfdetr.util.io._safe_torch_load", side_effect=fake_safe_load):
            load_pretrain_weights(module.model, module.model_config)

        redownload_calls = [c for c in mock_download.call_args_list if c.kwargs.get("redownload") is True]
        assert len(redownload_calls) >= 1
        assert load_calls[0] == 2

    @patch("rfdetr.models.weights.os.path.isfile", return_value=False)
    @patch("rfdetr.models.weights.download_pretrain_weights")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    @patch("rfdetr.models.weights.torch.load")
    def test_download_before_load_when_weights_absent(
        self, mock_torch_load, mock_validate, mock_download, mock_isfile, base_model_config, build_module
    ):
        """download_pretrain_weights must be called before torch.load so a fresh environment (e.g. Colab) downloads
        weights automatically.

        Regression test: previously download was only called as an except-block fallback, but ModelWeights.from_filename
        received the absolute path and returned None, causing a silent no-op and a FileNotFoundError.
        """
        mc = base_model_config(num_classes=90)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/content/rf-detr-base.pth"})
        load_pretrain_weights(module.model, module.model_config)

        # download_pretrain_weights must have been called at least once before any load
        assert mock_download.call_count >= 1
        first_call = mock_download.call_args_list[0]
        assert first_call.args[0] == "/content/rf-detr-base.pth"

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_seg_checkpoint_into_detection_model_raises(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Loading a segmentation checkpoint into a detection model must raise ValueError."""
        mc = base_model_config(num_classes=90)
        ckpt_args = SimpleNamespace(segmentation_head=True, patch_size=12)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        checkpoint["args"] = ckpt_args
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(
            update={"pretrain_weights": "/fake/weights.pth", "segmentation_head": False}
        )

        with pytest.raises(ValueError, match="segmentation head"):
            load_pretrain_weights(module.model, module.model_config)

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_detection_checkpoint_into_seg_model_raises(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Loading a detection checkpoint into a segmentation model must raise ValueError."""
        mc = base_model_config(num_classes=90)
        ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=16)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        checkpoint["args"] = ckpt_args
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(
            update={"pretrain_weights": "/fake/weights.pth", "segmentation_head": True}
        )

        with pytest.raises(ValueError, match="segmentation head"):
            load_pretrain_weights(module.model, module.model_config)

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_patch_size_mismatch_raises(self, mock_validate, mock_torch_load, base_model_config, build_module):
        """Loading a checkpoint with a different patch_size must raise ValueError."""
        mc = base_model_config(num_classes=90)
        ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=12)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        checkpoint["args"] = ckpt_args
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(
            update={"pretrain_weights": "/fake/weights.pth", "segmentation_head": False, "patch_size": 16}
        )

        with pytest.raises(ValueError, match="patch_size"):
            load_pretrain_weights(module.model, module.model_config)

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_compatible_checkpoint_does_not_raise(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """A checkpoint matching segmentation_head and patch_size must load without error."""
        mc = base_model_config(num_classes=90)
        ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=14, class_names=[])
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        checkpoint["args"] = ckpt_args
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(
            update={"pretrain_weights": "/fake/weights.pth", "segmentation_head": False, "patch_size": 14}
        )

        # Should not raise.
        load_pretrain_weights(module.model, module.model_config)


class TestApplyLora:
    """Tests for _apply_lora() — verifies that PEFT LoraConfig is constructed with the correct target modules and that
    the backbone encoder is replaced in-place with the wrapped PEFT model."""

    def _build_module_with_backbone(self, tmp_path):
        """Build module with a mock backbone that exposes backbone[0].encoder."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path)

        fake_model = MagicMock()
        fake_encoder = MagicMock()
        fake_backbone_0 = MagicMock()
        fake_backbone_0.encoder = fake_encoder
        fake_model.backbone = MagicMock()
        fake_model.backbone.__getitem__ = MagicMock(return_value=fake_backbone_0)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=fake_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(_fake_criterion(), _fake_postprocess()),
            ),
        ):
            from rfdetr.training.module_model import RFDETRModelModule

            module = RFDETRModelModule(mc, tc)

        return module, fake_model, fake_backbone_0, fake_encoder

    @patch("peft.get_peft_model")
    @patch("peft.LoraConfig")
    def test_calls_lora_config_with_correct_target_modules(self, mock_lora_cfg_class, mock_get_peft, tmp_path):
        """LoRA must target the expected attention and token projection modules."""
        module, _, _, _ = self._build_module_with_backbone(tmp_path)
        mock_get_peft.return_value = MagicMock()

        apply_lora(module.model)

        mock_lora_cfg_class.assert_called_once()
        target_modules = mock_lora_cfg_class.call_args.kwargs.get("target_modules")
        expected = ["q_proj", "v_proj", "k_proj", "qkv", "query", "key", "value", "cls_token", "register_tokens"]
        assert target_modules == expected

    @patch("peft.get_peft_model")
    @patch("peft.LoraConfig")
    def test_replaces_encoder_with_peft_model(self, mock_lora_cfg_class, mock_get_peft, tmp_path):
        """The backbone encoder must be replaced in-place with the PEFT-wrapped model."""
        module, _, fake_backbone_0, fake_encoder = self._build_module_with_backbone(tmp_path)
        peft_wrapped = MagicMock()
        mock_get_peft.return_value = peft_wrapped

        apply_lora(module.model)

        assert mock_get_peft.call_args[0][0] is fake_encoder
        assert fake_backbone_0.encoder is peft_wrapped


class TestOnFitStart:
    """Tests for on_fit_start() seeding behavior."""

    @patch("rfdetr.training.module_model.seed_everything")
    def test_seed_at_rank_zero(self, mock_seed, base_train_config, build_module):
        """Rank 0: seed_everything(seed + 0) == seed_everything(seed)."""
        tc = base_train_config(seed=7)
        module, _, _, _ = build_module(train_config=tc)

        with patch.object(type(module), "global_rank", new_callable=PropertyMock, return_value=0):
            module.on_fit_start()

        mock_seed.assert_called_once_with(7, workers=True)

    @patch("rfdetr.training.module_model.seed_everything")
    def test_seed_rank_offset(self, mock_seed, base_train_config, build_module):
        """Non-zero rank: seed_everything(seed + global_rank) must be called.

        Validates the rank-offset contract — each worker seeds with a unique value to prevent correlated data
        augmentation across DDP processes.
        """
        tc = base_train_config(seed=7)
        module, _, _, _ = build_module(train_config=tc)

        with patch.object(type(module), "global_rank", new_callable=PropertyMock, return_value=2):
            module.on_fit_start()

        mock_seed.assert_called_once_with(9, workers=True)  # 7 + 2

    @patch("rfdetr.training.module_model.seed_everything")
    def test_seed_skipped_when_none(self, mock_seed, base_train_config, build_module):
        """No seed means on_fit_start should not call seed_everything."""
        tc = base_train_config(seed=None)
        module, _, _, _ = build_module(train_config=tc)

        module.on_fit_start()

        mock_seed.assert_not_called()


class TestOnTrainBatchStart:
    """Tests for on_train_batch_start() — covers multi-scale interpolation of NestedTensor inputs and verifies
    regularization scheduling is delegated to DropPathCallback."""

    def _setup_module(
        self,
        tmp_path,
        multi_scale=False,
        do_random_resize_via_padding=False,
    ):
        tc = _base_train_config(
            tmp_path,
            multi_scale=multi_scale,
            do_random_resize_via_padding=do_random_resize_via_padding,
        )
        module, fake_model, _, _ = _build_module(train_config=tc)

        trainer = MagicMock()
        trainer.global_step = 0
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)

        return module, fake_model

    def test_drop_path_not_applied_in_module_hook(self, tmp_path):
        """Drop-path scheduling must be handled by DropPathCallback, not module hook."""
        module, fake_model = self._setup_module(tmp_path)
        module._trainer.global_step = 1

        module.on_train_batch_start(_make_batch(), batch_idx=1)

        fake_model.update_drop_path.assert_not_called()

    def test_dropout_not_applied_in_module_hook(self, tmp_path):
        """Dropout scheduling must be handled by DropPathCallback, not module hook."""
        module, fake_model = self._setup_module(tmp_path)
        module._trainer.global_step = 2

        module.on_train_batch_start(_make_batch(), batch_idx=2)

        fake_model.update_dropout.assert_not_called()

    @pytest.mark.parametrize(
        "method_name",
        [
            pytest.param("update_drop_path", id="drop-path"),
            pytest.param("update_dropout", id="dropout"),
        ],
    )
    def test_update_not_called_when_schedule_is_none(self, method_name, tmp_path):
        """Without a schedule, neither update_drop_path nor update_dropout must be called."""
        module, fake_model = self._setup_module(tmp_path)

        module.on_train_batch_start(_make_batch(), batch_idx=0)

        getattr(fake_model, method_name).assert_not_called()

    def test_multi_scale_resize_mutates_nested_tensor(self, tmp_path):
        """Multi-scale training must resize the input tensor to a square resolution."""
        module, _ = self._setup_module(tmp_path, multi_scale=True, do_random_resize_via_padding=False)
        module._trainer.global_step = 0
        samples, targets = _make_batch(batch_size=2, h=16, w=16)

        module.on_train_batch_start((samples, targets), batch_idx=0)

        new_h, new_w = samples.tensors.shape[2], samples.tensors.shape[3]
        assert new_h == new_w, "Multi-scale should produce square outputs"

    def test_multi_scale_skipped_when_random_resize_via_padding(self, tmp_path):
        """Padding-based resize takes precedence, so multi-scale must be a no-op."""
        module, _ = self._setup_module(tmp_path, multi_scale=True, do_random_resize_via_padding=True)
        samples, targets = _make_batch(batch_size=2, h=16, w=16)
        original_shape = samples.tensors.shape

        module.on_train_batch_start((samples, targets), batch_idx=0)

        assert samples.tensors.shape == original_shape


class TestTrainingStep:
    """Tests for training_step() — covers weighted loss aggregation, per-loss logging under the train/ prefix, prog_bar
    visibility, scalar tensor output, and that losses absent from weight_dict are excluded from the total."""

    def _run_step(self, tmp_path, loss_dict=None, weight_dict=None, accumulate_grad_batches=1, model_config=None):
        module, fake_model, fake_criterion, _ = _build_module(
            model_config=model_config,
            train_config=_base_train_config(tmp_path, grad_accum_steps=accumulate_grad_batches),
            tmp_path=tmp_path,
        )
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = loss_dict or {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = weight_dict or {"loss_ce": 1.0}
        module.log = MagicMock()
        module.log_dict = MagicMock()
        # Provide a real optimizer so param_groups carries a real "lr" key.
        real_param = nn.Parameter(torch.randn(4))
        real_optimizer = torch.optim.SGD([real_param], lr=1e-3)
        module.optimizers = MagicMock(return_value=real_optimizer)
        module.manual_backward = MagicMock()
        module.lr_schedulers = MagicMock(return_value=None)
        trainer = MagicMock()
        trainer.accumulate_grad_batches = 1
        trainer.num_training_batches = 1
        trainer.gradient_clip_val = 0.0
        trainer.gradient_clip_algorithm = "norm"
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        return module, samples, targets, fake_model, fake_criterion

    def test_returns_weighted_loss_sum(self, tmp_path):
        """Total loss must equal the sum of each loss multiplied by its weight."""
        loss_dict = {"loss_ce": torch.tensor(1.0), "loss_bbox": torch.tensor(2.0), "loss_giou": torch.tensor(3.0)}
        weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(1.0 + 10.0 + 6.0)

    def test_loss_backward_uses_box_normalizer_contract(self, tmp_path):
        """Backward loss for keypoint models is scaled by the criterion box normalizer (manual optimization owns
        accumulation), not by Lightning's ``accumulate_grad_batches``."""
        loss_dict = {"loss_ce": torch.tensor(4.0)}
        weight_dict = {"loss_ce": 1.0}
        keypoint_config = _base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17])
        module, samples, targets, _, _ = self._run_step(
            tmp_path,
            loss_dict,
            weight_dict,
            accumulate_grad_batches=4,
            model_config=keypoint_config,
        )
        module.criterion.num_boxes_for_targets.return_value = torch.tensor(4.0)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(1.0)
        backward_loss = module.manual_backward.call_args.args[0]
        assert backward_loss.item() == pytest.approx(1.0)

    def test_detection_loss_uses_lightning_grad_accum_scaling(self, tmp_path):
        """Detection (automatic optimization) divides loss by ``trainer.accumulate_grad_batches`` so the returned loss
        matches the legacy non-manual training path."""
        loss_dict = {"loss_ce": torch.tensor(4.0)}
        weight_dict = {"loss_ce": 1.0}
        module, samples, targets, _, _ = self._run_step(
            tmp_path,
            loss_dict,
            weight_dict,
            accumulate_grad_batches=1,
        )
        module._trainer.accumulate_grad_batches = 4

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(1.0)
        module.manual_backward.assert_not_called()

    def _make_keypoint_module(self, tmp_path, grad_accum_steps, num_training_batches):
        """Build a keypoint module wired with ``_ScalarLossModel`` and ``_BoxNormalizedCriterion`` for accum tests."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            train_config=_base_train_config(tmp_path, grad_accum_steps=grad_accum_steps),
            tmp_path=tmp_path,
        )
        model = _ScalarLossModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        module.model = model
        module.criterion = _BoxNormalizedCriterion()
        module.postprocess = MagicMock()
        module.log = MagicMock()
        module.log_dict = MagicMock()
        module.optimizers = MagicMock(return_value=optimizer)
        module.manual_backward = lambda loss: loss.backward()
        module.lr_schedulers = MagicMock(return_value=None)
        trainer = MagicMock()
        trainer.accumulate_grad_batches = 1
        trainer.num_training_batches = num_training_batches
        trainer.gradient_clip_val = 0.0
        trainer.gradient_clip_algorithm = "norm"
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        return module, model

    @pytest.mark.parametrize(
        "grad_accum_steps,box_counts,loss_numerators,expected_value",
        [
            pytest.param(1, (4,), (8.0,), -2.0, id="ga1-single-microbatch"),
            pytest.param(2, (2, 6), (10.0, 6.0), -2.0, id="ga2-balanced"),
            pytest.param(3, (2, 4, 6), (4.0, 8.0, 12.0), -2.0, id="ga3-balanced"),
            pytest.param(4, (1, 1, 1, 1), (2.0, 2.0, 2.0, 2.0), -2.0, id="ga4-uniform"),
            pytest.param(2, (1, 99), (1.0, 99.0), -1.0, id="ga2-skewed-1-vs-99"),
            pytest.param(4, (1, 1, 1, 97), (1.0, 1.0, 1.0, 97.0), -1.0, id="ga4-skewed-1-1-1-97"),
        ],
    )
    def test_box_normalized_accumulation_matches_large_effective_batch(
        self, tmp_path, grad_accum_steps, box_counts, loss_numerators, expected_value
    ):
        """Accumulated gradients across ``grad_accum_steps`` microbatches must equal a single large batch normalized by
        total boxes, regardless of how lopsided the per-microbatch box counts are."""
        large_module, large_model = self._make_keypoint_module(tmp_path, grad_accum_steps=1, num_training_batches=1)
        accum_module, accum_model = self._make_keypoint_module(
            tmp_path, grad_accum_steps=grad_accum_steps, num_training_batches=grad_accum_steps
        )

        microbatch_targets = [
            {
                "labels": torch.ones(box_count, dtype=torch.int64),
                "loss_numerator": torch.tensor(loss_numerator),
                "orig_size": torch.tensor([16, 16]),
            }
            for box_count, loss_numerator in zip(box_counts, loss_numerators, strict=True)
        ]
        samples, _ = _make_batch(batch_size=2)

        large_module.training_step((samples, microbatch_targets), batch_idx=0)
        for batch_idx, target in enumerate(microbatch_targets):
            accum_module.training_step((samples, [target]), batch_idx=batch_idx)

        torch.testing.assert_close(accum_model.value, large_model.value)
        assert large_model.value.item() == pytest.approx(expected_value)

    def test_logs_live_train_loss_to_progress_bar(self, tmp_path):
        """Aggregate training loss must be logged every step as a progress-only metric."""
        module, samples, targets, _, _ = self._run_step(tmp_path)

        module.training_step((samples, targets), batch_idx=0)

        progress_loss_calls = [c for c in module.log.call_args_list if c[0][0] == "loss"]
        assert len(progress_loss_calls) == 1
        assert progress_loss_calls[0].kwargs.get("prog_bar") is True
        assert progress_loss_calls[0].kwargs.get("logger") is False
        assert progress_loss_calls[0].kwargs.get("on_step") is True
        assert progress_loss_calls[0].kwargs.get("on_epoch") is False

    def test_logs_learning_rate_without_progress_bar(self, tmp_path):
        """Current learning rate should be logged without occupying progress-bar metric slots."""
        module, samples, targets, _, _ = self._run_step(tmp_path)

        module.training_step((samples, targets), batch_idx=0)

        lr_calls = [c for c in module.log.call_args_list if c[0][0] == "train/lr"]
        assert len(lr_calls) == 1
        assert lr_calls[0].kwargs.get("prog_bar") is False
        assert lr_calls[0].kwargs.get("on_step") is True
        assert lr_calls[0].kwargs.get("on_epoch") is False

    def test_logs_convergence_components_to_progress_bar(self, tmp_path):
        """Selected detection and keypoint losses should appear as compact progress-only metrics."""
        loss_dict = {
            "loss_ce": torch.tensor(0.5),
            "loss_bbox": torch.tensor(0.3),
            "loss_keypoints_l1": torch.tensor(0.4),
            "loss_keypoints_nll": torch.tensor(0.2),
        }
        weight_dict = {key: 1.0 for key in loss_dict}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict)

        module.training_step((samples, targets), batch_idx=0)

        progress_names = {c[0][0] for c in module.log.call_args_list if c.kwargs.get("prog_bar") is True}
        assert {"loss_cls", "loss_box", "kp_l1", "kp_nll"}.issubset(progress_names)

    def test_logs_individual_losses_as_dict(self, tmp_path):
        """Each component loss must be logged separately under train/ prefix."""
        loss_dict = {"loss_ce": torch.tensor(0.5), "loss_bbox": torch.tensor(0.3)}
        weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict)

        module.training_step((samples, targets), batch_idx=0)

        module.log_dict.assert_called_once()
        logged = module.log_dict.call_args[0][0]
        assert "train/loss_ce" in logged
        assert "train/loss_bbox" in logged

    def test_returns_scalar_tensor(self, tmp_path):
        """Loss must be a 0-dim tensor so Lightning can call .backward() on it."""
        module, samples, targets, _, _ = self._run_step(tmp_path)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.dim() == 0

    def test_returns_detached_predictions_when_train_metrics_enabled(self, tmp_path):
        """compute_train_metrics=True should expose detached predictions without changing the Lightning loss key."""
        tc = _base_train_config(tmp_path, compute_train_metrics=True)
        module, fake_model, fake_criterion, fake_postprocess = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        model_output = {"pred_logits": torch.randn(2, 3, requires_grad=True)}
        fake_model.return_value = model_output
        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        fake_postprocess.return_value = [{"boxes": torch.randn(1, 4, requires_grad=True)}]
        module.log = MagicMock()
        module.log_dict = MagicMock()
        real_param = nn.Parameter(torch.randn(4))
        module.optimizers = MagicMock(return_value=torch.optim.SGD([real_param], lr=1e-3))
        trainer = MagicMock()
        trainer.accumulate_grad_batches = 1
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)

        result = module.training_step((samples, targets), batch_idx=0)

        assert isinstance(result, dict)
        assert result["loss"].dim() == 0
        assert result["results"][0]["boxes"].requires_grad is False
        assert result["targets"] is targets

    def test_ignores_losses_not_in_weight_dict(self, tmp_path):
        """Losses absent from weight_dict (e.g. cardinality_error) must not affect total."""
        loss_dict = {"loss_ce": torch.tensor(1.0), "cardinality_error": torch.tensor(99.0)}
        weight_dict = {"loss_ce": 2.0}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(2.0)

    def test_train_metrics_slices_to_group0_queries(self, tmp_path):
        """compute_train_metrics postprocess must receive only group-0 queries ([:num_queries]).

        Group DETR emits group_detr×num_queries outputs in train mode. Without the slice, postprocess top-k draws from
        all groups and OKS/mAP reads ~50× below true accuracy. Assert the received pred_logits has shape (B,
        num_queries, C).
        """
        nq = 10
        group_detr = 3
        batch_size = 2
        num_classes = 5
        mc = _base_model_config(num_classes=num_classes, num_queries=nq)
        tc = _base_train_config(tmp_path, compute_train_metrics=True)
        module, fake_model, fake_criterion, _ = _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)

        full_logits = torch.randn(batch_size, group_detr * nq, num_classes)
        model_output = {
            "pred_logits": full_logits,
            "pred_boxes": torch.randn(batch_size, group_detr * nq, 4),
        }
        fake_model.return_value = model_output
        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}

        received: dict = {}

        def capture_postprocess(outputs, orig_sizes):
            received.update(outputs)
            return [
                {"boxes": torch.zeros(nq, 4), "scores": torch.ones(nq), "labels": torch.zeros(nq, dtype=torch.long)}
            ]

        module.postprocess = capture_postprocess
        module.log = MagicMock()
        module.log_dict = MagicMock()
        real_param = nn.Parameter(torch.randn(4))
        module.optimizers = MagicMock(return_value=torch.optim.SGD([real_param], lr=1e-3))
        trainer = MagicMock()
        trainer.accumulate_grad_batches = 1
        trainer.num_training_batches = 1
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        samples, targets = _make_batch(batch_size=batch_size)

        module.training_step((samples, targets), batch_idx=0)

        assert "pred_logits" in received
        assert received["pred_logits"].shape == (batch_size, nq, num_classes)
        torch.testing.assert_close(received["pred_logits"], full_logits[:, :nq])

    def test_train_metrics_skips_dict_pred_masks(self, tmp_path):
        """Dict-valued pred_masks (sparse_forward in train mode) must not crash training_step.

        In segmentation train mode lwdetr uses sparse_forward which returns pred_masks as a dict. PostProcess cannot
        handle a dict — it calls .shape[0] on it. The fix filters out non-tensor values so postprocess receives
        pred_masks=None (box path).
        """
        tc = _base_train_config(tmp_path, compute_train_metrics=True)
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)

        model_output = {
            "pred_logits": torch.randn(2, 10, 5),
            "pred_boxes": torch.randn(2, 10, 4),
            "pred_masks": {"spatial_features": torch.randn(2, 256, 8, 8), "query_features": torch.randn(2, 10, 256)},
        }
        fake_model.return_value = model_output
        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}

        received: dict = {}

        def capture_postprocess(outputs, orig_sizes):
            received.update(outputs)
            return [{"boxes": torch.zeros(1, 4), "scores": torch.ones(1), "labels": torch.zeros(1, dtype=torch.long)}]

        module.postprocess = capture_postprocess
        module.log = MagicMock()
        module.log_dict = MagicMock()
        real_param = nn.Parameter(torch.randn(4))
        module.optimizers = MagicMock(return_value=torch.optim.SGD([real_param], lr=1e-3))
        trainer = MagicMock()
        trainer.accumulate_grad_batches = 1
        trainer.num_training_batches = 1
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        samples, targets = _make_batch()

        module.training_step((samples, targets), batch_idx=0)

        assert "pred_masks" not in received


class TestShouldStepOptimizer:
    """Tests for ``_should_step_optimizer`` — covers the modulo path, the end-of-epoch fallback, and the iterable /
    infinite dataset case where ``trainer.num_training_batches`` is ``float('inf')``."""

    def _make_module_with_trainer(self, tmp_path, grad_accum_steps, num_training_batches):
        """Build a module with a stub trainer exposing ``num_training_batches`` for the test scenario."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            train_config=_base_train_config(tmp_path, grad_accum_steps=grad_accum_steps),
            tmp_path=tmp_path,
        )
        trainer = MagicMock()
        trainer.num_training_batches = num_training_batches
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        return module

    @pytest.mark.parametrize(
        "grad_accum_steps,num_training_batches,batch_idx,expected",
        [
            pytest.param(1, 10, 0, True, id="ga1-bidx0-steps-every-batch"),
            pytest.param(1, 10, 9, True, id="ga1-bidx9-steps-every-batch"),
            pytest.param(2, 10, 0, False, id="ga2-bidx0-mid-window"),
            pytest.param(2, 10, 1, True, id="ga2-bidx1-closes-window"),
            pytest.param(2, 10, 2, False, id="ga2-bidx2-opens-new-window"),
            pytest.param(2, 10, 9, True, id="ga2-bidx9-closes-final-window"),
            pytest.param(4, 10, 7, True, id="ga4-bidx7-closes-second-window"),
            pytest.param(4, 10, 8, False, id="ga4-bidx8-opens-partial-window"),
            pytest.param(4, 10, 9, True, id="ga4-bidx9-final-batch-flushes-partial"),
            pytest.param(4, 11, 8, False, id="ga4-bidx8-of-11-mid-window"),
            pytest.param(4, 11, 10, True, id="ga4-bidx10-final-batch-flushes-partial"),
        ],
    )
    def test_finite_dataset_steps_at_window_close_and_epoch_end(
        self, tmp_path, grad_accum_steps, num_training_batches, batch_idx, expected
    ):
        """Optimizer steps when the accumulation window closes or when the epoch ends with a partial window."""
        module = self._make_module_with_trainer(tmp_path, grad_accum_steps, num_training_batches)

        assert module._should_step_optimizer(batch_idx) is expected

    @pytest.mark.parametrize(
        "grad_accum_steps,batch_idx,expected",
        [
            pytest.param(2, 0, False, id="ga2-bidx0-mid-window"),
            pytest.param(2, 1, True, id="ga2-bidx1-closes-window"),
            pytest.param(4, 2, False, id="ga4-bidx2-mid-window"),
            pytest.param(4, 3, True, id="ga4-bidx3-closes-window"),
        ],
    )
    def test_infinite_dataset_uses_modulo_only(self, tmp_path, grad_accum_steps, batch_idx, expected):
        """Iterable datasets report ``num_training_batches=float('inf')``; only the modulo path can close the window."""
        module = self._make_module_with_trainer(tmp_path, grad_accum_steps, float("inf"))

        assert module._should_step_optimizer(batch_idx) is expected

    def test_none_num_training_batches_uses_modulo_only(self, tmp_path):
        """If trainer.num_training_batches is None (very early in fit), only the modulo path can trigger a step."""
        module = self._make_module_with_trainer(tmp_path, grad_accum_steps=2, num_training_batches=None)

        assert module._should_step_optimizer(batch_idx=0) is False
        assert module._should_step_optimizer(batch_idx=1) is True


class TestOnTrainEpochStart:
    """Tests for ``on_train_epoch_start`` — must reset the accumulated box normalizer between epochs."""

    def test_reset_clears_stale_accumulator(self, tmp_path):
        """A stale normalizer from a previous epoch must not leak into the new epoch's first microbatch."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        module._accumulated_box_normalizer = torch.tensor(42.0)

        module.on_train_epoch_start()

        assert module._accumulated_box_normalizer is None

    def test_is_noop_for_detection_module(self, tmp_path):
        """Detection models never populate _accumulated_box_normalizer; reset must leave it None."""
        module, *_ = _build_module(tmp_path=tmp_path)

        module.on_train_epoch_start()

        assert module._accumulated_box_normalizer is None

    def test_zeros_optimizer_grad_on_stale_accumulator(self, tmp_path):
        """When a partial window survived epoch end, optimizer gradients must be zeroed before reset."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        real_param = nn.Parameter(torch.randn(4))
        real_param.grad = torch.ones(4)
        optimizer = torch.optim.SGD([real_param], lr=1.0)
        module.optimizers = MagicMock(return_value=optimizer)
        module._accumulated_box_normalizer = torch.tensor(7.0)

        module.on_train_epoch_start()

        assert module._accumulated_box_normalizer is None
        assert real_param.grad is None or real_param.grad.abs().sum().item() == pytest.approx(0.0)


class TestRescaleAccumulatedGradients:
    """Direct contract tests for _rescale_accumulated_gradients."""

    def test_scales_all_parameter_grads_by_factor(self, tmp_path):
        """Calling _rescale with factor 0.5 must halve every parameter's .grad tensor."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        nano_model = nn.Linear(3, 5)
        # weight: [5, 3], bias: [5]
        nano_model.weight.grad = torch.full((5, 3), 4.0)
        nano_model.bias.grad = torch.full((5,), 8.0)
        module.model = nano_model

        module._rescale_accumulated_gradients(torch.tensor(0.5))

        torch.testing.assert_close(nano_model.weight.grad, torch.full((5, 3), 2.0))
        torch.testing.assert_close(nano_model.bias.grad, torch.full((5,), 4.0))

    def test_scale_one_leaves_grads_unchanged(self, tmp_path):
        """Scale factor 1.0 must leave gradients exactly unchanged (identity)."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        nano_model = nn.Linear(2, 2)
        nano_model.weight.grad = torch.full((2, 2), 3.0)
        nano_model.bias.grad = torch.full((2,), 7.0)
        module.model = nano_model

        module._rescale_accumulated_gradients(torch.tensor(1.0))

        torch.testing.assert_close(nano_model.weight.grad, torch.full((2, 2), 3.0))
        torch.testing.assert_close(nano_model.bias.grad, torch.full((2,), 7.0))

    def test_skips_params_with_no_grad(self, tmp_path):
        """Parameters without .grad must remain None after rescaling."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        nano_model = nn.Linear(2, 2)
        # No backward pass — all grads are None
        module.model = nano_model

        module._rescale_accumulated_gradients(torch.tensor(0.5))

        assert nano_model.weight.grad is None
        assert nano_model.bias.grad is None


class TestRecurrentTrainingStep:
    """Tests for causal time-major video training."""

    def test_inference_like_state_ignores_background_scores(self, tmp_path):
        """Prediction-driven births use the same foreground-only scoring contract as inference."""
        mc = _base_model_config(tracking={"enabled": True}, group_detr=1)
        module, *_ = _build_module(model_config=mc, tmp_path=tmp_path)
        prior_state = TrackQueryState.empty(1, mc.num_queries, mc.hidden_dim)
        tables = [TrackSlotTable.empty(mc.num_queries)]
        candidate_state = TrackQueryState(
            torch.ones(1, mc.num_queries, mc.hidden_dim),
            torch.full((1, mc.num_queries, 4), 0.5),
            torch.ones(1, mc.num_queries, dtype=torch.bool),
        )
        logits = torch.full((1, mc.num_queries, mc.num_classes + 1), -10.0)
        logits[0, 0, mc.num_classes] = 10.0
        frame = TrackingFrameOutput(
            pred_logits=logits,
            pred_boxes=candidate_state.reference_boxes,
            candidate_state=candidate_state,
            input_active_mask=prior_state.active_mask,
        )
        assignment = SequenceAssignment(
            continuing_indices=(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
            discovery_indices=(torch.tensor([0]), torch.tensor([0])),
            absent_query_indices=torch.empty(0, dtype=torch.long),
            slot_track_ids=(7, *(None for _ in range(mc.num_queries - 1))),
        )

        state, track_ids, next_tables = module._commit_tracking_state(
            frame,
            [assignment],
            prior_state,
            [tuple(None for _ in range(mc.num_queries))],
            tables,
            frame_index=0,
            inference_like=True,
        )

        assert track_ids[0][0] is None
        assert not state.active_mask.any()
        assert next_tables[0].slots[0].status == "inactive"

    def test_assignment_guided_mode_ignores_prediction_confidence(self, tmp_path):
        """The assignment-guided control commits ground-truth-matched identity regardless of score, unaffected by
        the new inference-like lifecycle machinery -- it remains the matched control required by PRD US-018/US-022."""
        mc = _base_model_config(tracking={"enabled": True}, group_detr=1)
        module, *_ = _build_module(model_config=mc, tmp_path=tmp_path)
        prior_state = TrackQueryState.empty(1, mc.num_queries, mc.hidden_dim)
        candidate_state = TrackQueryState(
            torch.ones(1, mc.num_queries, mc.hidden_dim),
            torch.full((1, mc.num_queries, 4), 0.5),
            torch.ones(1, mc.num_queries, dtype=torch.bool),
        )
        logits = torch.full((1, mc.num_queries, mc.num_classes + 1), -10.0)  # every foreground score is near zero
        frame = TrackingFrameOutput(
            pred_logits=logits,
            pred_boxes=candidate_state.reference_boxes,
            candidate_state=candidate_state,
            input_active_mask=prior_state.active_mask,
        )
        assignment = SequenceAssignment(
            continuing_indices=(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
            discovery_indices=(torch.tensor([0]), torch.tensor([0])),
            absent_query_indices=torch.empty(0, dtype=torch.long),
            slot_track_ids=(7, *(None for _ in range(mc.num_queries - 1))),
        )

        state, track_ids, next_tables = module._commit_tracking_state(
            frame,
            [assignment],
            prior_state,
            [tuple(None for _ in range(mc.num_queries))],
            None,
            frame_index=0,
            inference_like=False,
        )

        assert track_ids[0][0] == 7
        assert state.active_mask[0, 0]
        assert next_tables is None


class TestPredictionDrivenLifecyclePropagation:
    """PRD US-018: inference-like state commitment must run the exact deployment lifecycle state machine
    (:func:`transition_lifecycle`), so the model's own false-positive births and false-negative continuations
    propagate into subsequent recurrent frames -- including suspension, recovery, and termination -- instead of
    being silently repaired by ground truth."""

    @staticmethod
    def _module(tmp_path, **lifecycle_overrides):
        mc = _base_model_config(tracking={"enabled": True}, group_detr=1)
        lifecycle = TrackingSessionConfig(
            activation_threshold=0.5,
            continuation_threshold=0.3,
            tentative_confirmation_hits=1,
            tentative_confirmation_window_frames=1,
            tentative_max_misses=1,
            max_missed_frames=1,
            **lifecycle_overrides,
        )
        tc = _base_train_config(
            tmp_path, tracking={"lifecycle_mode": "inference_like", "lifecycle": lifecycle}
        )
        module, *_ = _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        return module, mc

    @staticmethod
    def _frame(mc, prior_state, *, discovery_query: int | None, feature_value: float):
        """Build a single-stream frame whose only strong foreground score is ``discovery_query``."""
        logits = torch.full((1, mc.num_queries, mc.num_classes + 1), -10.0)
        if discovery_query is not None:
            logits[0, discovery_query, 0] = 10.0
        candidate_state = TrackQueryState(
            torch.full((1, mc.num_queries, mc.hidden_dim), feature_value),
            torch.full((1, mc.num_queries, 4), 0.5),
            torch.ones(1, mc.num_queries, dtype=torch.bool),
        )
        return TrackingFrameOutput(
            pred_logits=logits,
            pred_boxes=candidate_state.reference_boxes,
            candidate_state=candidate_state,
            input_active_mask=prior_state.active_mask.clone(),
        )

    def test_unmatched_discovery_births_and_weak_continuation_suspends_recovers_and_terminates(self, tmp_path):
        """A single query's foreground score alone -- with no ground truth ever involved -- drives birth,
        suspension, recovery, and eventual termination, and the suspended slot holds its last trusted tensors
        (a false negative) rather than the current frame's weak candidate."""
        module, mc = self._module(tmp_path)
        prior_state = TrackQueryState.empty(1, mc.num_queries, mc.hidden_dim)
        tables = [TrackSlotTable.empty(mc.num_queries)]
        no_assignment = [_empty_assignment(mc.num_queries)]

        # Frame 0: strong score with no matching ground truth anywhere -- an unmatched false-positive discovery.
        frame0 = self._frame(mc, prior_state, discovery_query=0, feature_value=1.0)
        state, ids, tables = module._commit_tracking_state_inference_like(
            frame0, no_assignment, prior_state, tables, frame_index=0
        )
        assert ids[0][0] is not None
        birth_id = ids[0][0]
        assert tables[0].slots[0].status == "active"
        assert state.active_mask[0, 0]
        torch.testing.assert_close(state.query_features[0, 0], torch.full((mc.hidden_dim,), 1.0))
        prior_state = state

        # Frame 1: the same slot's score collapses (a false negative) -- it must suspend, not vanish, and must
        # hold its last trusted tensors rather than adopt this frame's weak candidate.
        frame1 = self._frame(mc, prior_state, discovery_query=None, feature_value=2.0)
        state, ids, tables = module._commit_tracking_state_inference_like(
            frame1, no_assignment, prior_state, tables, frame_index=1
        )
        assert tables[0].slots[0].status == "suspended"
        assert ids[0][0] is not None
        assert state.active_mask[0, 0]
        torch.testing.assert_close(state.query_features[0, 0], torch.full((mc.hidden_dim,), 1.0))
        prior_state = state

        # Frame 2: score recovers -- the suspended slot resumes as the same identity ("recovered").
        frame2 = self._frame(mc, prior_state, discovery_query=0, feature_value=3.0)
        state, ids, tables = module._commit_tracking_state_inference_like(
            frame2, no_assignment, prior_state, tables, frame_index=2
        )
        assert tables[0].slots[0].status == "active"
        assert ids[0][0] == birth_id
        torch.testing.assert_close(state.query_features[0, 0], torch.full((mc.hidden_dim,), 3.0))
        prior_state = state

        # Frames 3-4: weak again for long enough to exceed max_missed_frames -- the track terminates and its
        # slot is recycled, freeing the decoder role for future discoveries.
        frame3 = self._frame(mc, prior_state, discovery_query=None, feature_value=4.0)
        state, ids, tables = module._commit_tracking_state_inference_like(
            frame3, no_assignment, prior_state, tables, frame_index=3
        )
        assert tables[0].slots[0].status == "suspended"
        prior_state = state

        frame4 = self._frame(mc, prior_state, discovery_query=None, feature_value=5.0)
        state, ids, tables = module._commit_tracking_state_inference_like(
            frame4, no_assignment, prior_state, tables, frame_index=4
        )
        assert tables[0].slots[0].status == "inactive"
        assert ids[0][0] is None
        assert not state.active_mask[0, 0]

    def test_ground_truth_assignment_does_not_override_inference_like_state(self, tmp_path):
        """A contradictory ground-truth assignment must have no effect on inference-like commitment: only the
        model's own foreground scores and the lifecycle state machine decide identity and state."""
        module, mc = self._module(tmp_path)
        prior_state = TrackQueryState.empty(1, mc.num_queries, mc.hidden_dim)
        tables = [TrackSlotTable.empty(mc.num_queries)]
        frame = self._frame(mc, prior_state, discovery_query=0, feature_value=1.0)
        empty_ids = [tuple(None for _ in range(mc.num_queries))]

        contradictory_assignment = SequenceAssignment(
            continuing_indices=(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
            discovery_indices=(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
            absent_query_indices=torch.empty(0, dtype=torch.long),
            slot_track_ids=(999, *(None for _ in range(mc.num_queries - 1))),
        )

        state_with_assignment, ids_with_assignment, tables_with_assignment = module._commit_tracking_state(
            frame, [contradictory_assignment], prior_state, empty_ids, tables, frame_index=0, inference_like=True
        )
        state_without_assignment, ids_without_assignment, tables_without_assignment = module._commit_tracking_state(
            frame, [], prior_state, empty_ids, tables, frame_index=0, inference_like=True
        )

        assert ids_with_assignment == ids_without_assignment
        assert ids_with_assignment[0][0] != 999
        torch.testing.assert_close(state_with_assignment.query_features, state_without_assignment.query_features)
        assert tables_with_assignment[0].slots[0].track_id == tables_without_assignment[0].slots[0].track_id

    @pytest.mark.parametrize(
        "detach_state,expected_requires_grad",
        [
            pytest.param(False, True, id="temporal-gradients"),
            pytest.param(True, False, id="detached-boundary"),
        ],
    )
    def test_carries_predicted_state_and_averages_frame_losses(self, tmp_path, detach_state, expected_requires_grad):
        """Every frame should consume the preceding prediction and contribute equally to the clip loss."""
        mc = _base_model_config(tracking={"enabled": True}, group_detr=1)
        tc = _base_train_config(
            tmp_path,
            tracking={"clip_length": 2, "detach_state_between_frames": detach_state},
        )
        module, _, criterion, _ = _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        frame_batches = (_make_batch(batch_size=1)[0], _make_batch(batch_size=1)[0])
        target_batches = tuple(
            (
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                    "labels": torch.tensor([1]),
                    "track_ids": torch.tensor([7]),
                    "orig_size": torch.tensor([16, 16]),
                },
            )
            for _ in range(2)
        )

        class CausalModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prior_states: list[TrackQueryState | None] = []

            def forward_tracking(self, samples, prior_state=None):
                self.prior_states.append(prior_state)
                value = float(len(self.prior_states))
                features = torch.full((1, mc.num_queries, mc.hidden_dim), value, requires_grad=True)
                boxes = torch.full((1, mc.num_queries, 4), 0.5, requires_grad=True)
                active_mask = (
                    torch.zeros((1, mc.num_queries), dtype=torch.bool)
                    if prior_state is None
                    else prior_state.active_mask
                )
                return TrackingFrameOutput(
                    pred_logits=torch.zeros((1, mc.num_queries, mc.num_classes + 1), requires_grad=True),
                    pred_boxes=boxes,
                    candidate_state=TrackQueryState(features, boxes, torch.ones_like(active_mask)),
                    input_active_mask=active_mask,
                )

        causal_model = CausalModel()
        module.model = causal_model
        criterion.matcher = MagicMock()
        criterion.matcher.return_value = [(torch.tensor([0]), torch.tensor([0]))]
        criterion.side_effect = [
            {"loss_ce": torch.tensor(2.0, requires_grad=True)},
            {"loss_ce": torch.tensor(4.0, requires_grad=True)},
        ]
        criterion.weight_dict = {"loss_ce": 1.0}
        module.log = MagicMock()
        module.log_dict = MagicMock()
        real_param = nn.Parameter(torch.randn(1))
        module.optimizers = MagicMock(return_value=torch.optim.SGD([real_param], lr=1e-3))
        module._trainer = SimpleNamespace(accumulate_grad_batches=1)
        type(module).trainer = property(lambda self: self._trainer)

        loss = module.training_step((frame_batches, target_batches), batch_idx=0)

        assert loss.item() == pytest.approx(3.0)
        first_prior = causal_model.prior_states[0]
        assert first_prior is not None
        assert not first_prior.active_mask.any()
        second_prior = causal_model.prior_states[1]
        assert second_prior is not None
        assert second_prior.active_mask[0, 0]
        assert second_prior.query_features.requires_grad is expected_requires_grad
        torch.testing.assert_close(second_prior.query_features[0, 0], torch.ones(mc.hidden_dim))
        assert criterion.call_count == 2
        assert isinstance(criterion.call_args_list[0].args[2][0], SequenceAssignment)


class TestBurnInAndTruncatedBackprop:
    """PRD US-020: bounded-gradient long recurrence via no-grad burn-in and truncated
    backpropagation through time (TBPTT) chunk boundaries."""

    class _CausalModel(nn.Module):
        """Records every prior state and the ambient grad mode; every frame's ``pred_boxes``
        and recurrent ``query_features`` route through one shared trainable parameter so
        gradient reaching it demonstrates a real (non-mocked) autograd path per frame."""

        def __init__(self, mc) -> None:
            super().__init__()
            self.mc = mc
            self.weight = nn.Parameter(torch.ones(1))
            self.prior_states: list[TrackQueryState | None] = []
            self.grad_enabled_per_call: list[bool] = []

        def forward_tracking(self, samples, prior_state=None):
            self.prior_states.append(prior_state)
            self.grad_enabled_per_call.append(torch.is_grad_enabled())
            value = float(len(self.prior_states))
            features = (self.weight * value).expand(1, self.mc.num_queries, self.mc.hidden_dim)
            boxes = torch.full((1, self.mc.num_queries, 4), 0.5) * self.weight
            active_mask = (
                torch.zeros((1, self.mc.num_queries), dtype=torch.bool)
                if prior_state is None
                else prior_state.active_mask
            )
            return TrackingFrameOutput(
                pred_logits=torch.zeros((1, self.mc.num_queries, self.mc.num_classes + 1)),
                pred_boxes=boxes,
                candidate_state=TrackQueryState(features, boxes, torch.ones_like(active_mask)),
                input_active_mask=active_mask,
            )

    @staticmethod
    def _clip(num_frames: int) -> tuple[tuple, tuple]:
        frame_batches = tuple(_make_batch(batch_size=1)[0] for _ in range(num_frames))
        target_batches = tuple(
            (
                {
                    "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                    "labels": torch.tensor([1]),
                    "track_ids": torch.tensor([7]),
                    "orig_size": torch.tensor([16, 16]),
                },
            )
            for _ in range(num_frames)
        )
        return frame_batches, target_batches

    def _module_with_causal_model(self, tmp_path, *, clip_length: int):
        mc = _base_model_config(tracking={"enabled": True}, group_detr=1)
        tc = _base_train_config(tmp_path, tracking={"clip_length": clip_length})
        module, _, criterion, _ = _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        causal_model = self._CausalModel(mc)
        module.model = causal_model
        criterion.matcher = MagicMock()
        criterion.matcher.return_value = [(torch.tensor([0]), torch.tensor([0]))]
        criterion.weight_dict = {"loss_ce": 1.0}
        return module, causal_model, criterion

    def test_burn_in_runs_prediction_driven_recurrence_under_no_gradient(self, tmp_path):
        """Burn-in frames must run with autograd disabled and must never reach the criterion, so the
        loss scale is unaffected by how many burn-in frames precede the supervised suffix."""
        module, causal_model, criterion = self._module_with_causal_model(tmp_path, clip_length=3)
        criterion.side_effect = [
            {"loss_ce": torch.tensor(2.0, requires_grad=True)},
            {"loss_ce": torch.tensor(4.0, requires_grad=True)},
        ]
        frame_batches, target_batches = self._clip(3)

        loss_dict, frame_outputs = module._unroll_tracking_clip(
            frame_batches, target_batches, inference_like=False, burn_in_frames=1
        )

        assert causal_model.grad_enabled_per_call == [False, True, True]
        assert criterion.call_count == 2
        assert loss_dict["loss_ce"].item() == pytest.approx(3.0)
        assert len(frame_outputs) == 3

    def test_burn_in_frame_count_does_not_change_supervised_loss_scale(self, tmp_path):
        """Loss normalization must be invariant to how long the burn-in prefix is: two clips with the
        same two supervised-frame losses but different burn-in prefix lengths (0 vs. 3, for total
        clip lengths of 2 and 5) must produce the identical mean loss."""
        no_burn_in_module, _, no_burn_in_criterion = self._module_with_causal_model(tmp_path / "no-burn-in", clip_length=2)
        no_burn_in_criterion.side_effect = [
            {"loss_ce": torch.tensor(2.0, requires_grad=True)},
            {"loss_ce": torch.tensor(4.0, requires_grad=True)},
        ]
        no_burn_in_loss_dict, _ = no_burn_in_module._unroll_tracking_clip(
            *self._clip(2), inference_like=False, burn_in_frames=0
        )

        long_burn_in_module, _, long_burn_in_criterion = self._module_with_causal_model(
            tmp_path / "long-burn-in", clip_length=5
        )
        long_burn_in_criterion.side_effect = [
            {"loss_ce": torch.tensor(2.0, requires_grad=True)},
            {"loss_ce": torch.tensor(4.0, requires_grad=True)},
        ]
        long_burn_in_loss_dict, _ = long_burn_in_module._unroll_tracking_clip(
            *self._clip(5), inference_like=False, burn_in_frames=3
        )

        assert no_burn_in_loss_dict["loss_ce"].item() == pytest.approx(3.0)
        assert long_burn_in_loss_dict["loss_ce"].item() == pytest.approx(3.0)

    def test_state_detaches_only_at_tbptt_chunk_boundaries(self, tmp_path):
        """With clip_length=4 and tbptt_chunk_frames=2, state must stay connected across the one
        interior (mid-chunk) boundary and detach after every chunk -- never mid-chunk."""
        module, causal_model, criterion = self._module_with_causal_model(tmp_path, clip_length=4)
        criterion.side_effect = [{"loss_ce": torch.tensor(float(i), requires_grad=True)} for i in range(4)]
        frame_batches, target_batches = self._clip(4)

        module._unroll_tracking_clip(frame_batches, target_batches, inference_like=False, tbptt_chunk_frames=2)

        requires_grad_by_frame = [prior.query_features.requires_grad for prior in causal_model.prior_states]
        # frame0 <- initial empty state (no grad); frame1 <- end of frame0, mid-chunk (connected);
        # frame2 <- end of frame1, chunk boundary (detached); frame3 <- end of frame2, mid-chunk (connected).
        assert requires_grad_by_frame == [False, True, False, True]

    def test_no_chunking_matches_pre_curriculum_single_graph_behavior(self, tmp_path):
        """tbptt_chunk_frames=None (the default) must keep state connected across every frame in the
        supervised suffix, exactly like the pre-curriculum unroll with no truncation at all."""
        module, causal_model, criterion = self._module_with_causal_model(tmp_path, clip_length=3)
        criterion.side_effect = [{"loss_ce": torch.tensor(float(i), requires_grad=True)} for i in range(3)]
        frame_batches, target_batches = self._clip(3)

        module._unroll_tracking_clip(frame_batches, target_batches, inference_like=False, tbptt_chunk_frames=None)

        requires_grad_by_frame = [prior.query_features.requires_grad for prior in causal_model.prior_states]
        assert requires_grad_by_frame == [False, True, True]

    def test_gradient_reaches_every_supervised_frame_across_chunk_boundaries(self, tmp_path):
        """A parameter used identically by every frame's forward pass must receive gradient
        contributions from frames in both TBPTT chunks -- detaching the recurrent *state* between
        chunks must not sever other, non-recurrent paths back to shared parameters."""
        module, causal_model, criterion = self._module_with_causal_model(tmp_path, clip_length=4)
        criterion.side_effect = lambda outputs, targets, assignments: {"loss_ce": outputs["pred_boxes"].sum()}
        frame_batches, target_batches = self._clip(4)

        loss_dict, _ = module._unroll_tracking_clip(
            frame_batches, target_batches, inference_like=False, tbptt_chunk_frames=2
        )
        loss_dict["loss_ce"].backward()

        # Every one of the 4 frames (2 per chunk) contributes 0.5 * num_queries * 4 to the summed
        # pred_boxes; the mean over 4 frames leaves each frame's per-weight derivative intact at
        # 0.5 * num_queries * 4, so a gradient scaled by only one chunk's worth of frames would fail
        # this exact check.
        expected_grad = 0.5 * causal_model.mc.num_queries * 4
        assert causal_model.weight.grad is not None
        assert causal_model.weight.grad.item() == pytest.approx(expected_grad)

    def test_no_graph_survives_the_clip_boundary(self, tmp_path):
        """A second clip processed by the same model must start from a freshly constructed empty
        state, never from the (potentially graph-carrying) state committed at the end of a prior
        clip -- so no gradient graph can survive from one clip into the next."""
        module, causal_model, criterion = self._module_with_causal_model(tmp_path, clip_length=2)
        criterion.side_effect = [{"loss_ce": torch.tensor(1.0, requires_grad=True)}] * 4

        module._unroll_tracking_clip(*self._clip(2), inference_like=False, tbptt_chunk_frames=None)
        # A second clip's first frame must see a brand-new leaf state (no grad_fn), not whatever the
        # first clip's final committed state happened to be.
        module._unroll_tracking_clip(*self._clip(2), inference_like=False, tbptt_chunk_frames=None)

        second_clip_first_prior = causal_model.prior_states[2]
        assert second_clip_first_prior.query_features.grad_fn is None
        assert second_clip_first_prior.query_features.requires_grad is False

    @pytest.mark.parametrize(
        ("clip_length", "burn_in_frames", "supervised_frames", "tbptt_chunk_frames"),
        [
            pytest.param(16, 12, 4, 4, id="stage-1"),
            pytest.param(64, 48, 16, 8, id="stage-2"),
            pytest.param(128, 96, 32, 8, id="stage-3"),
        ],
    )
    def test_prd_section_7_6_curriculum_stages_unroll_successfully(
        self, tmp_path, clip_length, burn_in_frames, supervised_frames, tbptt_chunk_frames
    ):
        """The exact PRD Section 7.6 16/64/128-frame curriculum settings must unroll end-to-end
        without error and must produce exactly ``supervised_frames`` criterion calls."""
        module, causal_model, criterion = self._module_with_causal_model(tmp_path, clip_length=clip_length)
        criterion.side_effect = [
            {"loss_ce": torch.tensor(float(i), requires_grad=True)} for i in range(supervised_frames)
        ]
        frame_batches, target_batches = self._clip(clip_length)

        loss_dict, frame_outputs = module._unroll_tracking_clip(
            frame_batches,
            target_batches,
            inference_like=False,
            burn_in_frames=burn_in_frames,
            tbptt_chunk_frames=tbptt_chunk_frames,
        )

        assert criterion.call_count == supervised_frames
        assert len(frame_outputs) == clip_length
        assert causal_model.grad_enabled_per_call[:burn_in_frames] == [False] * burn_in_frames
        assert causal_model.grad_enabled_per_call[burn_in_frames:] == [True] * supervised_frames
        assert loss_dict["loss_ce"].item() == pytest.approx(sum(range(supervised_frames)) / supervised_frames)


class TestErrorExposurePilots:
    """PRD US-019: the ``"inference_like"`` commit path can optionally force a bounded number of
    ground-truth-unmatched discoveries to activate (false-positive injection) or force active slots to look
    missed (query dropout), deterministically, so training pilots can be attributed to one mechanism at a
    time before a full curriculum run."""

    @staticmethod
    def _module(tmp_path, **tracking_overrides):
        mc = _base_model_config(tracking={"enabled": True}, group_detr=1)
        lifecycle = TrackingSessionConfig(
            activation_threshold=0.5,
            continuation_threshold=0.3,
            tentative_confirmation_hits=1,
            tentative_confirmation_window_frames=1,
            tentative_max_misses=1,
            max_missed_frames=1,
        )
        overrides = dict(lifecycle_mode="inference_like", lifecycle=lifecycle)
        overrides.update(tracking_overrides)
        tc = _base_train_config(tmp_path, tracking=overrides)
        module, *_ = _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        return module, mc

    @staticmethod
    def _frame(mc, prior_state, *, discovery_queries=(), feature_value=1.0):
        """Build a single-stream frame whose only strong foreground scores are ``discovery_queries``.

        Each query gets a distinct, non-overlapping box (unlike a single shared box) so that
        multiple simultaneous discoveries in one frame are not duplicate-suppressed against
        each other.
        """
        logits = torch.full((1, mc.num_queries, mc.num_classes + 1), -10.0)
        for query in discovery_queries:
            logits[0, query, 0] = 10.0
        index = torch.arange(mc.num_queries, dtype=torch.float32)
        cx = 0.05 + 0.01 * (index % 40)
        boxes = torch.stack(
            [cx, torch.full((mc.num_queries,), 0.5), torch.full((mc.num_queries,), 0.005), torch.full((mc.num_queries,), 0.005)],
            dim=-1,
        ).unsqueeze(0)
        candidate_state = TrackQueryState(
            torch.full((1, mc.num_queries, mc.hidden_dim), feature_value),
            boxes,
            torch.ones(1, mc.num_queries, dtype=torch.bool),
        )
        return TrackingFrameOutput(
            pred_logits=logits,
            pred_boxes=boxes,
            candidate_state=candidate_state,
            input_active_mask=prior_state.active_mask.clone(),
        )

    def test_false_positive_injection_births_unmatched_slot_despite_weak_score(self, tmp_path):
        """Injection alone -- with the raw candidate score weak everywhere -- must still be able to produce a
        birth, because it perturbs the candidate score fed into the lifecycle, not the lifecycle decision."""
        module, mc = self._module(
            tmp_path,
            false_positive_injection_enabled=True,
            false_positive_injection_probability=1.0,
            false_positive_injection_max_per_sample=1,
        )
        prior_state = TrackQueryState.empty(1, mc.num_queries, mc.hidden_dim)
        tables = [TrackSlotTable.empty(mc.num_queries)]
        frame = self._frame(mc, prior_state, discovery_queries=(), feature_value=1.0)
        no_assignment = [_empty_assignment(mc.num_queries)]
        remaining = [1]

        state, ids, tables = module._commit_tracking_state_inference_like(
            frame, no_assignment, prior_state, tables, frame_index=0, fp_injection_remaining=remaining
        )

        assert sum(value is not None for value in ids[0]) == 1
        assert remaining == [0]

    def test_false_positive_injection_excludes_ground_truth_matched_candidates(self, tmp_path):
        """A discovery query with a real ground-truth match is not an "unmatched query state" and must never be
        chosen for injection, even when every other candidate is eligible."""
        module, mc = self._module(tmp_path)
        table = TrackSlotTable.empty(mc.num_queries)
        matched_assignment = SequenceAssignment(
            continuing_indices=(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)),
            discovery_indices=(torch.tensor([0]), torch.tensor([0])),
            absent_query_indices=torch.empty(0, dtype=torch.long),
            slot_track_ids=(7, *(None for _ in range(mc.num_queries - 1))),
        )

        selected = module._select_false_positive_injection_indices(
            table, matched_assignment, frame_index=0, batch_index=0, probability=1.0, max_count=mc.num_queries
        )

        assert 0 not in selected
        assert len(selected) == mc.num_queries - 1

    def test_false_positive_injection_caps_selection_at_remaining_budget(self, tmp_path):
        """The per-sample injection budget is a hard cap on how many candidates one call may select."""
        module, mc = self._module(tmp_path)
        table = TrackSlotTable.empty(mc.num_queries)
        no_match = _empty_assignment(mc.num_queries)

        selected = module._select_false_positive_injection_indices(
            table, no_match, frame_index=0, batch_index=0, probability=1.0, max_count=2
        )

        assert len(selected) == 2

    def test_false_positive_injection_budget_is_spent_across_frames_within_one_clip(self, tmp_path):
        """The per-sample cap is cumulative across the whole clip, not reset every frame."""
        module, mc = self._module(
            tmp_path,
            false_positive_injection_enabled=True,
            false_positive_injection_probability=1.0,
            false_positive_injection_max_per_sample=2,
        )
        prior_state = TrackQueryState.empty(1, mc.num_queries, mc.hidden_dim)
        tables = [TrackSlotTable.empty(mc.num_queries)]
        no_assignment = [_empty_assignment(mc.num_queries)]
        remaining = [2]

        frame0 = self._frame(mc, prior_state, discovery_queries=(), feature_value=1.0)
        state, ids0, tables = module._commit_tracking_state_inference_like(
            frame0, no_assignment, prior_state, tables, frame_index=0, fp_injection_remaining=remaining
        )
        assert sum(value is not None for value in ids0[0]) == 2
        assert remaining == [0]

        frame1 = self._frame(mc, state, discovery_queries=(), feature_value=2.0)
        state, ids1, tables = module._commit_tracking_state_inference_like(
            frame1, no_assignment, state, tables, frame_index=1, fp_injection_remaining=remaining
        )

        # No further injections once the per-sample budget is exhausted; the two already-born
        # tracks may continue (recorded via their own score, unrelated to injection).
        assert remaining == [0]
        newly_born = sum(1 for old, new in zip(ids0[0], ids1[0]) if old is None and new is not None)
        assert newly_born == 0

    def test_false_positive_injection_is_deterministic_given_the_same_seed(self, tmp_path):
        """Repeating the exact same selection call must return the exact same candidates."""
        module, mc = self._module(tmp_path, error_exposure_seed=42)
        table = TrackSlotTable.empty(mc.num_queries)
        assignment = _empty_assignment(mc.num_queries)

        first = module._select_false_positive_injection_indices(
            table, assignment, frame_index=3, batch_index=1, probability=0.5, max_count=5
        )
        second = module._select_false_positive_injection_indices(
            table, assignment, frame_index=3, batch_index=1, probability=0.5, max_count=5
        )

        assert first == second

    def test_query_dropout_never_removes_every_active_slot(self, tmp_path):
        """A single active slot must never be dropped, even at probability 1.0, since that would remove every
        active slot in the sample."""
        module, mc = self._module(tmp_path)
        slots = list(TrackSlotTable.empty(mc.num_queries).slots)
        slots[0] = replace(
            slots[0], track_id=1, status="active", age=1, hits=1, missed_frames=0, last_reliable_frame=0, confidence=0.9, class_id=0
        )
        table = TrackSlotTable(slots=tuple(slots), next_track_id=2)

        selected = module._select_query_dropout_indices(table, frame_index=0, batch_index=0, probability=1.0)

        assert selected == []

    def test_query_dropout_at_full_probability_drops_all_but_one_active_slot(self, tmp_path):
        """With multiple active slots at probability 1.0, every candidate qualifies except the guard-spared
        survivor, so exactly one slot must remain undropped."""
        module, mc = self._module(tmp_path)
        slots = list(TrackSlotTable.empty(mc.num_queries).slots)
        for index in range(3):
            slots[index] = replace(
                slots[index],
                track_id=index + 1,
                status="active",
                age=1,
                hits=1,
                missed_frames=0,
                last_reliable_frame=0,
                confidence=0.9,
                class_id=0,
            )
        table = TrackSlotTable(slots=tuple(slots), next_track_id=4)

        selected = module._select_query_dropout_indices(table, frame_index=0, batch_index=0, probability=1.0)

        assert len(selected) == 2
        assert set(selected).issubset({0, 1, 2})

    def test_query_dropout_is_deterministic_given_the_same_seed(self, tmp_path):
        """Repeating the exact same selection call must return the exact same candidates."""
        module, mc = self._module(tmp_path, error_exposure_seed=7)
        slots = list(TrackSlotTable.empty(mc.num_queries).slots)
        for index in range(3):
            slots[index] = replace(
                slots[index],
                track_id=index + 1,
                status="active",
                age=1,
                hits=1,
                missed_frames=0,
                last_reliable_frame=0,
                confidence=0.9,
                class_id=0,
            )
        table = TrackSlotTable(slots=tuple(slots), next_track_id=4)

        first = module._select_query_dropout_indices(table, frame_index=2, batch_index=0, probability=0.5)
        second = module._select_query_dropout_indices(table, frame_index=2, batch_index=0, probability=0.5)

        assert first == second

    def test_query_dropout_suspends_one_of_two_active_slots_via_commit(self, tmp_path):
        """End-to-end: with two organically-born active slots and dropout at probability 1.0, exactly one must
        suspend this frame -- the model's own strong score is overridden by the synthetic missed detection --
        while the other remains active, honoring the never-drop-everything guard."""
        module, mc = self._module(tmp_path)
        prior_state = TrackQueryState.empty(1, mc.num_queries, mc.hidden_dim)
        tables = [TrackSlotTable.empty(mc.num_queries)]
        no_assignment = [_empty_assignment(mc.num_queries)]

        frame0 = self._frame(mc, prior_state, discovery_queries=(0, 1), feature_value=1.0)
        state, ids, tables = module._commit_tracking_state_inference_like(
            frame0, no_assignment, prior_state, tables, frame_index=0
        )
        assert tables[0].slots[0].status == "active"
        assert tables[0].slots[1].status == "active"

        module.train_config.tracking.query_dropout_enabled = True
        module.train_config.tracking.query_dropout_probability = 1.0
        frame1 = self._frame(mc, state, discovery_queries=(0, 1), feature_value=2.0)
        state, ids, tables = module._commit_tracking_state_inference_like(
            frame1, no_assignment, state, tables, frame_index=1
        )

        statuses = {tables[0].slots[0].status, tables[0].slots[1].status}
        assert statuses == {"active", "suspended"}


class TestValidationStep:
    """Tests for validation_step() — verifies output dict shape, postprocessor invocation with correct original sizes,
    and val/loss logging."""

    def _run_val_step(
        self,
        tmp_path,
        loss_dict: dict[str, torch.Tensor] | None = None,
        weight_dict: dict[str, float] | None = None,
    ):
        module, fake_model, fake_criterion, fake_pp = _build_module(tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = loss_dict or {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = weight_dict or {"loss_ce": 1.0}
        module.log = MagicMock()
        module.log_dict = MagicMock()
        result = module.validation_step((samples, targets), batch_idx=0)
        return result, fake_pp, module

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("results", id="results-key"),
            pytest.param("targets", id="targets-key"),
        ],
    )
    def test_returns_dict_with_required_key(self, key, tmp_path):
        """Output dict must contain both 'results' and 'targets' for downstream metric computation."""
        result, _, _ = self._run_val_step(tmp_path)
        assert key in result

    def test_postprocess_called_with_orig_sizes(self, tmp_path):
        """Postprocessor must receive original image sizes to rescale predictions."""
        result, fake_pp, _ = self._run_val_step(tmp_path)
        fake_pp.assert_called_once()
        orig_sizes = fake_pp.call_args[0][1]
        assert orig_sizes.shape == (2, 2)

    def test_logs_val_loss(self, tmp_path):
        """Validation loss must be logged for monitoring and early stopping."""
        _, _, module = self._run_val_step(tmp_path)
        val_loss_calls = [c for c in module.log.call_args_list if c[0][0] == "val/loss"]
        assert len(val_loss_calls) == 1

    def test_logs_val_keypoint_loss_components_once(self, tmp_path):
        """Validation should expose full keypoint losses without duplicate progress aliases."""
        loss_dict = {
            "loss_ce": torch.tensor(0.5),
            "loss_keypoints_l1": torch.tensor(0.4),
            "loss_keypoints_findable": torch.tensor(0.3),
            "loss_keypoints_visible": torch.tensor(0.2),
            "loss_keypoints_nll": torch.tensor(0.1),
        }
        weight_dict = {key: 1.0 for key in loss_dict}

        _, _, module = self._run_val_step(tmp_path, loss_dict=loss_dict, weight_dict=weight_dict)

        module.log_dict.assert_called_once()
        logged = module.log_dict.call_args.args[0]
        assert "val/loss_keypoints_l1" in logged
        assert "val/loss_keypoints_findable" in logged
        logged_names = {c[0][0] for c in module.log.call_args_list}
        assert "val/kp_l1" not in logged_names
        assert "val/kp_find" not in logged_names
        assert "val/kp_vis" not in logged_names
        assert "val/kp_nll" not in logged_names

    def test_val_detection_loss_components_are_not_relogged_as_progress_aliases(self, tmp_path):
        """Validation component losses should be logged once under canonical ``val/loss_*`` names."""
        loss_dict = {
            "loss_ce": torch.tensor(0.5),
            "loss_bbox": torch.tensor(0.3),
            "loss_giou": torch.tensor(0.2),
        }
        weight_dict = {key: 1.0 for key in loss_dict}

        _, _, module = self._run_val_step(tmp_path, loss_dict=loss_dict, weight_dict=weight_dict)

        logged_loss_names = set(module.log_dict.call_args.args[0])
        direct_log_names = {c[0][0] for c in module.log.call_args_list}
        assert "val/loss_giou" in logged_loss_names
        assert "val/loss_giou" not in direct_log_names
        assert "val/giou" not in direct_log_names

    def test_can_disable_val_loss_computation(self, tmp_path):
        """compute_val_loss=False skips criterion call and val/loss logging."""
        tc = _base_train_config(tmp_path, compute_val_loss=False)
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        module.log = MagicMock()

        result = module.validation_step((samples, targets), batch_idx=0)

        fake_criterion.assert_not_called()
        logged_keys = [c[0][0] for c in module.log.call_args_list]
        assert "val/loss" not in logged_keys
        assert "results" in result and "targets" in result


class TestTestStep:
    """Tests for test_step() — verifies output dict shape, postprocessor invocation with correct original sizes, and
    test/loss logging.

    Mirrors :class:`TestValidationStep` since both steps share the same forward+postprocess logic and differ only in the
    logged metric prefix.
    """

    def _run_test_step(self, tmp_path):
        module, fake_model, fake_criterion, fake_pp = _build_module(tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        module.log = MagicMock()
        result = module.test_step((samples, targets), batch_idx=0)
        return result, fake_pp, module

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("results", id="results-key"),
            pytest.param("targets", id="targets-key"),
        ],
    )
    def test_returns_dict_with_required_key(self, key, tmp_path):
        """Output dict must contain both 'results' and 'targets' for COCOEvalCallback."""
        result, _, _ = self._run_test_step(tmp_path)
        assert key in result

    def test_postprocess_called_with_orig_sizes(self, tmp_path):
        """Postprocessor must receive original image sizes to rescale predictions."""
        result, fake_pp, _ = self._run_test_step(tmp_path)
        fake_pp.assert_called_once()
        orig_sizes = fake_pp.call_args[0][1]
        assert orig_sizes.shape == (2, 2)

    def test_logs_test_loss(self, tmp_path):
        """Test loss must be logged under test/ prefix for monitoring."""
        _, _, module = self._run_test_step(tmp_path)
        test_loss_calls = [c for c in module.log.call_args_list if c[0][0] == "test/loss"]
        assert len(test_loss_calls) == 1

    def test_model_called_with_samples_only(self, tmp_path):
        """Test step must pass only samples (not targets) to the model forward."""
        module, fake_model, fake_criterion, _ = _build_module(tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        module.log = MagicMock()

        module.test_step((samples, targets), batch_idx=0)

        fake_model.assert_called_once_with(samples)

    def test_loss_prefix_differs_from_validation(self, tmp_path):
        """test_step must log 'test/loss', not 'val/loss', to keep metric namespaces separate."""
        _, _, module = self._run_test_step(tmp_path)
        logged_keys = [c[0][0] for c in module.log.call_args_list]
        assert "test/loss" in logged_keys
        assert "val/loss" not in logged_keys

    def test_can_disable_test_loss_computation(self, tmp_path):
        """compute_test_loss=False skips criterion call and test/loss logging."""
        tc = _base_train_config(tmp_path, compute_test_loss=False)
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        module.log = MagicMock()

        result = module.test_step((samples, targets), batch_idx=0)

        fake_criterion.assert_not_called()
        logged_keys = [c[0][0] for c in module.log.call_args_list]
        assert "test/loss" not in logged_keys
        assert "results" in result and "targets" in result


class TestConfigureOptimizers:
    """Tests for configure_optimizers() — covers required output keys, AdamW optimizer type, step-interval scheduler, LR
    lambda warmup ramp, and step-decay behaviour before and after lr_drop."""

    def _setup_module(self, tmp_path, **train_overrides):
        tc = _base_train_config(tmp_path, **train_overrides)
        module, _, _, _ = _build_module(train_config=tc)

        trainer = MagicMock()
        trainer.estimated_stepping_batches = 1000
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)

        real_param = nn.Parameter(torch.randn(4, 4))
        param_dicts = [{"params": real_param, "lr": tc.lr}]
        return module, param_dicts

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("optimizer", id="optimizer-key"),
            pytest.param("lr_scheduler", id="lr-scheduler-key"),
        ],
    )
    @patch("rfdetr.training.module_model.get_param_dict")
    def test_configure_optimizers_returns_required_key(self, mock_get_param_dict, key, tmp_path):
        """Lightning requires both 'optimizer' and 'lr_scheduler' keys in the returned config dict."""
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts

        assert key in module.configure_optimizers()

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_optimizer_is_adamw(self, mock_get_param_dict, tmp_path):
        """RF-DETR must use AdamW for its decoupled weight decay behavior."""
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts

        assert isinstance(module.configure_optimizers()["optimizer"], torch.optim.AdamW)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_scheduler_interval_is_step(self, mock_get_param_dict, tmp_path):
        """Scheduler must step per batch (not per epoch) for fine-grained warmup."""
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts

        assert module.configure_optimizers()["lr_scheduler"]["interval"] == "step"

    @pytest.mark.parametrize(
        "step, expected_behavior",
        [
            pytest.param(0, "warmup_start", id="warmup-start"),
            pytest.param(50, "warmup_mid", id="warmup-midpoint"),
        ],
    )
    @patch("rfdetr.training.module_model.get_param_dict")
    def test_lr_lambda_warmup_phase(self, mock_get_param_dict, step, expected_behavior, tmp_path):
        """LR lambda must produce a linear ramp during the warmup phase."""
        module, param_dicts = self._setup_module(tmp_path, warmup_epochs=1.0, epochs=10)
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # steps_per_epoch=100, warmup_steps=100
        expected = float(step) / float(max(1, 100))
        assert lr_lambda(step) == pytest.approx(expected)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_lr_lambda_step_decay_before_drop(self, mock_get_param_dict, tmp_path):
        """Before lr_drop epoch, the LR multiplier must remain at 1.0."""
        module, param_dicts = self._setup_module(tmp_path, warmup_epochs=0.0, epochs=10, lr_drop=8)
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # lr_drop * steps_per_epoch = 8 * 100 = 800; step 500 < 800 → factor 1.0
        assert lr_lambda(500) == pytest.approx(1.0)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_lr_lambda_step_decay_after_drop(self, mock_get_param_dict, tmp_path):
        """After lr_drop epoch, the LR multiplier must decay to 0.1."""
        module, param_dicts = self._setup_module(tmp_path, warmup_epochs=0.0, epochs=10, lr_drop=8)
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # step 900 > 800 → factor 0.1
        assert lr_lambda(900) == pytest.approx(0.1)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_lr_lambda_cosine_reads_train_config_fields(self, mock_get_param_dict, tmp_path):
        """Cosine scheduler must read lr_scheduler/lr_min_factor from TrainConfig."""
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            epochs=10,
            lr_scheduler="cosine",
            lr_min_factor=0.2,
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # At the final step, cosine schedule must end at lr_min_factor.
        assert lr_lambda(1000) == pytest.approx(0.2)

    @patch("rfdetr.training.module_model.get_param_dict")
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_fused_optimizer_disabled_when_precision_not_bf16(
        self,
        mock_cuda_available,
        mock_bf16_supported,
        mock_get_param_dict,
        tmp_path,
    ):
        """Fused AdamW must be disabled when trainer precision is not bf16-mixed.

        On Ampere+ GPUs torch.cuda.is_bf16_supported() is True even when the trainer is configured for 32-true
        precision.  The old code always enabled fused AdamW based on GPU capability alone, crashing with ``params,
        grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout`` when DDP gradient bucket views had
        non-matching strides. The fix checks ``trainer.precision`` before enabling fused.
        """
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts
        # Simulate trainer configured for full FP32 precision.
        module._trainer.precision = "32-true"

        optimizer = module.configure_optimizers()["optimizer"]

        assert not optimizer.defaults.get("fused")

    @patch("rfdetr.training.module_model.get_param_dict")
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_fused_optimizer_enabled_when_precision_is_bf16_mixed(
        self,
        mock_cuda_available,
        mock_bf16_supported,
        mock_get_param_dict,
        tmp_path,
    ):
        """Fused AdamW must be enabled when both GPU supports BF16 and trainer uses bf16-mixed.

        The fused path is beneficial (and safe) only when training precision is actually BF16: parameters, gradients,
        and optimizer state all stay in the same dtype/layout, satisfying the fused kernel requirements.
        """
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts
        # Simulate trainer configured for BF16 mixed precision.
        module._trainer.precision = "bf16-mixed"

        optimizer = module.configure_optimizers()["optimizer"]

        assert optimizer.defaults.get("fused") is True

    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=False)
    def test_fused_optimizer_disabled_when_cuda_unavailable(self, mock_cuda_available, tmp_path):
        """_use_fused_optimizer must return False when CUDA is not available, regardless of precision."""
        module, _ = self._setup_module(tmp_path)
        module._trainer.precision = "bf16-mixed"

        assert not module._use_fused_optimizer

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_total_steps_divided_by_grad_accum_for_keypoint_module(self, mock_get_param_dict, tmp_path):
        """Keypoint (manual-opt) path must divide estimated_stepping_batches by grad_accum_steps for LR scheduling.

        With microbatches=100, grad_accum_steps=4, epochs=1, warmup_epochs=0 the scheduler should span 25 optimizer
        steps (ceil(100/4)).  At step 24 (0-indexed last step) a cosine LR schedule should be nearly at lr_min_factor;
        if total_steps were mistakenly 100 the LR would still be near its peak at step 24.
        """
        import math

        grad_accum_steps = 4
        microbatches = 100
        lr_min_factor = 0.1
        tc = _base_train_config(
            tmp_path,
            grad_accum_steps=grad_accum_steps,
            warmup_epochs=0,
            epochs=1,
            lr_scheduler="cosine",
            lr_min_factor=lr_min_factor,
        )
        module, _, _, _ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            train_config=tc,
        )
        trainer = MagicMock()
        trainer.estimated_stepping_batches = microbatches
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        real_param = nn.Parameter(torch.randn(4, 4))
        mock_get_param_dict.return_value = [{"params": real_param, "lr": tc.lr}]

        result = module.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        expected_total_steps = max(1, math.ceil(microbatches / grad_accum_steps))  # 25
        # The cosine schedule reaches lr_min_factor exactly at step == total_steps (progress=1.0).
        # If total_steps were wrongly 100, lr at step 25 would still be ~0.87 (near peak).
        lr_at_decay_end = lr_lambda(expected_total_steps)
        assert lr_at_decay_end == pytest.approx(lr_min_factor, abs=1e-6)


class TestDecodedFrameLRProgress:
    """PRD Section 7.6 / US-021 AC: "Learning-rate schedules advance in decoded-frame units, not
    dataloader-step units." configure_optimizers() still computes lr_lambda in the step-equivalent
    domain; _decoded_frame_count / _decoded_frames_per_optimizer_step / _step_lr_scheduler /
    lr_scheduler_step convert real decoded-frame progress back into that domain so that
    non-curriculum runs are unaffected but curriculum/mixed-frame-count runs advance the schedule by
    their true share of decoded compute."""

    @staticmethod
    def _module_with_scheduler(tmp_path, *, estimated_stepping_batches=1000, **train_overrides):
        tc = _base_train_config(tmp_path, **train_overrides)
        module, *_ = _build_module(train_config=tc)
        trainer = MagicMock()
        trainer.estimated_stepping_batches = estimated_stepping_batches
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        real_param = nn.Parameter(torch.randn(4, 4))
        with patch(
            "rfdetr.training.module_model.get_param_dict",
            return_value=[{"params": real_param, "lr": tc.lr}],
        ):
            result = module.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        # _step_lr_scheduler() looks the scheduler up via Lightning's own lr_schedulers()
        # accessor, which requires a real Trainer/strategy wiring this test never sets up.
        module.lr_schedulers = MagicMock(return_value=scheduler)
        return module, scheduler

    # --- _decoded_frame_count ---------------------------------------------------

    def test_decoded_frame_count_is_flat_one_for_stateless_image_batch(self, tmp_path):
        """A plain image microbatch always reports 1, independent of its (possibly auto-probed)
        sample count, so image-only training's LR schedule stays step-counted."""
        module, _, _, _ = _build_module(model_config=_base_model_config(), train_config=_base_train_config(tmp_path))

        assert module._decoded_frame_count(samples=object(), targets=[{}, {}, {}]) == 1

    def test_decoded_frame_count_is_flat_one_for_degenerate_length_one_clip(self, tmp_path):
        """A tracking clip with clip_length == 1 also reports a flat 1, matching the
        non-curriculum reference configure_optimizers() computes for it."""
        mc = _base_model_config(tracking={"enabled": True}, group_detr=1)
        tc = _base_train_config(tmp_path, tracking={"clip_length": 1})
        module, *_ = _build_module(model_config=mc, train_config=tc)
        samples = (object(),)
        targets = ([{}, {}],)

        assert module._decoded_frame_count(samples=samples, targets=targets) == 1

    def test_decoded_frame_count_is_batch_size_times_clip_length_for_curriculum_clip(self, tmp_path):
        """A genuine multi-frame clip reports every forwarded frame (PRD "decoded-video-frame
        budget"), including burn-in frames that carry no gradient but still run the model."""
        mc = _base_model_config(tracking={"enabled": True}, group_detr=1)
        tc = _base_train_config(tmp_path, tracking={"clip_length": 16})
        module, *_ = _build_module(model_config=mc, train_config=tc)
        samples = tuple(object() for _ in range(16))
        targets = tuple([{}, {}, {}] for _ in range(16))  # batch_size=3

        assert module._decoded_frame_count(samples=samples, targets=targets) == 3 * 16

    # --- _decoded_frames_per_optimizer_step reference ----------------------------

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_reference_is_grad_accum_steps_for_non_curriculum_runs(self, mock_get_param_dict, tmp_path):
        """Image-only (or degenerate clip_length<=1) runs use grad_accum_steps alone as the
        reference, independent of batch size -- the AC guard that keeps non-curriculum schedules
        byte-for-byte identical to the old +1-per-call counter."""
        real_param = nn.Parameter(torch.randn(4, 4))
        mock_get_param_dict.return_value = [{"params": real_param, "lr": 1e-4}]
        tc = _base_train_config(tmp_path, grad_accum_steps=4, batch_size=7)
        module, *_ = _build_module(train_config=tc)
        trainer = MagicMock()
        trainer.estimated_stepping_batches = 1000
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)

        module.configure_optimizers()

        assert module._decoded_frames_per_optimizer_step == pytest.approx(4.0)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_reference_scales_by_batch_size_and_clip_length_for_curriculum_runs(self, mock_get_param_dict, tmp_path):
        """A curriculum stage's reference is batch_size * clip_length * grad_accum_steps, matching
        exactly the frames one full optimizer-step window forwards."""
        real_param = nn.Parameter(torch.randn(4, 4))
        mock_get_param_dict.return_value = [{"params": real_param, "lr": 1e-4}]
        mc = _base_model_config(tracking={"enabled": True}, group_detr=1)
        tc = _base_train_config(tmp_path, batch_size=2, grad_accum_steps=3, tracking={"clip_length": 16})
        module, *_ = _build_module(model_config=mc, train_config=tc)
        trainer = MagicMock()
        trainer.estimated_stepping_batches = 1000
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)

        module.configure_optimizers()

        assert module._decoded_frames_per_optimizer_step == pytest.approx(2 * 16 * 3)

    # --- stepping mechanism -------------------------------------------------------

    def test_non_curriculum_stepping_matches_the_old_plus_one_per_call_progression(self, tmp_path):
        """For a constant frame count per optimizer step, feeding _step_lr_scheduler exactly
        _decoded_frames_per_optimizer_step frames per call must reproduce the historical
        +1-per-call LambdaLR progression exactly."""
        module, scheduler = self._module_with_scheduler(tmp_path, warmup_epochs=0.0, epochs=1, lr_scheduler="cosine")
        lr_lambda = scheduler.lr_lambdas[0]
        reference = module._decoded_frames_per_optimizer_step

        for expected_step in range(1, 6):
            module._pending_decoded_frames = int(reference)
            module._step_lr_scheduler()
            assert scheduler.last_epoch == pytest.approx(expected_step)
            assert scheduler.get_last_lr()[0] == pytest.approx(lr_lambda(expected_step) * scheduler.base_lrs[0])

    def test_heterogeneous_frame_counts_advance_progress_by_their_true_share(self, tmp_path):
        """A short 1-frame step and a long 16-frame step must NOT both count as "one step": the
        16-frame step must advance the schedule roughly 16x further than the 1-frame step."""
        module, scheduler = self._module_with_scheduler(tmp_path, warmup_epochs=0.0, epochs=1, lr_scheduler="cosine")
        reference = module._decoded_frames_per_optimizer_step

        module._pending_decoded_frames = int(reference)
        module._step_lr_scheduler()
        after_one_reference_worth = scheduler.last_epoch

        module._pending_decoded_frames = int(reference) * 16
        module._step_lr_scheduler()
        after_sixteen_reference_worth = scheduler.last_epoch

        assert after_one_reference_worth == pytest.approx(1.0)
        assert after_sixteen_reference_worth == pytest.approx(17.0)

    def test_lr_scheduler_step_hook_delegates_to_decoded_frame_progress(self, tmp_path):
        """The automatic-optimization hook (lr_scheduler_step) must drive the same decoded-frame
        accounting as the manual-optimization path (_step_lr_scheduler)."""
        module, scheduler = self._module_with_scheduler(tmp_path, warmup_epochs=0.0, epochs=1, lr_scheduler="cosine")
        reference = module._decoded_frames_per_optimizer_step

        module._pending_decoded_frames = int(reference)
        module.lr_scheduler_step(scheduler)

        assert scheduler.last_epoch == pytest.approx(1.0)
        assert module._pending_decoded_frames == 0

    def test_pending_frames_accumulate_across_grad_accum_microbatches_before_stepping(self, tmp_path):
        """training_step must add every microbatch's decoded frames to the pending counter, so a
        multi-microbatch accumulation window's full frame count reaches the scheduler in one step."""
        module, samples, targets, _, _ = TestTrainingStep()._run_step(
            tmp_path,
            loss_dict={"loss_ce": torch.tensor(1.0)},
            weight_dict={"loss_ce": 1.0},
            accumulate_grad_batches=2,
        )

        module.training_step((samples, targets), batch_idx=0)
        module.training_step((samples, targets), batch_idx=1)

        # _make_batch's default batch_size=2, but non-tracking microbatches always report a flat 1
        # (see _decoded_frame_count), so two microbatches accumulate exactly 2 pending frames.
        assert module._pending_decoded_frames == 2


class TestClipGradients:
    """Tests for clip_gradients() — verifies precision gating mirrors configure_optimizers()."""

    def _setup_module(self, tmp_path, precision: str):
        tc = _base_train_config(tmp_path)
        module, _, _, _ = _build_module(train_config=tc)
        trainer = MagicMock()
        trainer.precision = precision
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        return module

    @pytest.mark.parametrize(
        "precision",
        [
            pytest.param("32-true", id="fp32"),
            pytest.param("16-mixed", id="fp16-mixed"),
        ],
    )
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_clip_gradients_delegates_to_super_when_not_bf16(
        self,
        mock_cuda_available,
        mock_bf16_supported,
        precision,
        tmp_path,
    ):
        """clip_gradients must delegate to super() when trainer precision is not a BF16 variant.

        On Ampere+ GPUs is_bf16_supported() is True regardless of actual precision. The method must check
        trainer.precision before choosing the fused path, mirroring the same gate in configure_optimizers() to prevent
        silent divergence.
        """
        module = self._setup_module(tmp_path, precision=precision)

        with patch.object(type(module).__bases__[0], "clip_gradients") as mock_super_clip:
            module.clip_gradients(MagicMock(), gradient_clip_val=0.1)

        mock_super_clip.assert_called_once()

    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    @patch("rfdetr.training.module_model.torch.nn.utils.clip_grad_norm_")
    def test_clip_gradients_uses_clip_grad_norm_when_bf16_mixed(
        self,
        mock_clip_grad_norm,
        mock_cuda_available,
        mock_bf16_supported,
        tmp_path,
    ):
        """clip_gradients must call clip_grad_norm_ directly when precision is bf16-mixed.

        When fused AdamW is active (BF16, no GradScaler), the standard PTL AMP plugin refuses to clip gradients.
        clip_grad_norm_ is called directly instead, bypassing the scaler-aware path that would otherwise raise.
        """
        module = self._setup_module(tmp_path, precision="bf16-mixed")

        module.clip_gradients(MagicMock(), gradient_clip_val=0.5)

        mock_clip_grad_norm.assert_called_once()
        _, call_kwargs = mock_clip_grad_norm.call_args
        # Positional arg[1] is max_norm
        assert mock_clip_grad_norm.call_args[0][1] == pytest.approx(0.5)


class TestPredictStep:
    """Tests for predict_step() — verifies that only samples (not targets) are passed to the model, that postprocess
    receives the correct original sizes, and that the postprocessor output is returned directly to the caller."""

    def test_calls_postprocess_with_orig_sizes(self, build_module):
        """Postprocessor must receive a (batch, 2) tensor of original image sizes."""
        module, fake_model, _, fake_pp = build_module()
        samples, targets = _make_batch(batch_size=3)
        fake_model.return_value = {}

        module.predict_step((samples, targets), batch_idx=0)

        fake_pp.assert_called_once()
        orig_sizes = fake_pp.call_args[0][1]
        assert orig_sizes.shape == (3, 2)

    def test_returns_postprocess_output(self, build_module):
        """predict_step must return the postprocessor output directly to the caller."""
        module, fake_model, _, fake_pp = build_module()
        samples, targets = _make_batch()
        fake_model.return_value = {}
        expected_output = [{"boxes": torch.zeros(1, 4)}]
        fake_pp.return_value = expected_output

        assert module.predict_step((samples, targets), batch_idx=0) is expected_output

    def test_model_called_with_samples_only(self, build_module):
        """Inference must pass only samples (not targets) to the model forward."""
        module, fake_model, _, _ = build_module()
        samples, targets = _make_batch()
        fake_model.return_value = {}

        module.predict_step((samples, targets), batch_idx=0)

        fake_model.assert_called_once_with(samples)

    def test_default_dataloader_idx_is_zero(self, build_module):
        """predict_step must work with the default dataloader_idx without errors."""
        module, fake_model, _, _ = build_module()
        fake_model.return_value = {}

        # Should not raise with default dataloader_idx.
        module.predict_step(_make_batch(), batch_idx=0)


class TestReinitializeDetectionHead:
    """Tests for reinitialize_detection_head() — verifies that the module delegates to the underlying model and that
    arbitrary class counts are forwarded unchanged."""

    def test_delegates_to_model(self, build_module):
        """Module must delegate head reinitialization to the underlying model."""
        module, fake_model, _, _ = build_module()

        module.reinitialize_detection_head(num_classes=42)

        fake_model.reinitialize_detection_head.assert_called_once_with(42)

    @pytest.mark.parametrize(
        "num_classes",
        [
            pytest.param(1, id="single-class"),
            pytest.param(80, id="coco-80"),
            pytest.param(365, id="objects365"),
        ],
    )
    def test_passes_various_class_counts(self, num_classes, build_module):
        """Arbitrary class counts must be forwarded to the underlying model unchanged."""
        module, fake_model, _, _ = build_module()

        module.reinitialize_detection_head(num_classes=num_classes)

        fake_model.reinitialize_detection_head.assert_called_once_with(num_classes)


class TestOnLoadCheckpoint:
    """Tests for on_load_checkpoint() — covers legacy .pth normalisation and positional-embedding interpolation for
    custom-resolution PTL checkpoints.

    Regression: issue #998 — resume with custom resolution crashed because
    on_load_checkpoint did not interpolate PE before PTL applied the state dict.
    """

    _PE_KEY = "model.backbone.0.encoder.encoder.embeddings.position_embeddings"

    def _make_ptl_checkpoint(self, pe_size_src: int, _pe_size_tgt: int, dim: int = 16) -> dict:
        """Build a minimal PTL checkpoint with mismatched PE shape.

        Args:
            pe_size_src: Source grid side length (checkpoint was saved with this PE).
            _pe_size_tgt: Target grid side length (model was built with this PE),
                accepted for test readability but intentionally unused here.
            dim: Embedding dimension (small value for fast tests).

        Returns:
            Checkpoint dict in PTL format with ``state_dict`` key.
        """
        n_src = pe_size_src * pe_size_src + 1  # +1 for class token
        return {
            "state_dict": {
                self._PE_KEY: torch.randn(1, n_src, dim),
                "model.other_layer.weight": torch.randn(4, 4),
            },
            "epoch": 44,
            "global_step": 1000,
        }

    def _make_legacy_pth_checkpoint(self, pe_size_src: int, dim: int = 16) -> dict:
        """Build a minimal legacy .pth checkpoint (no ``state_dict`` key).

        Args:
            pe_size_src: Source grid side length.
            dim: Embedding dimension.

        Returns:
            Checkpoint dict in legacy format with ``model`` key only.
        """
        n_src = pe_size_src * pe_size_src + 1
        pe_key_no_prefix = self._PE_KEY[len("model.") :]
        return {
            "model": {
                pe_key_no_prefix: torch.randn(1, n_src, dim),
                "other_layer.weight": torch.randn(4, 4),
            }
        }

    @pytest.mark.parametrize(
        "pe_src,pe_tgt",
        [
            pytest.param(36, 56, id="pe_interpolated_in_ptl_checkpoint"),
            pytest.param(36, 36, id="pe_unchanged_when_shapes_match"),
        ],
    )
    def test_ptl_checkpoint_pe_shape(self, pe_src, pe_tgt, build_module):
        """on_load_checkpoint must produce PE with tokens matching the model's positional_encoding_size.

        Regression for #998: resume from .ckpt with custom resolution crashed because PTL applied the checkpoint state
        dict before PE shapes were reconciled.
        """
        checkpoint = self._make_ptl_checkpoint(pe_size_src=pe_src, _pe_size_tgt=pe_tgt)

        module, _, _, _ = build_module(model_config=_base_model_config(positional_encoding_size=pe_tgt))
        module.on_load_checkpoint(checkpoint)

        pe_after = checkpoint["state_dict"][self._PE_KEY]
        expected_tokens = pe_tgt * pe_tgt + 1
        assert pe_after.shape == (
            1,
            expected_tokens,
            16,
        ), f"PE should have {expected_tokens} tokens, got shape {tuple(pe_after.shape)}"

    def test_legacy_pth_normalised_and_pe_interpolated(self, build_module):
        """Legacy .pth checkpoint (no state_dict key) must be normalised and PE interpolated.

        on_load_checkpoint converts the raw "model" dict to PTL format and must also interpolate PE so that PTL's
        subsequent load_state_dict does not crash.
        """
        pe_src, pe_tgt = 36, 56
        checkpoint = self._make_legacy_pth_checkpoint(pe_size_src=pe_src)

        module, _, _, _ = build_module(model_config=_base_model_config(positional_encoding_size=pe_tgt))
        module.on_load_checkpoint(checkpoint)

        assert "state_dict" in checkpoint, "Legacy checkpoint must be normalised to PTL format."
        pe_after = checkpoint["state_dict"][self._PE_KEY]
        expected_tokens = pe_tgt * pe_tgt + 1
        assert pe_after.shape == (1, expected_tokens, 16)

    def test_non_pe_tensors_not_modified(self, build_module):
        """on_load_checkpoint must not alter non-PE tensors in the state dict."""
        pe_src, pe_tgt = 36, 56
        checkpoint = self._make_ptl_checkpoint(pe_size_src=pe_src, _pe_size_tgt=pe_tgt)
        original_other = checkpoint["state_dict"]["model.other_layer.weight"].clone()

        module, _, _, _ = build_module(model_config=_base_model_config(positional_encoding_size=pe_tgt))
        module.on_load_checkpoint(checkpoint)

        assert torch.equal(checkpoint["state_dict"]["model.other_layer.weight"], original_other)

    def test_no_pe_keys_in_state_dict_is_noop(self, build_module):
        """on_load_checkpoint must not raise when state_dict contains no PE keys."""
        checkpoint = {
            "state_dict": {"model.some_layer.weight": torch.randn(4, 4)},
            "epoch": 1,
        }
        original_keys = set(checkpoint["state_dict"].keys())

        module, _, _, _ = build_module(model_config=_base_model_config(positional_encoding_size=36))
        module.on_load_checkpoint(checkpoint)

        assert set(checkpoint["state_dict"].keys()) == original_keys


class TestOnSaveCheckpoint:
    """Native last/periodic resume checkpoints use the same authoritative metadata."""

    def test_native_checkpoint_serializes_authoritative_configuration(self, build_module, tmp_path):
        model_config = _base_model_config(
            group_detr=1,
            tracking={"enabled": True, "max_active_tracks": 40, "discovery_reserve": 10},
        )
        train_config = _base_train_config(
            tmp_path,
            dataset_file="video",
            tracking={"clip_length": 2},
            group_detr=13,
            ia_bce_loss=False,
            num_select=100,
            segmentation_head=True,
        )
        module, _, _, _ = build_module(model_config=model_config, train_config=train_config)
        checkpoint = {"epoch": 4, "state_dict": {}}

        module.on_save_checkpoint(checkpoint)

        assert checkpoint["checkpoint_schema_version"] == 1
        assert checkpoint["model_config_type"] == "RFDETRBaseConfig"
        assert checkpoint["model_config"]["group_detr"] == 1
        assert checkpoint["model_config"]["tracking"] == {
            "enabled": True,
            "max_active_tracks": 40,
            "discovery_reserve": 10,
        }
        assert checkpoint["class_schema"] == checkpoint["model_config"]["class_schema"]
        assert checkpoint["epoch"] == 4
        assert checkpoint["weight_flavor"] == "regular"
        assert "source_checkpoint_hash" in checkpoint
        for deprecated in ("group_detr", "ia_bce_loss", "num_select", "segmentation_head"):
            assert deprecated not in checkpoint["train_config"]
            assert deprecated not in checkpoint["args"]
            assert deprecated not in checkpoint["hyper_parameters"]

    def test_source_checkpoint_hash_uses_file_content(self, tmp_path):
        """Source lineage hashes are computed from checkpoint bytes."""
        source = tmp_path / "source.pth"
        source.write_bytes(b"source detector")

        result = source_checkpoint_hash(SimpleNamespace(pretrain_weights=source))

        assert result == sha256(b"source detector").hexdigest()
