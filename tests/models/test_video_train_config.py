# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Behavioral tests for the video-training configuration contract."""

from pathlib import Path

import pytest

from rfdetr.config import (
    ErrorExposureCurriculumConfig,
    RFDETRBaseConfig,
    TrackingConfig,
    TrackingTrainConfig,
    TrainConfig,
)


def _model_config(**overrides: object) -> RFDETRBaseConfig:
    """Build a supported tracking architecture with optional overrides."""
    values: dict[str, object] = {
        "pretrain_weights": None,
        "num_classes": 1,
        "group_detr": 1,
        "tracking": TrackingConfig(enabled=True),
        "class_schema": {
            "foreground_classes": [{"class_id": 0, "name": "person", "external_category_id": 0}],
            "background_logit_index": 1,
        },
    }
    values.update(overrides)
    return RFDETRBaseConfig(**values)


def _train_config(tmp_path: Path, **overrides: object) -> TrainConfig:
    """Build a valid video training configuration with optional overrides."""
    values: dict[str, object] = {
        "dataset_dir": tmp_path,
        "dataset_file": "video",
        "tracking": TrackingTrainConfig(
            clip_length=4,
            clip_stride=2,
            annotation_path="annotations/train.json",
        ),
    }
    values.update(overrides)
    return TrainConfig(**values)


def test_video_dataset_configuration_round_trips_without_changing_image_defaults(tmp_path: Path) -> None:
    """Video clip indexing settings survive the ordinary training-config serialization path."""
    image_config = TrainConfig(dataset_dir=tmp_path)
    video_config = _train_config(tmp_path)

    restored = TrainConfig.model_validate(video_config.model_dump())

    assert image_config.dataset_file == "roboflow"
    assert image_config.tracking == TrackingTrainConfig()
    assert restored.tracking == video_config.tracking
    assert restored.dataset_file == video_config.dataset_file
    assert restored.tracking.clip_stride == 2
    assert restored.tracking.annotation_path == "annotations/train.json"
    assert video_config.model_dump(mode="json")["tracking"] == {
        "clip_length": 4,
        "clip_stride": 2,
        "annotation_path": "annotations/train.json",
        "detach_state_between_frames": False,
        "burn_in_frames": 0,
        "supervised_frames": 4,
        "tbptt_chunk_frames": None,
        "lifecycle_mode": "assignment_guided",
        "lifecycle_commitment_curriculum": {
            "mode": "disabled",
            "warmup_epochs": 0,
            "warmup_steps": 0,
        },
        "lifecycle": {
            "activation_threshold": 0.5,
            "continuation_threshold": 0.3,
            "duplicate_iou_threshold": 0.7,
            "max_missed_frames": 30,
            "tentative_confirmation_hits": 2,
            "tentative_confirmation_window_frames": 3,
            "tentative_max_misses": 2,
            "tentative_association_iou_threshold": 0.3,
            "max_discovery_candidates_per_frame": 10,
            "max_tentative_tracks": 10,
            "collision_iou_threshold": 0.7,
            "collision_persistence_frames": 3,
            "collision_loser_outcome": "suspended",
            "reassociation_enabled": False,
            "reassociation_iou_threshold": 0.5,
            "motion_reference_prediction_enabled": False,
            "identity_memory_enabled": False,
            "identity_memory_length": 8,
            "identity_memory_reliable_threshold": 0.7,
            "collect_timing": False,
        },
        "tracking_eval_interval_epochs": 5,
        "false_positive_injection_enabled": False,
        "false_positive_injection_probability": 0.10,
        "false_positive_injection_max_per_sample": 2,
        "query_dropout_enabled": False,
        "query_dropout_probability": 0.10,
        "error_exposure_seed": 0,
        "error_exposure_curriculum": {
            "mode": "disabled",
            "warmup_epochs": 0,
            "ramp_epochs": 0,
            "warmup_steps": 0,
            "ramp_steps": 0,
        },
    }


def test_tracking_eval_interval_epochs_defaults_to_five() -> None:
    """PRD Section 7.5: the chronological calibration-tracking eval schedule defaults to every 5 epochs."""
    assert TrackingTrainConfig().tracking_eval_interval_epochs == 5


@pytest.mark.parametrize("value", [1, 2, 100])
def test_tracking_eval_interval_epochs_accepts_positive_integers(value: int) -> None:
    """Any integer >= 1 is a valid evaluation interval."""
    assert TrackingTrainConfig(tracking_eval_interval_epochs=value).tracking_eval_interval_epochs == value


@pytest.mark.parametrize("value", [0, -1])
def test_tracking_eval_interval_epochs_rejects_non_positive_integers(value: int) -> None:
    """An interval of zero or fewer epochs would evaluate never or with undefined semantics."""
    with pytest.raises(ValueError):
        TrackingTrainConfig(tracking_eval_interval_epochs=value)


def test_error_exposure_defaults_match_prd_us019() -> None:
    """PRD US-019 AC: both mechanisms default to disabled, probability 0.10, and a 2-per-sample FP cap."""
    tracking = TrackingTrainConfig()
    assert tracking.false_positive_injection_enabled is False
    assert tracking.false_positive_injection_probability == pytest.approx(0.10)
    assert tracking.false_positive_injection_max_per_sample == 2
    assert tracking.query_dropout_enabled is False
    assert tracking.query_dropout_probability == pytest.approx(0.10)
    assert tracking.error_exposure_seed == 0


@pytest.mark.parametrize(
    "field",
    ["false_positive_injection_probability", "query_dropout_probability"],
)
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_error_exposure_probabilities_are_bounded(field: str, value: float) -> None:
    """Probabilities outside [0, 1] are not valid sampling parameters."""
    with pytest.raises(ValueError):
        TrackingTrainConfig(**{field: value})


def test_false_positive_injection_max_per_sample_rejects_negative() -> None:
    """A negative per-sample injection cap has no meaning."""
    with pytest.raises(ValueError):
        TrackingTrainConfig(false_positive_injection_max_per_sample=-1)


def test_error_exposure_curriculum_defaults_to_disabled() -> None:
    """Duplicate-FP PRD US-010: the sampling-rate curriculum is off by default, so the
    configured injection/dropout probabilities apply unchanged for the whole run."""
    curriculum = TrackingTrainConfig().error_exposure_curriculum
    assert curriculum.mode == "disabled"
    assert curriculum.factor_at(epoch=0, step=0) == 1.0
    assert curriculum.factor_at(epoch=99, step=9999) == 1.0


def test_error_exposure_curriculum_epoch_warmup_then_linear_ramp() -> None:
    """US-010: an epoch schedule holds the factor at 0 through the warm-up, then ramps it
    linearly to 1 over the ramp window and sustains 1 afterwards."""
    curriculum = ErrorExposureCurriculumConfig(mode="epoch", warmup_epochs=2, ramp_epochs=4)
    assert curriculum.factor_at(epoch=1, step=0) == 0.0
    assert curriculum.factor_at(epoch=2, step=0) == 0.0
    assert curriculum.factor_at(epoch=4, step=0) == pytest.approx(0.5)
    assert curriculum.factor_at(epoch=6, step=0) == 1.0
    assert curriculum.factor_at(epoch=100, step=0) == 1.0


def test_error_exposure_curriculum_step_schedule_without_ramp_is_a_step_function() -> None:
    """US-010: a zero ramp window switches straight from 0 to 1 once the step warm-up elapses."""
    curriculum = ErrorExposureCurriculumConfig(mode="step", warmup_steps=10, ramp_steps=0)
    assert curriculum.factor_at(epoch=0, step=9) == 0.0
    assert curriculum.factor_at(epoch=0, step=10) == 1.0


@pytest.mark.parametrize("field", ["warmup_epochs", "ramp_epochs", "warmup_steps", "ramp_steps"])
def test_error_exposure_curriculum_rejects_negative_windows(field: str) -> None:
    """Negative warm-up or ramp windows have no schedule meaning."""
    with pytest.raises(ValueError):
        ErrorExposureCurriculumConfig(**{field: -1})


def test_burn_in_and_tbptt_default_to_no_curriculum() -> None:
    """PRD US-020 AC: a plain clip_length-only config keeps the whole clip supervised in one chunk."""
    tracking = TrackingTrainConfig(clip_length=4)
    assert tracking.burn_in_frames == 0
    assert tracking.supervised_frames == 4
    assert tracking.tbptt_chunk_frames is None


@pytest.mark.parametrize(
    ("clip_length", "burn_in_frames", "supervised_frames", "tbptt_chunk_frames"),
    [
        pytest.param(16, 12, 4, 4, id="stage-1"),
        pytest.param(64, 48, 16, 8, id="stage-2"),
        pytest.param(128, 96, 32, 8, id="stage-3"),
    ],
)
def test_prd_section_7_6_curriculum_stages_validate(
    clip_length: int, burn_in_frames: int, supervised_frames: int, tbptt_chunk_frames: int
) -> None:
    """PRD US-020 AC: the exact 16/64/128-frame Section 7.6 curriculum settings validate successfully."""
    tracking = TrackingTrainConfig(
        clip_length=clip_length,
        burn_in_frames=burn_in_frames,
        supervised_frames=supervised_frames,
        tbptt_chunk_frames=tbptt_chunk_frames,
    )
    assert tracking.clip_length == clip_length
    assert tracking.burn_in_frames == burn_in_frames
    assert tracking.supervised_frames == supervised_frames
    assert tracking.tbptt_chunk_frames == tbptt_chunk_frames


def test_burn_in_plus_supervised_must_equal_clip_length() -> None:
    """An inconsistent split is rejected rather than silently truncating or padding the clip."""
    with pytest.raises(ValueError, match="must equal clip_length"):
        TrackingTrainConfig(clip_length=16, burn_in_frames=12, supervised_frames=5)


def test_supervised_frames_defaults_around_explicit_burn_in() -> None:
    """Setting only burn_in_frames still derives a consistent supervised_frames default."""
    tracking = TrackingTrainConfig(clip_length=16, burn_in_frames=12)
    assert tracking.supervised_frames == 4


def test_tbptt_chunk_frames_rejects_non_positive() -> None:
    """A chunk of zero or fewer frames has no meaning."""
    with pytest.raises(ValueError):
        TrackingTrainConfig(tbptt_chunk_frames=0)


@pytest.mark.parametrize(
    ("model_overrides", "train_overrides", "message"),
    [
        pytest.param(
            {"tracking": TrackingConfig(enabled=False)},
            {},
            "model_config.tracking.enabled=True",
            id="tracking-disabled",
        ),
        pytest.param(
            {},
            {"tracking": TrackingTrainConfig(clip_length=1)},
            "tracking.clip_length greater than one",
            id="single-frame",
        ),
        pytest.param({}, {"batch_size": "auto"}, "explicit integer batch_size", id="auto-batch"),
        pytest.param({}, {"augmentation_backend": "gpu"}, "augmentation_backend='cpu'", id="gpu-augmentation"),
        pytest.param({}, {"augmentation_backend": "auto"}, "augmentation_backend='cpu'", id="auto-augmentation"),
    ],
)
def test_video_training_rejects_unsupported_combinations(
    tmp_path: Path,
    model_overrides: dict[str, object],
    train_overrides: dict[str, object],
    message: str,
) -> None:
    """Unsupported first-version video combinations fail before dataset construction."""
    model_config = _model_config(**model_overrides)
    train_config = _train_config(tmp_path, **train_overrides)

    with pytest.raises(ValueError, match=message):
        train_config.validate_for_model(model_config)


def test_image_training_does_not_apply_video_only_constraints(tmp_path: Path) -> None:
    """Legacy image settings remain valid even when they are unsupported for video."""
    train_config = TrainConfig(
        dataset_dir=tmp_path,
        batch_size="auto",
        augmentation_backend="gpu",
    )

    train_config.validate_for_model(RFDETRBaseConfig(pretrain_weights=None))
