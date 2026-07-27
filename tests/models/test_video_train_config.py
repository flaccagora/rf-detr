# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Behavioral tests for the video-training configuration contract."""

from pathlib import Path

import pytest

from rfdetr.config import RFDETRBaseConfig, TrackingConfig, TrackingTrainConfig, TrainConfig


def _model_config(**overrides: object) -> RFDETRBaseConfig:
    """Build a supported tracking architecture with optional overrides."""
    values: dict[str, object] = {
        "pretrain_weights": None,
        "group_detr": 1,
        "tracking": TrackingConfig(enabled=True),
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
        "lifecycle_mode": "assignment_guided",
    }


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
