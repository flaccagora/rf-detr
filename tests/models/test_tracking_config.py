# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
from pydantic import ValidationError

from rfdetr.config import (
    RFDETRBaseConfig,
    TrackingConfig,
    TrackingSessionConfig,
    TrackingTrainConfig,
)


class TestTrackingConfiguration:
    """Behavioral tests for persistent-query tracking configuration."""

    def test_existing_model_configs_disable_tracking_by_default(self) -> None:
        """Existing model configuration remains stateless unless tracking is explicitly enabled."""
        config = RFDETRBaseConfig(pretrain_weights=None)

        assert config.tracking == TrackingConfig()
        assert config.tracking.enabled is False

    def test_tracking_configuration_domains_are_independent(self) -> None:
        """Architecture, clip training, and inference lifecycle settings use distinct schemas."""
        architecture = TrackingConfig(enabled=True, max_active_tracks=250, discovery_reserve=50)
        training = TrackingTrainConfig(clip_length=4)
        lifecycle = TrackingSessionConfig(activation_threshold=0.7, max_missed_frames=12)

        assert architecture.max_active_tracks == 250
        assert training.clip_length == 4
        assert lifecycle.activation_threshold == 0.7

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            pytest.param({"segmentation_head": True}, "detection-only", id="segmentation"),
            pytest.param({"use_grouppose_keypoints": True}, "detection-only", id="keypoints"),
            pytest.param({"two_stage": False}, "two_stage=True", id="one-stage"),
            pytest.param({"group_detr": 2}, "group_detr=1", id="query-groups"),
        ],
    )
    def test_enabled_tracking_rejects_unsupported_model_capabilities(
        self, override: dict[str, object], message: str
    ) -> None:
        """Unsupported tracking architectures fail during configuration construction."""
        with pytest.raises(ValidationError, match=message):
            RFDETRBaseConfig(
                pretrain_weights=None,
                tracking=TrackingConfig(enabled=True),
                **override,
            )

    @pytest.mark.parametrize(
        "tracking",
        [
            pytest.param(
                TrackingConfig(enabled=True, max_active_tracks=300, discovery_reserve=1),
                id="capacity-plus-reserve-exceeds-queries",
            ),
            pytest.param(
                TrackingConfig(enabled=True, max_active_tracks=299, discovery_reserve=2),
                id="combined-capacity-exceeds-queries",
            ),
        ],
    )
    def test_invalid_tracking_capacity_has_actionable_error(self, tracking: TrackingConfig) -> None:
        """Active capacity and discovery reserve must fit within the fixed query count."""
        with pytest.raises(
            ValidationError,
            match=r"max_active_tracks .* discovery_reserve .* num_queries",
        ):
            RFDETRBaseConfig(pretrain_weights=None, tracking=tracking)

    def test_tracking_configuration_round_trips_through_serialization(self) -> None:
        """Nested tracking settings survive the existing Pydantic serialization path."""
        config = RFDETRBaseConfig(
            pretrain_weights=None,
            group_detr=1,
            tracking=TrackingConfig(enabled=True, max_active_tracks=240, discovery_reserve=60),
        )

        restored = RFDETRBaseConfig.model_validate(config.model_dump())

        assert restored == config
