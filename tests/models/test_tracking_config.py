# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
from pydantic import ValidationError

from rfdetr.config import (
    ClassSchema,
    ForegroundClass,
    RFDETRBaseConfig,
    TrackingConfig,
    TrackingPolicy,
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

    def test_dancetrack_class_schema_is_explicit_stable_and_hashable(self) -> None:
        """DanceTrack declares person at logit zero and no-object at logit one."""
        schema = ClassSchema(
            foreground_classes=(ForegroundClass(class_id=0, name="person", external_category_id=0),),
            background_logit_index=1,
            logit_activation="sigmoid_independent",
        )
        restored = ClassSchema.model_validate_json(schema.canonical_json())

        assert restored == schema
        assert restored.foreground_class_ids == (0,)
        assert restored.external_category_mapping == {0: 0}
        assert restored.sha256() == schema.sha256()
        assert hash(restored) == hash(schema)

    def test_multiclass_schema_round_trips_in_canonical_class_order(self) -> None:
        """Foreground entries serialize by model class ID independent of input ordering."""
        schema = ClassSchema(
            foreground_classes=(
                ForegroundClass(class_id=1, name="dog", external_category_id=12),
                ForegroundClass(class_id=0, name="cat", external_category_id=7),
            ),
            background_logit_index=2,
            logit_activation="softmax_exclusive",
        )

        assert schema.foreground_class_ids == (0, 1)
        assert schema.external_category_mapping == {0: 7, 1: 12}
        assert ClassSchema.model_validate(schema.model_dump()) == schema
        assert (
            ClassSchema(
                foreground_classes=(ForegroundClass(class_id=0, name="cat", external_category_id=7),)
            ).model_dump()["background_logit_index"]
            is None
        )

    @pytest.mark.parametrize(
        ("override", "message"),
        [
            pytest.param(
                {
                    "foreground_classes": (
                        ForegroundClass(class_id=0, name="cat", external_category_id=1),
                        ForegroundClass(class_id=0, name="dog", external_category_id=2),
                    )
                },
                "class_id values must be unique",
                id="duplicate-foreground",
            ),
            pytest.param(
                {
                    "foreground_classes": (ForegroundClass(class_id=0, name="cat", external_category_id=1),),
                    "background_logit_index": 0,
                },
                "background_logit_index must not identify a foreground class",
                id="background-overlap",
            ),
        ],
    )
    def test_class_schema_rejects_ambiguous_logit_roles(self, override: dict[str, object], message: str) -> None:
        """A logit cannot have duplicate foreground roles or both foreground/background roles."""
        with pytest.raises(ValidationError, match=message):
            ClassSchema(**override)

    def test_tracking_requires_an_authoritative_class_schema(self) -> None:
        """Temporal models fail closed when foreground/background roles are unknown."""
        with pytest.raises(ValidationError, match="class_schema.*temporal checkpoint"):
            RFDETRBaseConfig(
                pretrain_weights=None,
                group_detr=1,
                tracking=TrackingConfig(enabled=True),
            )

    def test_tracking_configuration_domains_are_independent(self) -> None:
        """Architecture, clip training, and inference lifecycle settings use distinct schemas."""
        architecture = TrackingConfig(enabled=True, max_active_tracks=250, discovery_reserve=50)
        training = TrackingTrainConfig(clip_length=4)
        lifecycle = TrackingSessionConfig(activation_threshold=0.7, max_missed_frames=12)

        assert architecture.max_active_tracks == 250
        assert training.clip_length == 4
        assert lifecycle.activation_threshold == 0.7
        assert lifecycle.tentative_confirmation_hits == 2
        assert lifecycle.tentative_confirmation_window_frames == 3
        assert lifecycle.tentative_max_misses == 2

        with pytest.raises(ValidationError, match="hits cannot exceed"):
            TrackingSessionConfig(tentative_confirmation_hits=4, tentative_confirmation_window_frames=3)

    def test_discovery_bounds_are_independent_of_decoder_query_capacity(self) -> None:
        """Discovery and tentative caps are lifecycle policy, not architecture capacity."""
        architecture = TrackingConfig(enabled=True, max_active_tracks=250, discovery_reserve=50)
        lifecycle = TrackingSessionConfig()

        assert lifecycle.max_discovery_candidates_per_frame == 10
        assert lifecycle.max_tentative_tracks == 10
        assert architecture.active_capacity(num_queries=300) == 250

    def test_reassociation_is_disabled_by_default(self) -> None:
        """The discovery-to-suspended reassociation experiment (PRD US-024) opts in explicitly."""
        lifecycle = TrackingSessionConfig()

        assert lifecycle.reassociation_enabled is False
        assert lifecycle.reassociation_iou_threshold == 0.5

        enabled = TrackingSessionConfig(reassociation_enabled=True, reassociation_iou_threshold=0.6)
        assert enabled.reassociation_enabled is True
        assert enabled.reassociation_iou_threshold == 0.6

    def test_reassociation_iou_threshold_is_bounded(self) -> None:
        """The reassociation threshold shares the same normalized IoU domain as other lifecycle IoUs."""
        with pytest.raises(ValidationError):
            TrackingSessionConfig(reassociation_iou_threshold=1.5)
        with pytest.raises(ValidationError):
            TrackingSessionConfig(reassociation_iou_threshold=-0.1)

    def test_locked_policy_recovery_mode_is_unaffected_by_the_reassociation_experiment(self) -> None:
        """`to_session_config` never implies the conditional reassociation experiment."""
        policy = TrackingPolicy(**self._policy_fields())

        session_config = policy.to_session_config()

        assert session_config.reassociation_enabled is False

    def test_motion_reference_prediction_is_disabled_by_default(self) -> None:
        """The elapsed-time/velocity reference-prediction experiment (PRD US-025) opts in explicitly."""
        lifecycle = TrackingSessionConfig()

        assert lifecycle.motion_reference_prediction_enabled is False

        enabled = TrackingSessionConfig(motion_reference_prediction_enabled=True)
        assert enabled.motion_reference_prediction_enabled is True

    def test_locked_policy_is_unaffected_by_the_motion_experiment(self) -> None:
        """`to_session_config` never implies the conditional motion experiment."""
        policy = TrackingPolicy(**self._policy_fields())

        session_config = policy.to_session_config()

        assert session_config.motion_reference_prediction_enabled is False

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
            RFDETRBaseConfig(pretrain_weights=None, group_detr=1, tracking=tracking)

    def _policy_fields(self, **overrides: object) -> dict[str, object]:
        base = {
            "foreground_schema_hash": "a" * 64,
            "activation_threshold": 0.5,
            "continuation_threshold": 0.3,
            "duplicate_iou_threshold": 0.7,
            "max_missed_frames": 30,
            "tentative_confirmation_hits": 2,
            "tentative_confirmation_window_frames": 3,
            "tentative_max_misses": 2,
            "max_discovery_candidates_per_frame": 10,
            "max_tentative_tracks": 10,
            "max_active_tracks": 40,
        }
        base.update(overrides)
        return base

    def test_tracking_policy_is_the_one_versioned_deployment_contract(self) -> None:
        """A locked policy carries the schema, thresholds, caps, capacity, and recovery mode."""
        policy = TrackingPolicy(**self._policy_fields())

        assert policy.schema_version == "clevis.rfdetr-tracking-policy-lock-v1"
        assert policy.foreground_schema_hash == "a" * 64
        assert policy.recovery_mode == "same_slot_suspended"
        assert policy.max_active_tracks == 40
        restored = TrackingPolicy.model_validate_json(policy.canonical_json())
        assert restored == policy
        assert restored.sha256() == policy.sha256()

    def test_tracking_policy_converts_to_the_equivalent_session_config(self) -> None:
        """The session config a policy implies matches its own thresholds exactly."""
        policy = TrackingPolicy(**self._policy_fields(activation_threshold=0.6))

        session_config = policy.to_session_config(collect_timing=True)

        assert session_config == TrackingSessionConfig(
            activation_threshold=0.6,
            continuation_threshold=0.3,
            duplicate_iou_threshold=0.7,
            max_missed_frames=30,
            tentative_confirmation_hits=2,
            tentative_confirmation_window_frames=3,
            tentative_max_misses=2,
            max_discovery_candidates_per_frame=10,
            max_tentative_tracks=10,
            collect_timing=True,
        )

    def test_tracking_policy_rejects_invalid_tentative_confirmation(self) -> None:
        """Locked policies enforce the same tentative-confirmation invariant as raw configs."""
        with pytest.raises(ValidationError, match="hits cannot exceed"):
            TrackingPolicy(
                **self._policy_fields(tentative_confirmation_hits=4, tentative_confirmation_window_frames=3)
            )

    def test_tracking_configuration_round_trips_through_serialization(self) -> None:
        """Nested tracking settings survive the existing Pydantic serialization path."""
        config = RFDETRBaseConfig(
            pretrain_weights=None,
            group_detr=1,
            tracking=TrackingConfig(enabled=True, max_active_tracks=240, discovery_reserve=60),
            class_schema=ClassSchema(
                foreground_classes=(ForegroundClass(class_id=0, name="person", external_category_id=0),),
                background_logit_index=1,
            ),
            num_classes=1,
        )

        restored = RFDETRBaseConfig.model_validate(config.model_dump())

        assert restored == config
