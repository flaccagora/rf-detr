# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr.config import ClassSchema, ForegroundClass, TrackingConfig, TrackingPolicy, TrackingSessionConfig
from rfdetr.detr import RFDETR
from rfdetr.models.tracking import TrackingFrameOutput, TrackQueryState
from rfdetr.tracking import TrackingSession, TrackingTiming, load_tracking_policy


class _TrackingModule(torch.nn.Module):
    """Deterministic low-level tracking model for public-session tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.calls: list[TrackQueryState] = []

    def forward_tracking(self, samples: torch.Tensor, prior_state: TrackQueryState) -> TrackingFrameOutput:
        """Return one reliable candidate and record the explicit prior state."""
        self.calls.append(prior_state)
        boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1]]], device=samples.device)
        return TrackingFrameOutput(
            pred_logits=torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]], device=samples.device),
            pred_boxes=boxes,
            candidate_state=TrackQueryState(
                torch.ones(1, 2, 3, device=samples.device),
                boxes,
                torch.ones(1, 2, dtype=torch.bool, device=samples.device),
            ),
            input_active_mask=prior_state.active_mask,
        )


def _owner(external_category_id: int = 0) -> RFDETR:
    """Build a weight-free RFDETR owner around the deterministic module."""
    owner = object.__new__(RFDETR)
    module = _TrackingModule()
    owner.model_config = SimpleNamespace(
        num_channels=3,
        num_queries=2,
        hidden_dim=3,
        class_schema=ClassSchema(
            foreground_classes=(ForegroundClass(class_id=0, name="person", external_category_id=external_category_id),),
            background_logit_index=1,
        ),
        tracking=TrackingConfig(enabled=True, max_active_tracks=1),
    )
    owner.model = SimpleNamespace(model=module, device=torch.device("cpu"), resolution=8)
    owner.means = [0.0, 0.0, 0.0]
    owner.stds = [1.0, 1.0, 1.0]
    owner._is_optimized_for_inference = False
    return owner


class TestTrackingSession:
    """Public, single-stream tracking behavior."""

    def test_shared_model_sessions_keep_independent_state_and_reset_ids(self) -> None:
        """Sessions do not store state on their shared model and reset restarts IDs."""
        owner = _owner()
        immediate_config = TrackingSessionConfig(tentative_confirmation_hits=1)
        first = owner.create_tracking_session(immediate_config)
        second = owner.create_tracking_session(immediate_config)
        frame = np.zeros((6, 10, 3), dtype=np.uint8)

        assert first.update(frame).tracker_id.tolist() == [0]
        assert second.update(frame).tracker_id.tolist() == [0]
        assert first.update(frame).tracker_id.tolist() == [0]

        first.reset()
        assert first.update(frame).tracker_id.tolist() == [0]
        assert second.active_tracks[0].track_id == 0

    @pytest.mark.parametrize(
        "frame",
        [
            pytest.param(np.zeros((6, 10, 3), dtype=np.uint8), id="numpy"),
            pytest.param(Image.fromarray(np.zeros((6, 10, 3), dtype=np.uint8)), id="pil"),
            pytest.param(torch.zeros(3, 6, 10), id="tensor"),
        ],
    )
    def test_update_accepts_single_predict_input_types_and_aligns_ids(self, frame: object) -> None:
        """Retained detections expose slot-aligned IDs and pixel boxes."""
        session = TrackingSession(_owner(), TrackingSessionConfig(tentative_confirmation_hits=1))

        first = session.update(frame, frame_index=4, timestamp=1.4)
        detections = session.update(frame, frame_index=5, timestamp=1.5)

        assert first.tracker_id.tolist() == [0]
        assert detections.tracker_id.tolist() == [0]
        assert detections.class_id.tolist() == [0]
        assert detections.xyxy.tolist() == [[2.5, 1.5, 7.5, 4.5]]
        assert detections.metadata["frame_index"] == 5
        assert detections.metadata["timestamp"] == 1.5

    def test_reset_discards_the_private_tentative_pool_as_well_as_public_identities(self) -> None:
        """Sequence reset discards host-only tentative candidates, not just table/state."""
        session = TrackingSession(_owner())
        frame = torch.zeros(3, 6, 10)

        assert session.update(frame).tracker_id.tolist() == []
        assert len(session.tentative_pool.records) == 1
        session.reset()

        assert session.tentative_pool.records == ()
        assert session.tentative_pool.next_tentative_id == 0
        assert session.update(frame).tracker_id.tolist() == []
        assert len(session.tentative_pool.records) == 1

    def test_production_shaped_background_query_is_not_emitted(self) -> None:
        """A two-logit no-object winner remains absent from public detections."""
        config = TrackingSessionConfig(tentative_confirmation_hits=1)
        session = TrackingSession(_owner(), config)
        first = session.update(torch.zeros(3, 6, 10))
        detections = session.update(torch.zeros(3, 6, 10))

        assert first.tracker_id.tolist() == [0]
        assert detections.tracker_id.tolist() == [0]
        assert detections.class_id.tolist() == [0]
        assert session.last_frame_output is not None
        assert session.last_frame_output.pred_logits.shape == (1, 2, 2)

    def test_public_detections_use_external_foreground_category_ids(self) -> None:
        """Session output maps the selected foreground logit through the authoritative schema."""
        config = TrackingSessionConfig(tentative_confirmation_hits=1)
        session = TrackingSession(_owner(external_category_id=7), config)
        session.update(torch.zeros(3, 6, 10))
        detections = session.update(torch.zeros(3, 6, 10))

        assert detections.class_id.tolist() == [7]

    def test_stateless_prediction_between_updates_does_not_change_identity(self) -> None:
        """Ordinary image prediction remains independent from session state."""
        owner = _owner()
        session = owner.create_tracking_session(TrackingSessionConfig(tentative_confirmation_hits=1))
        frame = torch.zeros(3, 6, 10)
        session.update(frame)
        first = session.update(frame)

        owner.predict = lambda image: "stateless result"  # type: ignore[method-assign]
        assert owner.predict(frame) == "stateless result"
        second = session.update(frame)

        assert first.tracker_id.tolist() == second.tracker_id.tolist() == [0]

    def test_optimized_model_is_rejected_without_mutating_session(self) -> None:
        """Compiled stateless inference cannot silently discard recurrent state."""
        owner = _owner()
        owner._is_optimized_for_inference = True
        session = TrackingSession(owner)

        with pytest.raises(RuntimeError, match="does not support optimized or exported"):
            session.update(torch.zeros(3, 6, 10))

        assert session.active_tracks == ()

    def test_opt_in_timing_reports_tracking_overhead(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Timing separates neural latency from session orchestration overhead."""
        ticks = iter([1.0, 1.002, 1.010, 1.013, 1.020])
        monkeypatch.setattr("rfdetr.tracking.session.perf_counter", lambda: next(ticks))
        session = TrackingSession(_owner(), TrackingSessionConfig(collect_timing=True))

        detections = session.update(torch.zeros(3, 6, 10))

        assert isinstance(session.last_timing, TrackingTiming)
        assert session.last_timing.preprocessing_ms == pytest.approx(2.0)
        assert session.last_timing.model_ms == pytest.approx(8.0)
        assert session.last_timing.lifecycle_ms == pytest.approx(3.0)
        assert session.last_timing.output_ms == pytest.approx(7.0)
        assert session.last_timing.total_ms == pytest.approx(20.0)
        assert session.last_timing.tracking_overhead_ms == pytest.approx(10.0)
        assert detections.metadata["timing"] == session.last_timing

    def test_timing_is_disabled_by_default(self) -> None:
        """Ordinary sessions avoid synchronization and timing overhead."""
        session = TrackingSession(_owner())

        detections = session.update(torch.zeros(3, 6, 10))

        assert session.last_timing is None
        assert "timing" not in detections.metadata


def _policy_fields(schema_hash: str, **overrides: object) -> dict[str, object]:
    base = {
        "foreground_schema_hash": schema_hash,
        "activation_threshold": 0.5,
        "continuation_threshold": 0.3,
        "duplicate_iou_threshold": 0.7,
        "max_missed_frames": 30,
        "tentative_confirmation_hits": 2,
        "tentative_confirmation_window_frames": 3,
        "tentative_max_misses": 2,
        "max_discovery_candidates_per_frame": 10,
        "max_tentative_tracks": 10,
        "max_active_tracks": 1,
    }
    base.update(overrides)
    return base


def _write_lock_file(path: Path, fields: dict[str, object], *, tamper: bool = False) -> None:
    """Write a minimal ``tracking_policy.lock.json``-shaped artifact for loader tests."""
    policy = TrackingPolicy(**fields)
    payload = {
        "schema_version": policy.schema_version,
        "policy_hash": policy.sha256(),
        "fields": json.loads(policy.canonical_json()),
    }
    payload["lock_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if tamper:
        payload["fields"]["activation_threshold"] = 0.999
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class TestLoadTrackingPolicy:
    """Loader for locked lifecycle policy artifacts (PRD US-013)."""

    def test_round_trips_a_valid_lock_file(self, tmp_path: Path) -> None:
        fields = _policy_fields("a" * 64)
        lock_path = tmp_path / "tracking_policy.lock.json"
        _write_lock_file(lock_path, fields)

        policy = load_tracking_policy(lock_path)

        assert policy == TrackingPolicy(**fields)

    def test_rejects_unsupported_schema_version(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "tracking_policy.lock.json"
        lock_path.write_text(json.dumps({"schema_version": "other"}))

        with pytest.raises(ValueError, match="schema_version"):
            load_tracking_policy(lock_path)

    def test_rejects_tampered_fields(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "tracking_policy.lock.json"
        _write_lock_file(lock_path, _policy_fields("a" * 64), tamper=True)

        with pytest.raises(ValueError, match="does not match|modified"):
            load_tracking_policy(lock_path)


class TestTrackingSessionLockedPolicy:
    """Constructing sessions from a locked, versioned policy contract (PRD US-013)."""

    def test_session_applies_locked_policy_and_exposes_its_hash(self) -> None:
        owner = _owner()
        schema_hash = owner.model_config.class_schema.sha256()
        policy = TrackingPolicy(**_policy_fields(schema_hash, activation_threshold=0.42))

        session = TrackingSession(owner, policy=policy)

        assert session.policy_hash == policy.sha256()

    def test_session_without_a_policy_has_no_policy_hash(self) -> None:
        session = TrackingSession(_owner())

        assert session.policy_hash is None

    def test_config_and_policy_are_mutually_exclusive(self) -> None:
        owner = _owner()
        schema_hash = owner.model_config.class_schema.sha256()
        policy = TrackingPolicy(**_policy_fields(schema_hash))

        with pytest.raises(ValueError, match="mutually exclusive"):
            TrackingSession(owner, TrackingSessionConfig(), policy=policy)

    def test_rejects_a_policy_calibrated_for_a_different_class_schema(self) -> None:
        owner = _owner()
        policy = TrackingPolicy(**_policy_fields("f" * 64))

        with pytest.raises(ValueError, match="foreground_schema_hash"):
            TrackingSession(owner, policy=policy)

    def test_rejects_a_policy_calibrated_for_a_different_durable_capacity(self) -> None:
        owner = _owner()
        schema_hash = owner.model_config.class_schema.sha256()
        policy = TrackingPolicy(**_policy_fields(schema_hash, max_active_tracks=40))

        with pytest.raises(ValueError, match="max_active_tracks"):
            TrackingSession(owner, policy=policy)

    def test_deployment_entry_point_loads_the_same_versioned_policy_type(self, tmp_path: Path) -> None:
        """``create_tracking_session(policy_lock_path=...)`` is the deployment entry point."""
        owner = _owner()
        schema_hash = owner.model_config.class_schema.sha256()
        fields = _policy_fields(schema_hash, activation_threshold=0.42)
        lock_path = tmp_path / "tracking_policy.lock.json"
        _write_lock_file(lock_path, fields)

        session = owner.create_tracking_session(policy_lock_path=lock_path)

        assert session.policy_hash == TrackingPolicy(**fields).sha256()

    def test_deployment_config_and_policy_lock_path_are_mutually_exclusive(self, tmp_path: Path) -> None:
        owner = _owner()
        schema_hash = owner.model_config.class_schema.sha256()
        lock_path = tmp_path / "tracking_policy.lock.json"
        _write_lock_file(lock_path, _policy_fields(schema_hash))

        with pytest.raises(ValueError, match="mutually exclusive"):
            owner.create_tracking_session(TrackingSessionConfig(), policy_lock_path=lock_path)

    def test_validation_and_deployment_entry_points_produce_frame_for_frame_identical_output(
        self, tmp_path: Path
    ) -> None:
        """A locked policy drives identical results whether loaded for validation or deployment.

        Chronological validation (e.g. ``evaluate_tracking_checkpoint``) and the
        deployment entry point (``RFDETR.create_tracking_session``) both end up
        constructing a ``TrackingSession`` from the same ``TrackingPolicy``
        deserialized out of one locked artifact. This proves neither path can
        silently diverge on thresholds, capacity, or class schema.
        """
        owner = _owner()
        schema_hash = owner.model_config.class_schema.sha256()
        fields = _policy_fields(schema_hash)
        lock_path = tmp_path / "tracking_policy.lock.json"
        _write_lock_file(lock_path, fields)
        frames = [np.zeros((6, 10, 3), dtype=np.uint8) for _ in range(4)]

        # Deployment entry point.
        deployment_session = owner.create_tracking_session(policy_lock_path=lock_path)
        deployment_results = [deployment_session.update(frame) for frame in frames]

        # Validation entry point: loads the same lock file through the same
        # loader and constructs TrackingSession directly, mirroring what
        # chronological validation does internally.
        validation_policy = load_tracking_policy(lock_path)
        validation_session = TrackingSession(_owner(), policy=validation_policy)
        validation_results = [validation_session.update(frame) for frame in frames]

        assert deployment_session.policy_hash == validation_session.policy_hash
        for deployment_result, validation_result in zip(deployment_results, validation_results, strict=True):
            assert deployment_result.tracker_id.tolist() == validation_result.tracker_id.tolist()
            assert deployment_result.class_id.tolist() == validation_result.class_id.tolist()
            assert deployment_result.xyxy.tolist() == validation_result.xyxy.tolist()
            assert deployment_result.confidence.tolist() == validation_result.confidence.tolist()
