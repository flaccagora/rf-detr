# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr.config import TrackingConfig, TrackingSessionConfig
from rfdetr.detr import RFDETR
from rfdetr.models.tracking import TrackingFrameOutput, TrackQueryState
from rfdetr.tracking import TrackingSession, TrackingTiming


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
            pred_logits=torch.tensor([[[5.0], [-5.0]]], device=samples.device),
            pred_boxes=boxes,
            candidate_state=TrackQueryState(
                torch.ones(1, 2, 3, device=samples.device),
                boxes,
                torch.ones(1, 2, dtype=torch.bool, device=samples.device),
            ),
            input_active_mask=prior_state.active_mask,
        )


def _owner() -> RFDETR:
    """Build a weight-free RFDETR owner around the deterministic module."""
    owner = object.__new__(RFDETR)
    module = _TrackingModule()
    owner.model_config = SimpleNamespace(
        num_channels=3,
        num_queries=2,
        hidden_dim=3,
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
        first = owner.create_tracking_session()
        second = owner.create_tracking_session()
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
        session = TrackingSession(_owner(), TrackingSessionConfig())

        detections = session.update(frame, frame_index=4, timestamp=1.5)

        assert detections.tracker_id.tolist() == [0]
        assert detections.class_id.tolist() == [0]
        assert detections.xyxy.tolist() == [[2.5, 1.5, 7.5, 4.5]]
        assert detections.metadata["frame_index"] == 4
        assert detections.metadata["timestamp"] == 1.5

    def test_stateless_prediction_between_updates_does_not_change_identity(self) -> None:
        """Ordinary image prediction remains independent from session state."""
        owner = _owner()
        session = owner.create_tracking_session()
        frame = torch.zeros(3, 6, 10)
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
