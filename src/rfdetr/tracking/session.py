# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public single-stream orchestration for persistent-query tracking."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torchvision.transforms.functional as F  # noqa: N812
from PIL import Image

from rfdetr.config import TrackingSessionConfig
from rfdetr.models.tracking import TrackQueryState
from rfdetr.tracking.lifecycle import LifecycleEvent, TrackSlot, TrackSlotTable, transition_lifecycle

if TYPE_CHECKING:
    from supervision import Detections

    from rfdetr.detr import RFDETR


class TrackingSession:
    """Own recurrent state for one logical video stream.

    The associated :class:`RFDETR` instance remains stateless and may therefore
    be shared by multiple sessions or used for ordinary ``predict()`` calls.
    Optimized/export inference is deliberately unsupported until those paths
    expose explicit recurrent state inputs and outputs.
    """

    def __init__(self, model: RFDETR, config: TrackingSessionConfig | None = None) -> None:
        """Create an empty tracking session.

        Args:
            model: Public RF-DETR model associated with this stream.
            config: Optional lifecycle thresholds and expiry policy.
        """
        if not model.model_config.tracking.enabled:
            raise ValueError("TrackingSession requires model_config.tracking.enabled=True")
        self._model = model
        self._config = config or TrackingSessionConfig()
        self._table = TrackSlotTable.empty(model.model_config.num_queries)
        self._state: TrackQueryState | None = None
        self._next_frame_index = 0
        self._last_events: tuple[LifecycleEvent, ...] = ()

    @property
    def active_tracks(self) -> tuple[TrackSlot, ...]:
        """Return an immutable snapshot of active and suspended track metadata."""
        return tuple(slot for slot in self._table.slots if slot.status != "inactive")

    @property
    def last_events(self) -> tuple[LifecycleEvent, ...]:
        """Return lifecycle diagnostics emitted by the most recent update."""
        return self._last_events

    def reset(self) -> None:
        """Clear neural and host state and restart session-local IDs at zero."""
        self._table = TrackSlotTable.empty(self._model.model_config.num_queries)
        self._state = None
        self._next_frame_index = 0
        self._last_events = ()

    @torch.inference_mode()
    def update(
        self,
        frame: str | Image.Image | np.ndarray | torch.Tensor,
        frame_index: int | None = None,
        timestamp: float | None = None,
    ) -> Detections:
        """Process one frame and return visible detections with persistent IDs.

        Args:
            frame: One image accepted by ``predict()``: path/URL, PIL image,
                HWC NumPy array, or normalized CHW tensor.
            frame_index: Optional monotonically increasing source-frame index.
                Omitted indices advance by one from the previous update.
            timestamp: Optional source timestamp copied to result metadata.

        Returns:
            A Supervision ``Detections`` object whose ``tracker_id`` entries
            align with the retained visible tracks.
        """
        from supervision import Detections

        if self._model._is_optimized_for_inference:
            raise RuntimeError(
                "TrackingSession does not support optimized or exported inference; use the eager PyTorch model."
            )
        from rfdetr.detr import _move_model_context_to_device

        _move_model_context_to_device(self._model.model)
        module = self._model.model.model
        if not hasattr(module, "forward_tracking"):
            raise RuntimeError("This model does not expose the explicit-state forward_tracking capability.")
        module.eval()

        tensor, original_size = self._prepare_frame(frame)
        if frame_index is None:
            frame_index = self._next_frame_index
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise TypeError("frame_index must be an integer")

        if self._state is None:
            parameter = next(module.parameters(), None)
            dtype = parameter.dtype if parameter is not None and parameter.is_floating_point() else tensor.dtype
            hidden_dim = self._model.model_config.hidden_dim
            self._state = TrackQueryState.empty(
                1,
                self._model.model_config.num_queries,
                hidden_dim,
                device=self._model.model.device,
                dtype=dtype,
            )

        tensor = tensor.to(dtype=self._state.query_features.dtype)
        frame_output = module.forward_tracking(tensor, self._state)
        transition = transition_lifecycle(
            self._table,
            self._state,
            frame_output,
            self._config,
            max_active_tracks=self._model.model_config.tracking.active_capacity(self._model.model_config.num_queries),
            frame_index=frame_index,
        )
        self._table = transition.table
        self._state = transition.state
        self._last_events = transition.events
        self._next_frame_index = frame_index + 1

        visible_slots = [index for index, slot in enumerate(self._table.slots) if slot.status == "active"]
        height, width = original_size
        if visible_slots:
            boxes = self._state.reference_boxes[0, visible_slots]
            cx, cy, box_width, box_height = boxes.unbind(-1)
            xyxy = torch.stack(
                (
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    (cx + box_width / 2) * width,
                    (cy + box_height / 2) * height,
                ),
                dim=-1,
            )
            slots = [self._table.slots[index] for index in visible_slots]
            detections = Detections(
                xyxy=xyxy.float().cpu().numpy(),
                confidence=np.asarray([slot.confidence for slot in slots], dtype=np.float32),
                class_id=np.asarray([slot.class_id for slot in slots], dtype=np.int64),
                tracker_id=np.asarray([slot.track_id for slot in slots], dtype=np.int64),
            )
        else:
            detections = Detections.empty()
            detections.tracker_id = np.empty(0, dtype=np.int64)
        detections.metadata["frame_index"] = frame_index
        detections.metadata["timestamp"] = timestamp
        detections.metadata["lifecycle_events"] = self._last_events
        return detections

    def _prepare_frame(
        self, frame: str | Image.Image | np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Load, validate, resize, and normalize one public image input."""
        image: Any = frame
        if isinstance(image, str):
            if urlparse(image).scheme in ("http", "https"):
                response = requests.get(image, timeout=30)
                response.raise_for_status()
                image = io.BytesIO(response.content)
            image = Image.open(image)
        if not isinstance(image, torch.Tensor):
            if isinstance(image, np.ndarray):
                image = F.to_tensor(image)
            elif isinstance(image, Image.Image):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image = F.to_tensor(image)
            else:
                raise TypeError("frame must be a path/URL, PIL image, NumPy array, or torch tensor")
        if image.ndim != 3 or image.shape[0] != self._model.model_config.num_channels:
            raise ValueError("Tensor frames must have CHW shape with channels matching the model configuration")
        if not image.is_floating_point():
            image = image.float() / 255
        if not torch.isfinite(image).all() or ((image < 0) | (image > 1)).any():
            raise ValueError("frame pixel values must be finite and normalized to [0, 1]")
        original_size = (int(image.shape[1]), int(image.shape[2]))
        image = image.to(self._model.model.device)
        image = F.resize(image, [self._model.model.resolution, self._model.model.resolution])
        image = F.normalize(image, self._model.means, self._model.stds).unsqueeze(0)
        return image, original_size


__all__ = ["TrackingSession"]
