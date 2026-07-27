# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Sequence-aware detection and association evaluation records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable, Iterable, Literal

if TYPE_CHECKING:
    from rfdetr.tracking.lifecycle import LifecycleEvent

DiagnosticKind = Literal[
    "birth",
    "recovery",
    "suspension",
    "termination",
    "duplicate",
    "capacity_suppression",
    "false_track_creation",
]

_DIAGNOSTIC_KINDS: dict[str, DiagnosticKind] = {
    "activated": "birth",
    "recovered": "recovery",
    "suspended": "suspension",
    "terminated": "termination",
    "duplicate_suppressed": "duplicate",
    "capacity_suppressed": "capacity_suppression",
}


@dataclass(frozen=True, slots=True)
class SequencePrediction:
    """One box prediction before sequence and frame provenance are attached.

    Args:
        box: Absolute ``xyxy`` box coordinates.
        class_id: Zero-based model class identifier.
        score: Finite confidence score.
        track_id: Non-negative identity local to one sequence.
    """

    box: tuple[float, float, float, float]
    class_id: int
    score: float
    track_id: int

    def __post_init__(self) -> None:
        """Validate detection geometry, confidence, class, and identity."""
        if len(self.box) != 4 or not all(math.isfinite(value) for value in self.box):
            raise ValueError("prediction box must contain four finite xyxy coordinates")
        x1, y1, x2, y2 = self.box
        if x2 < x1 or y2 < y1:
            raise ValueError("prediction box must have non-negative width and height")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int) or self.class_id < 0:
            raise ValueError("prediction class_id must be a non-negative integer")
        if not math.isfinite(self.score):
            raise ValueError("prediction score must be finite")
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int) or self.track_id < 0:
            raise ValueError("prediction track_id must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class SequenceEvaluationRecord:
    """One prediction with complete sequence and source-frame provenance."""

    sequence_id: Hashable
    frame_index: int
    box: tuple[float, float, float, float]
    class_id: int
    score: float
    track_id: int | None


@dataclass(frozen=True, slots=True)
class SequenceLifecycleDiagnostic:
    """One normalized lifecycle diagnostic scoped to a sequence."""

    sequence_id: Hashable
    frame_index: int
    kind: DiagnosticKind
    track_id: int | None
    slot: int | None


class SequenceEvaluationOutput:
    """Accumulate sequence predictions without erasing reset boundaries.

    Rich :attr:`records` retain class and tracking identity for association
    evaluation. :attr:`detection_records` deliberately removes identity so the
    same predictions can be scored by ordinary per-frame detection metrics.
    TrackEval export is grouped by sequence, preventing equal session-local IDs
    from being merged across resets.
    """

    def __init__(self) -> None:
        """Create an empty accumulator."""
        self._records: list[SequenceEvaluationRecord] = []
        self._diagnostics: list[SequenceLifecycleDiagnostic] = []
        self._frames: set[tuple[Hashable, int]] = set()
        self._sequences: set[Hashable] = set()
        self._reset_boundaries: list[tuple[Hashable, int]] = []

    @property
    def records(self) -> tuple[SequenceEvaluationRecord, ...]:
        """Return all identity-bearing predictions in insertion order."""
        return tuple(self._records)

    @property
    def detection_records(self) -> tuple[SequenceEvaluationRecord, ...]:
        """Return the same per-frame detections with association identity removed."""
        return tuple(
            SequenceEvaluationRecord(
                record.sequence_id,
                record.frame_index,
                record.box,
                record.class_id,
                record.score,
                None,
            )
            for record in self._records
        )

    @property
    def lifecycle_diagnostics(self) -> tuple[SequenceLifecycleDiagnostic, ...]:
        """Return normalized lifecycle and false-track diagnostics."""
        return tuple(self._diagnostics)

    @property
    def reset_boundaries(self) -> tuple[tuple[Hashable, int], ...]:
        """Return the first source frame following each sequence reset."""
        return tuple(self._reset_boundaries)

    def add_frame(
        self,
        *,
        sequence_id: Hashable,
        frame_index: int,
        predictions: Iterable[SequencePrediction],
        reset: bool = False,
        lifecycle_events: Iterable[LifecycleEvent] = (),
        matched_track_ids: set[int] | None = None,
    ) -> None:
        """Add one uniquely identified source frame.

        Args:
            sequence_id: Dataset sequence identity and TrackEval stream key.
            frame_index: Non-negative source-frame position.
            predictions: Identity-bearing detections for this frame.
            reset: Whether the caller explicitly reset before this frame.
            lifecycle_events: Host lifecycle events emitted for this frame.
            matched_track_ids: Optional predicted IDs matched to ground truth.
                An unmatched birth is additionally reported as false-track
                creation.

        Raises:
            ValueError: If provenance is invalid, a frame is duplicated, or a
                reset attempts to reuse an existing sequence namespace.
        """
        try:
            hash(sequence_id)
        except TypeError as error:
            raise ValueError("sequence_id must be hashable") from error
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        frame_key = (sequence_id, frame_index)
        if frame_key in self._frames:
            raise ValueError(f"sequence frame {frame_key!r} has already been added")
        if reset and sequence_id in self._sequences:
            raise ValueError("a reset must begin a new sequence identity namespace")
        if sequence_id not in self._sequences:
            self._reset_boundaries.append(frame_key)
            self._sequences.add(sequence_id)
        self._frames.add(frame_key)

        frame_predictions = tuple(predictions)
        track_ids = [prediction.track_id for prediction in frame_predictions]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("track IDs must be unique within one sequence frame")
        self._records.extend(
            SequenceEvaluationRecord(
                sequence_id,
                frame_index,
                prediction.box,
                prediction.class_id,
                prediction.score,
                prediction.track_id,
            )
            for prediction in frame_predictions
        )

        matched = matched_track_ids
        for event in lifecycle_events:
            if event.frame_index != frame_index:
                raise ValueError("lifecycle event frame_index must match the added frame")
            kind = _DIAGNOSTIC_KINDS[event.kind]
            self._diagnostics.append(
                SequenceLifecycleDiagnostic(sequence_id, frame_index, kind, event.track_id, event.slot)
            )
            if kind == "birth" and matched is not None and event.track_id not in matched:
                self._diagnostics.append(
                    SequenceLifecycleDiagnostic(
                        sequence_id,
                        frame_index,
                        "false_track_creation",
                        event.track_id,
                        event.slot,
                    )
                )

    def to_trackeval_mot(self) -> dict[Hashable, tuple[str, ...]]:
        """Return TrackEval-compatible MOTChallenge prediction rows by sequence.

        Source frame and session track IDs are converted from RF-DETR's
        zero-based convention to MOTChallenge's one-based convention. Boxes are
        converted from ``xyxy`` to ``xywh``. The final three placeholder fields
        follow the standard ten-column MOT prediction format.
        """
        rows: dict[Hashable, list[str]] = {sequence_id: [] for sequence_id in self._sequences}
        for record in sorted(
            self._records,
            key=lambda item: (type(item.sequence_id).__name__, str(item.sequence_id), item.frame_index, item.track_id),
        ):
            if record.track_id is None:
                continue
            x1, y1, x2, y2 = record.box
            values = (
                record.frame_index + 1,
                record.track_id + 1,
                x1,
                y1,
                x2 - x1,
                y2 - y1,
                record.score,
                -1,
                -1,
                -1,
            )
            rows[record.sequence_id].append(",".join(f"{value:g}" for value in values))
        return {sequence_id: tuple(sequence_rows) for sequence_id, sequence_rows in rows.items()}


__all__ = [
    "DiagnosticKind",
    "SequenceEvaluationOutput",
    "SequenceEvaluationRecord",
    "SequenceLifecycleDiagnostic",
    "SequencePrediction",
]
