# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Behavioral tests for sequence-aware tracking evaluation output."""

import pytest

from rfdetr.evaluation.sequence import SequenceEvaluationOutput, SequencePrediction
from rfdetr.tracking.lifecycle import LifecycleEvent


def test_sequence_output_preserves_detection_and_identity_fields() -> None:
    """Rich records should retain every field needed by detection and association metrics."""
    output = SequenceEvaluationOutput()

    output.add_frame(
        sequence_id="clip-a",
        frame_index=7,
        predictions=[SequencePrediction((10.0, 20.0, 30.0, 50.0), 2, 0.75, 0)],
    )

    assert output.records[0].sequence_id == "clip-a"
    assert output.records[0].frame_index == 7
    assert output.records[0].box == (10.0, 20.0, 30.0, 50.0)
    assert output.records[0].class_id == 2
    assert output.records[0].score == pytest.approx(0.75)
    assert output.records[0].track_id == 0
    assert output.detection_records[0].track_id is None


def test_mot_export_keeps_sequence_reset_boundaries() -> None:
    """Equal local IDs in different sequences must be exported to different TrackEval streams."""
    output = SequenceEvaluationOutput()
    prediction = SequencePrediction((10.0, 20.0, 30.0, 50.0), 2, 0.75, 0)
    output.add_frame(sequence_id="clip-a", frame_index=0, predictions=[prediction], reset=True)
    output.add_frame(sequence_id="clip-b", frame_index=0, predictions=[prediction], reset=True)

    records = output.to_trackeval_mot()

    assert set(records) == {"clip-a", "clip-b"}
    assert records["clip-a"] == ("1,1,10,20,20,30,0.75,-1,-1,-1",)
    assert records["clip-b"] == records["clip-a"]
    assert output.reset_boundaries == (("clip-a", 0), ("clip-b", 0))


def test_lifecycle_diagnostics_include_policy_and_false_track_events() -> None:
    """Sequence output should expose lifecycle policy events and false-track creation separately."""
    output = SequenceEvaluationOutput()
    output.add_frame(
        sequence_id="clip-a",
        frame_index=3,
        predictions=[SequencePrediction((0.0, 0.0, 10.0, 10.0), 1, 0.9, 4)],
        lifecycle_events=(
            LifecycleEvent("activated", slot=2, track_id=4, frame_index=3),
            LifecycleEvent(
                "duplicate_suppressed",
                slot=5,
                track_id=None,
                frame_index=3,
                reason="duplicate_overlap",
                score=0.8,
                class_id=1,
                compared_status="active",
                overlap=0.9,
            ),
        ),
        matched_track_ids=set(),
    )

    assert [event.kind for event in output.lifecycle_diagnostics] == [
        "birth",
        "false_track_creation",
        "duplicate",
    ]


@pytest.mark.parametrize(
    "event_kind,diagnostic_kind",
    [
        pytest.param("recovered", "recovery", id="recovery"),
        pytest.param("suspended", "suspension", id="suspension"),
        pytest.param("terminated", "termination", id="termination"),
    ],
)
def test_lifecycle_state_changes_have_evaluation_names(event_kind: str, diagnostic_kind: str) -> None:
    """Recovery, suspension, and termination should remain individually measurable."""
    output = SequenceEvaluationOutput()
    output.add_frame(
        sequence_id="clip-a",
        frame_index=3,
        predictions=[],
        lifecycle_events=(LifecycleEvent(event_kind, slot=2, track_id=4, frame_index=3),),
    )

    assert output.lifecycle_diagnostics[0].kind == diagnostic_kind


def test_frame_identity_must_be_unique_within_sequence() -> None:
    """Overlapping clips must not silently duplicate one evaluated source frame."""
    output = SequenceEvaluationOutput()
    output.add_frame(sequence_id=9, frame_index=1, predictions=[])

    with pytest.raises(ValueError, match="already been added"):
        output.add_frame(sequence_id=9, frame_index=1, predictions=[])
