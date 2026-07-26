# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Behavioral tests for explicit video annotation metadata and clip indexing."""

from __future__ import annotations

import pytest

from rfdetr.datasets.video import build_video_clip_index


def _annotations() -> dict[str, list[dict[str, object]]]:
    return {
        "images": [
            {"id": 30, "file_name": "arbitrary-c.png", "sequence_id": "case-b", "frame_index": 8},
            {"id": 11, "file_name": "z.png", "sequence_id": "case-a", "frame_index": 4},
            {"id": 10, "file_name": "a.png", "sequence_id": "case-a", "frame_index": 1},
            {"id": 31, "file_name": "arbitrary-a.png", "sequence_id": "case-b", "frame_index": 3},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 10,
                "category_id": 0,
                "track_id": 7,
                "identity_provenance": "human",
            },
            {
                "id": 2,
                "image_id": 11,
                "category_id": 0,
                "track_id": 7,
                "identity_provenance": "human",
            },
            {
                "id": 3,
                "image_id": 30,
                "category_id": 0,
                "track_id": None,
                "identity_provenance": "pseudo",
            },
        ],
    }


def test_clip_index_uses_explicit_sequence_and_frame_metadata() -> None:
    """Filename and JSON order must not influence chronological clip order."""
    clips = build_video_clip_index(_annotations(), clip_length=2)

    assert [clip.sequence_id for clip in clips] == ["case-a", "case-b"]
    assert [clip.image_ids for clip in clips] == [(10, 11), (31, 30)]
    assert [clip.frame_indices for clip in clips] == [(1, 4), (3, 8)]


def test_clip_index_preserves_unknown_identity() -> None:
    """Missing identity knowledge is explicit rather than synthesized per frame."""
    clips = build_video_clip_index(_annotations(), clip_length=2)

    assert clips[1].frames[1].objects[0].track_id is None
    assert clips[1].frames[1].objects[0].identity_provenance == "pseudo"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data["images"][0].pop("sequence_id"), "sequence_id"),
        (lambda data: data["images"][0].pop("frame_index"), "frame_index"),
        (lambda data: data["annotations"][0].pop("track_id"), "track_id"),
        (
            lambda data: data["annotations"][0].pop("identity_provenance"),
            "identity_provenance",
        ),
    ],
)
def test_clip_index_rejects_implicit_video_metadata(change, message: str) -> None:
    """Required chronology and identity fields may not be inferred."""
    data = _annotations()
    change(data)

    with pytest.raises(ValueError, match=message):
        build_video_clip_index(data, clip_length=2)


def test_clip_index_rejects_duplicate_frame_positions() -> None:
    """A sequence cannot have two ambiguous frames at the same source position."""
    data = _annotations()
    data["images"][1]["frame_index"] = 1

    with pytest.raises(ValueError, match="duplicate frame_index"):
        build_video_clip_index(data, clip_length=2)


def test_clip_index_rejects_identity_class_changes() -> None:
    """One sequence-local identity must keep one category."""
    data = _annotations()
    data["annotations"][1]["category_id"] = 1

    with pytest.raises(ValueError, match="track_id 7.*multiple categories"):
        build_video_clip_index(data, clip_length=2)


def test_clip_index_uses_sliding_windows_without_crossing_sequences() -> None:
    """Clip stride creates windows only inside a sequence."""
    data = _annotations()
    data["images"].insert(
        2,
        {"id": 12, "file_name": "ignored.png", "sequence_id": "case-a", "frame_index": 9},
    )

    clips = build_video_clip_index(data, clip_length=2, stride=1)

    assert [clip.image_ids for clip in clips] == [(10, 11), (11, 12), (31, 30)]
