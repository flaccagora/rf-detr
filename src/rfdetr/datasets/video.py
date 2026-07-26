# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Validated COCO-video metadata and deterministic clip indexing.

Chronology and identity are read only from annotation fields. In particular,
this module never parses a filename to determine either a sequence or a frame
position.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Sequence


@dataclass(frozen=True)
class VideoObject:
    """Sequence identity metadata for one annotated object.

    Args:
        annotation_id: Dataset-local annotation identifier, when supplied.
        category_id: Object category identifier.
        track_id: Sequence-local identity, or ``None`` when identity is unknown.
        identity_provenance: Non-empty source label such as ``human``,
            ``synthetic``, or ``pseudo``.
    """

    annotation_id: int | None
    category_id: int
    track_id: int | None
    identity_provenance: str


@dataclass(frozen=True)
class VideoFrame:
    """One explicitly positioned frame in a sequence."""

    image_id: int
    sequence_id: Hashable
    frame_index: int
    timestamp: float | None
    objects: tuple[VideoObject, ...]


@dataclass(frozen=True)
class VideoClip:
    """One chronological, fixed-length clip from a single sequence."""

    sequence_id: Hashable
    frames: tuple[VideoFrame, ...]

    @property
    def image_ids(self) -> tuple[int, ...]:
        """Return image identifiers in chronological order."""
        return tuple(frame.image_id for frame in self.frames)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        """Return explicit source-frame indices in chronological order."""
        return tuple(frame.frame_index for frame in self.frames)


def _required_alias(item: Mapping[str, Any], names: Sequence[str], context: str) -> Any:
    """Read one required field from an ordered list of schema aliases."""
    for name in names:
        if name in item:
            return item[name]
    aliases = " or ".join(repr(name) for name in names)
    raise ValueError(f"{context} requires explicit {aliases}")


def _integer(value: Any, field: str, context: str, *, allow_none: bool = False) -> int | None:
    """Validate an integer annotation field without accepting booleans."""
    if allow_none and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        suffix = " or null" if allow_none else ""
        raise ValueError(f"{context} {field} must be an integer{suffix}")
    return value


def _parse_object(annotation: Mapping[str, Any]) -> VideoObject:
    """Validate identity metadata from one COCO-style annotation."""
    context = f"annotation {annotation.get('id', '<unknown>')}"
    if "track_id" not in annotation:
        raise ValueError(f"{context} requires explicit 'track_id'; use null when identity is unknown")
    track_id = _integer(annotation["track_id"], "track_id", context, allow_none=True)
    if track_id is not None and track_id < 0:
        raise ValueError(f"{context} track_id must be non-negative or null")

    provenance = _required_alias(
        annotation,
        ("identity_provenance", "annotation_provenance"),
        context,
    )
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(f"{context} identity_provenance must be a non-empty string")

    category_id = _integer(annotation.get("category_id"), "category_id", context)
    annotation_id = _integer(annotation["id"], "id", context) if "id" in annotation else None
    return VideoObject(annotation_id, category_id, track_id, provenance)


def _parse_frame(
    image: Mapping[str, Any],
    annotations: Sequence[Mapping[str, Any]],
) -> VideoFrame:
    """Validate one image record and its aligned object identity metadata."""
    context = f"image {image.get('id', '<unknown>')}"
    image_id = _integer(image.get("id"), "id", context)
    sequence_id = _required_alias(image, ("sequence_id", "video_id"), context)
    if isinstance(sequence_id, bool) or not isinstance(sequence_id, (str, int)):
        raise ValueError(f"{context} sequence_id must be a string or integer")
    if isinstance(sequence_id, str) and not sequence_id:
        raise ValueError(f"{context} sequence_id must not be empty")
    frame_index = _integer(
        _required_alias(image, ("frame_index", "source_frame_index", "frame_id"), context),
        "frame_index",
        context,
    )
    if frame_index < 0:
        raise ValueError(f"{context} frame_index must be non-negative")

    timestamp_value = image.get("timestamp")
    if timestamp_value is None:
        timestamp = None
    elif isinstance(timestamp_value, bool) or not isinstance(timestamp_value, (int, float)):
        raise ValueError(f"{context} timestamp must be a finite number or null")
    else:
        timestamp = float(timestamp_value)
        if not math.isfinite(timestamp):
            raise ValueError(f"{context} timestamp must be a finite number or null")

    objects = tuple(_parse_object(annotation) for annotation in annotations)
    known_ids = [obj.track_id for obj in objects if obj.track_id is not None]
    if len(known_ids) != len(set(known_ids)):
        raise ValueError(f"{context} contains duplicate track_id values")
    return VideoFrame(image_id, sequence_id, frame_index, timestamp, objects)


def build_video_clip_index(
    annotations: Mapping[str, Any],
    clip_length: int,
    stride: int = 1,
) -> tuple[VideoClip, ...]:
    """Validate a COCO-style video schema and build chronological clip windows.

    Supported explicit aliases are ``sequence_id``/``video_id`` and
    ``frame_index``/``source_frame_index``/``frame_id``. Object annotations
    require a ``track_id`` key; JSON ``null`` means unknown identity and remains
    ``None``. Identity provenance is required as ``identity_provenance`` (or
    ``annotation_provenance``).

    Args:
        annotations: Mapping containing COCO-style ``images`` and
            ``annotations`` arrays.
        clip_length: Number of frames in every returned clip.
        stride: Distance between starts of adjacent windows in a sequence.

    Returns:
        Frozen clips grouped by sequence and sorted by explicit frame index.

    Raises:
        ValueError: If required metadata is absent, ambiguous, or inconsistent.
    """
    if isinstance(clip_length, bool) or not isinstance(clip_length, int) or clip_length < 1:
        raise ValueError("clip_length must be a positive integer")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")

    images = annotations.get("images")
    objects = annotations.get("annotations")
    if not isinstance(images, list) or not isinstance(objects, list):
        raise ValueError("video annotations require 'images' and 'annotations' arrays")

    annotations_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for annotation in objects:
        if not isinstance(annotation, Mapping):
            raise ValueError("every annotation must be an object")
        context = f"annotation {annotation.get('id', '<unknown>')}"
        image_id = _integer(annotation.get("image_id"), "image_id", context)
        annotations_by_image[image_id].append(annotation)

    frames_by_sequence: dict[Hashable, list[VideoFrame]] = defaultdict(list)
    image_ids: set[int] = set()
    for image in images:
        if not isinstance(image, Mapping):
            raise ValueError("every image must be an object")
        image_id = _integer(image.get("id"), "id", "image")
        if image_id in image_ids:
            raise ValueError(f"duplicate image id {image_id}")
        image_ids.add(image_id)
        frame = _parse_frame(image, annotations_by_image.pop(image_id, ()))
        frames_by_sequence[frame.sequence_id].append(frame)
    if annotations_by_image:
        unknown_id = next(iter(annotations_by_image))
        raise ValueError(f"annotation references unknown image_id {unknown_id}")

    clips: list[VideoClip] = []
    for sequence_id in sorted(frames_by_sequence, key=lambda value: (type(value).__name__, str(value))):
        frames = sorted(frames_by_sequence[sequence_id], key=lambda frame: frame.frame_index)
        frame_indices = [frame.frame_index for frame in frames]
        if len(frame_indices) != len(set(frame_indices)):
            raise ValueError(f"sequence {sequence_id!r} has duplicate frame_index values")

        categories_by_track: dict[int, set[int]] = defaultdict(set)
        for frame in frames:
            for obj in frame.objects:
                if obj.track_id is not None:
                    categories_by_track[obj.track_id].add(obj.category_id)
        for track_id, categories in categories_by_track.items():
            if len(categories) > 1:
                raise ValueError(
                    f"sequence {sequence_id!r} track_id {track_id} appears in multiple categories: {sorted(categories)}"
                )

        for start in range(0, len(frames) - clip_length + 1, stride):
            clips.append(VideoClip(sequence_id, tuple(frames[start : start + clip_length])))
    return tuple(clips)


# Descriptive aliases for callers using COCO-video terminology.
CocoVideoClip = VideoClip
build_coco_video_clip_index = build_video_clip_index
