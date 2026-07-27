# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Behavioral tests for explicit video annotation metadata and clip indexing."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from albumentations import Crop, HorizontalFlip, Resize
from PIL import Image
from torchvision.transforms.v2 import Compose

from rfdetr.datasets.coco import make_coco_transforms
from rfdetr.datasets.transforms import AlbumentationsWrapper
from rfdetr.datasets.video import (
    SharedSequenceTransform,
    VideoSequenceDataset,
    build_video_clip_index,
    sequence_collate_fn,
)


def _annotations() -> dict[str, list[dict[str, object]]]:
    return {
        "images": [
            {
                "id": 30,
                "file_name": "arbitrary-c.png",
                "width": 6,
                "height": 4,
                "sequence_id": "case-b",
                "frame_index": 8,
                "timestamp": 0.8,
            },
            {"id": 11, "file_name": "z.png", "width": 6, "height": 4, "sequence_id": "case-a", "frame_index": 4},
            {"id": 10, "file_name": "a.png", "width": 6, "height": 4, "sequence_id": "case-a", "frame_index": 1},
            {
                "id": 31,
                "file_name": "arbitrary-a.png",
                "width": 6,
                "height": 4,
                "sequence_id": "case-b",
                "frame_index": 3,
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 10,
                "category_id": 0,
                "bbox": [1, 1, 2, 2],
                "track_id": 7,
                "identity_provenance": "human",
            },
            {
                "id": 2,
                "image_id": 11,
                "category_id": 0,
                "bbox": [2, 1, 2, 2],
                "track_id": 7,
                "identity_provenance": "human",
            },
            {
                "id": 3,
                "image_id": 30,
                "category_id": 0,
                "bbox": [0, 0, 1, 1],
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
        {
            "id": 12,
            "file_name": "ignored.png",
            "width": 6,
            "height": 4,
            "sequence_id": "case-a",
            "frame_index": 9,
        },
    )

    clips = build_video_clip_index(data, clip_length=2, stride=1)

    assert [clip.image_ids for clip in clips] == [(10, 11), (11, 12), (31, 30)]


def test_video_dataset_loads_a_chronological_clip_with_aligned_targets(tmp_path: Path) -> None:
    """A dataset item loads index-selected paths and preserves all recurrent metadata."""
    data = _annotations()
    data["annotations"][2]["track_id"] = None
    for image_record, value in zip(data["images"], (30, 11, 10, 31)):
        Image.new("RGB", (6, 4), (value, 0, 0)).save(tmp_path / str(image_record["file_name"]))
    dataset = VideoSequenceDataset(tmp_path, build_video_clip_index(data, clip_length=2))

    images, targets = dataset[1]

    assert len(images) == len(targets) == 2
    assert [target["image_id"].item() for target in targets] == [31, 30]
    assert [target["frame_index"] for target in targets] == [3, 8]
    assert [target["sequence_id"] for target in targets] == ["case-b", "case-b"]
    assert targets[1]["timestamp"] == pytest.approx(0.8)
    assert targets[1]["track_ids"] == [None]
    assert targets[1]["identity_provenance"] == ("pseudo",)
    assert targets[1]["boxes"].tolist() == [[0.0, 0.0, 1.0, 1.0]]
    assert targets[1]["labels"].tolist() == [0]
    assert targets[1]["orig_size"].tolist() == [4, 6]
    assert images[0].getpixel((0, 0))[0] == 31


def test_video_dataset_reports_the_missing_indexed_image(tmp_path: Path) -> None:
    """Missing files identify the image and source frame that failed."""
    clips = build_video_clip_index(_annotations(), clip_length=2)

    with pytest.raises(FileNotFoundError, match=r"image 10.*frame 1.*a\.png"):
        VideoSequenceDataset(tmp_path, clips)[0]


def test_video_dataset_rejects_decoded_dimension_mismatch(tmp_path: Path) -> None:
    """Manifest dimensions must agree with the explicitly selected file."""
    data = _annotations()
    Image.new("RGB", (5, 4)).save(tmp_path / "a.png")
    clips = build_video_clip_index(data, clip_length=2)

    with pytest.raises(ValueError, match=r"image 10 dimensions.*do not match annotation"):
        VideoSequenceDataset(tmp_path, clips)[0]


@pytest.mark.parametrize(
    "bbox",
    [
        pytest.param([1, 1, 0, 2], id="zero-width"),
        pytest.param([5, 1, 2, 2], id="outside-image"),
        pytest.param([1, 1, float("nan"), 2], id="non-finite"),
    ],
)
def test_clip_index_rejects_invalid_boxes(bbox: list[float]) -> None:
    """Invalid geometry fails at indexing rather than entering training."""
    data = _annotations()
    data["annotations"][0]["bbox"] = bbox

    with pytest.raises(ValueError, match="bbox"):
        build_video_clip_index(data, clip_length=2)


def test_shared_sequence_transform_reuses_random_parameters_across_frames() -> None:
    """Every frame in a clip receives the same sampled spatial transform."""

    def transform(image: torch.Tensor, target: dict[str, torch.Tensor]):
        offset = random.random() + float(np.random.random()) + float(torch.rand(()))
        return image + offset, {**target, "boxes": target["boxes"] + offset}

    images = (torch.zeros(3, 2, 2), torch.ones(3, 2, 2))
    targets = ({"boxes": torch.zeros(1, 4)}, {"boxes": torch.ones(1, 4)})

    transformed_images, transformed_targets = SharedSequenceTransform(transform)(images, targets)

    assert torch.allclose(transformed_images[1] - transformed_images[0], torch.ones(3, 2, 2))
    assert torch.allclose(transformed_targets[1]["boxes"] - transformed_targets[0]["boxes"], torch.ones(1, 4))


def test_shared_sequence_transform_replays_transform_owned_rng() -> None:
    """Transform objects with private generators also share clip geometry."""

    class StatefulTransform:
        def __init__(self) -> None:
            self.generator = random.Random(9)

        def __call__(self, image, target):
            offset = self.generator.random()
            return image + offset, target

    images, _ = SharedSequenceTransform(StatefulTransform())(
        (torch.zeros(1), torch.ones(1)),
        ({}, {}),
    )

    assert torch.allclose(images[1] - images[0], torch.ones(1))


def test_shared_sequence_transform_applies_the_same_horizontal_flip_to_every_frame() -> None:
    """A spatial flip keeps corresponding trajectories geometrically coherent."""
    transform = SharedSequenceTransform(Compose([AlbumentationsWrapper(HorizontalFlip(p=1.0))]))
    images = (Image.new("RGB", (10, 6)), Image.new("RGB", (10, 6)))
    targets = (
        {"boxes": torch.tensor([[1.0, 1.0, 4.0, 5.0]]), "labels": torch.tensor([2])},
        {"boxes": torch.tensor([[2.0, 1.0, 5.0, 5.0]]), "labels": torch.tensor([2])},
    )

    _, transformed_targets = transform(images, targets)

    assert torch.allclose(transformed_targets[0]["boxes"], torch.tensor([[6.0, 1.0, 9.0, 5.0]]))
    assert torch.allclose(transformed_targets[1]["boxes"], torch.tensor([[5.0, 1.0, 8.0, 5.0]]))


def test_resize_and_crop_filter_every_instance_aligned_video_field_and_unknown_identity() -> None:
    """Objects removed after resize/crop cannot leave labels or identity metadata behind."""
    transform = SharedSequenceTransform(
        Compose(
            [
                AlbumentationsWrapper(Resize(height=12, width=20, p=1.0)),
                AlbumentationsWrapper(Crop(x_min=0, y_min=0, x_max=10, y_max=12, p=1.0)),
            ]
        )
    )
    target = {
        "boxes": torch.tensor([[1.0, 1.0, 4.0, 5.0], [6.0, 1.0, 9.0, 5.0]]),
        "labels": torch.tensor([2, 3]),
        "track_ids": [7, None],
        "identity_provenance": ("human", "pseudo"),
        "confidence": torch.tensor([0.9, 0.4]),
        "image_id": torch.tensor(10),
    }

    _, transformed_targets = transform((Image.new("RGB", (10, 6)),), (target,))

    transformed = transformed_targets[0]
    assert torch.allclose(transformed["boxes"], torch.tensor([[2.0, 2.0, 8.0, 10.0]]))
    assert transformed["labels"].tolist() == [2]
    assert transformed["track_ids"] == [7]
    assert transformed["identity_provenance"] == ["human"]
    assert transformed["confidence"].tolist() == pytest.approx([0.9])


def test_crop_preserves_valid_empty_frames() -> None:
    """Frames without detections remain valid after a shared spatial transform."""
    transform = SharedSequenceTransform(
        Compose([AlbumentationsWrapper(Crop(x_min=0, y_min=0, x_max=5, y_max=6, p=1.0))])
    )
    target = {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty((0,), dtype=torch.long),
        "track_ids": [],
        "identity_provenance": (),
    }

    images, targets = transform((Image.new("RGB", (10, 6)),), (target,))

    assert images[0].size == (5, 6)
    assert targets[0]["boxes"].shape == (0, 4)
    assert targets[0]["labels"].shape == (0,)
    assert targets[0]["track_ids"] == []


def test_photometric_randomness_is_frame_local() -> None:
    """Pixel-level augmentation draws independently from shared spatial decisions."""

    class PhotometricTransform:
        _is_geometric = False

        def __init__(self) -> None:
            self.generator = random.Random(12)

        def __call__(self, image, target):
            return image + self.generator.random(), target

    transform = SharedSequenceTransform(Compose([PhotometricTransform()]))

    transformed_images, _ = transform((torch.zeros(1), torch.zeros(1)), ({}, {}))

    assert not torch.equal(transformed_images[0], transformed_images[1])


def test_ordinary_image_transform_pipeline_is_not_wrapped_or_changed() -> None:
    """Only the video dataset wraps the existing ordinary image pipeline."""
    transform = make_coco_transforms("val_speed", resolution=32)
    video_dataset = VideoSequenceDataset(".", (), transform)

    assert isinstance(transform, Compose)
    assert not isinstance(transform, SharedSequenceTransform)
    assert isinstance(video_dataset.transform, SharedSequenceTransform)
    assert video_dataset.transform.transform is transform


def test_sequence_collation_is_time_major_for_backbone_reuse() -> None:
    """A clip batch becomes one ordinary nested image batch per chronological step."""
    batch = [
        (
            (torch.full((3, 2, 3), 10.0), torch.full((3, 2, 2), 11.0)),
            ({"frame_index": 10}, {"frame_index": 11}),
        ),
        (
            (torch.full((3, 1, 2), 20.0), torch.full((3, 2, 4), 21.0)),
            ({"frame_index": 20}, {"frame_index": 21}),
        ),
    ]

    frames, targets = sequence_collate_fn(batch, block_size=2)

    assert len(frames) == 2
    assert frames[0].tensors.shape == (2, 3, 2, 4)
    assert frames[1].tensors.shape == (2, 3, 2, 4)
    assert [target["frame_index"] for target in targets[0]] == [10, 20]
    assert [target["frame_index"] for target in targets[1]] == [11, 21]
