# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torchvision
from torch.utils.data import Dataset, Subset

from rfdetr.datasets._keypoint_schema import infer_coco_keypoint_schema as infer_coco_keypoint_schema
from rfdetr.datasets._keypoint_schema import infer_yolo_keypoint_schema as infer_yolo_keypoint_schema
from rfdetr.datasets.coco import build_coco, build_roboflow_from_coco, make_coco_transforms
from rfdetr.datasets.o365 import build_o365
from rfdetr.datasets.video import SharedSequenceTransform as SharedSequenceTransform
from rfdetr.datasets.video import VideoClip as VideoClip
from rfdetr.datasets.video import VideoFrame as VideoFrame
from rfdetr.datasets.video import VideoObject as VideoObject
from rfdetr.datasets.video import VideoSequenceDataset as VideoSequenceDataset
from rfdetr.datasets.video import build_video_clip_index as build_video_clip_index
from rfdetr.datasets.video import build_video_validation_clip_index as build_video_validation_clip_index
from rfdetr.datasets.video import make_sequence_collate_fn as make_sequence_collate_fn
from rfdetr.datasets.video import sequence_collate_fn as sequence_collate_fn
from rfdetr.datasets.yolo import YoloDetection, build_roboflow_from_yolo


def _video_split_annotations(
    annotations: dict[str, Any],
    image_set: str,
    *,
    annotation_path: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    """Select one split from a shared COCO-video annotation artifact.

    New artifacts carry ``split`` on every image.  Older Clevis exports keep
    the video-to-split mapping in the adjacent ``manifest.json``; support that
    representation as well so existing datasets do not need rebuilding.
    """
    images = annotations.get("images")
    objects = annotations.get("annotations")
    if not isinstance(images, list) or not isinstance(objects, list):
        return annotations

    requested = "val" if image_set == "valid" else image_set
    explicit = [image.get("split") for image in images if isinstance(image, dict)]
    has_explicit_split = bool(explicit) and all(isinstance(value, str) for value in explicit)

    split_map: dict[str, str] = {}
    if not has_explicit_split:
        manifest_candidates = (annotation_path.parent / "manifest.json", dataset_root / "manifest.json")
        for manifest_path in dict.fromkeys(manifest_candidates):
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            candidate = manifest.get("splits") if isinstance(manifest, dict) else None
            if isinstance(candidate, dict) and all(
                isinstance(key, str) and isinstance(value, str) for key, value in candidate.items()
            ):
                split_map = candidate
                break

    def image_split(image: dict[str, Any]) -> str | None:
        value = image.get("split")
        if isinstance(value, str):
            return "val" if value == "valid" else value
        lineage = image.get("source_lineage")
        video_id = lineage.get("video_id") if isinstance(lineage, dict) else None
        value = split_map.get(video_id) if isinstance(video_id, str) else None
        return "val" if value == "valid" else value

    if not has_explicit_split and not split_map:
        return annotations

    selected_images = [
        image for image in images if isinstance(image, dict) and image_split(image) == requested
    ]
    selected_ids = {image.get("id") for image in selected_images}
    selected = dict(annotations)
    selected["images"] = selected_images
    selected["annotations"] = [
        annotation
        for annotation in objects
        if isinstance(annotation, dict) and annotation.get("image_id") in selected_ids
    ]
    return selected


def get_coco_api_from_dataset(dataset: Dataset[Any]) -> Any | None:
    for _ in range(10):
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    if isinstance(dataset, YoloDetection):
        return dataset.coco
    return None


def detect_roboflow_format(dataset_dir: Path) -> str:
    """Detect if a Roboflow dataset is in COCO or YOLO format.

    Args:
        dataset_dir: Path to the Roboflow dataset root directory

    Returns:
        'coco' if COCO format detected, 'yolo' if YOLO format detected

    Raises:
        ValueError: If neither format is detected
    """
    # Check for COCO format: look for _annotations.coco.json in train folder
    coco_annotation = dataset_dir / "train" / "_annotations.coco.json"
    if coco_annotation.exists():
        return "coco"

    # Check for YOLO format: look for data.yaml or data.yml and train/images folder
    yolo_data_file_yaml = dataset_dir / "data.yaml"
    yolo_data_file_yml = dataset_dir / "data.yml"
    yolo_images_dir = dataset_dir / "train" / "images"
    if (yolo_data_file_yaml.exists() or yolo_data_file_yml.exists()) and yolo_images_dir.exists():
        return "yolo"

    raise ValueError(
        f"Could not detect dataset format in {dataset_dir}. "
        f"Expected either COCO format (train/_annotations.coco.json) "
        f"or YOLO format (data.yaml or data.yml + train/images/)"
    )


def build_roboflow(image_set: str, args: Any, resolution: int) -> Dataset[Any]:
    """Build a Roboflow dataset, auto-detecting COCO or YOLO format.

    This function detects the dataset format and delegates to the appropriate builder function.
    """
    root = Path(args.dataset_dir)
    assert root.exists(), f"provided Roboflow path {root} does not exist"

    dataset_format = detect_roboflow_format(root)

    if dataset_format == "coco":
        return build_roboflow_from_coco(image_set, args, resolution)
    return build_roboflow_from_yolo(image_set, args, resolution)


def build_video(image_set: str, args: Any, resolution: int) -> VideoSequenceDataset:
    """Build a chronological video dataset from an on-disk COCO-video split.

    Args:
        image_set: Dataset split, normally ``"train"`` or ``"val"``.
        args: Combined model and training configuration namespace.
        resolution: Model input resolution.

    Returns:
        A clip-atomic sequence dataset with ordinary RF-DETR transforms.

    Raises:
        FileNotFoundError: If the dataset root or annotation file is missing.
        ValueError: If the annotation JSON is not an object or contains an
            invalid video schema.
    """
    root = Path(args.dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"video dataset path {root} does not exist")

    configured_path = args.tracking.annotation_path
    if configured_path is None:
        annotation_path = root / image_set / "_annotations.coco.json"
        image_root = root / image_set
    else:
        annotation_path = Path(configured_path)
        if not annotation_path.is_absolute():
            annotation_path = root / annotation_path
        image_root = root
    if not annotation_path.is_file():
        raise FileNotFoundError(f"video annotation file does not exist at {annotation_path}")

    with annotation_path.open(encoding="utf-8") as annotation_file:
        annotations = json.load(annotation_file)
    if not isinstance(annotations, dict):
        raise ValueError(f"video annotation file {annotation_path} must contain a JSON object")
    annotations = _video_split_annotations(
        annotations,
        image_set,
        annotation_path=annotation_path,
        dataset_root=root,
    )

    clip_length = args.tracking.clip_length
    clips = (
        build_video_clip_index(annotations, clip_length, args.tracking.clip_stride)
        if image_set == "train"
        else build_video_validation_clip_index(annotations, clip_length)
    )
    transform = make_coco_transforms(
        image_set=image_set,
        resolution=resolution,
        multi_scale=args.multi_scale,
        expanded_scales=args.expanded_scales,
        skip_random_resize=args.do_random_resize_via_padding,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        aug_config=args.aug_config,
        gpu_postprocess=False,
    )
    return VideoSequenceDataset(image_root, clips, transform)


def build_dataset(image_set: str, args: Any, resolution: int) -> Dataset[Any]:
    if args.dataset_file == "coco":
        return build_coco(image_set, args, resolution)
    if args.dataset_file == "o365":
        return build_o365(image_set, args, resolution)
    if args.dataset_file == "roboflow":
        return build_roboflow(image_set, args, resolution)
    if args.dataset_file == "yolo":
        return build_roboflow_from_yolo(image_set, args, resolution)
    if args.dataset_file == "video":
        return build_video(image_set, args, resolution)
    raise ValueError(f"dataset {args.dataset_file} not supported")
