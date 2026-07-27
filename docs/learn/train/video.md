---
description: Fine-tune an RF-DETR detection model on chronological COCO-video clips for persistent-query tracking.
---

# Train RF-DETR for Video Tracking

Temporal fine-tuning teaches RF-DETR to carry object queries between chronological frames. Use it when you need stable
object identities from the persistent-query [tracking interface](../run/tracking.md), rather than independent detections
from each frame.

!!! warning "Current limitations"

    Video training currently supports detection models only. It requires `group_detr=1`, CPU augmentation
    (`augmentation_backend="cpu"`), an explicit integer `batch_size`, and clips containing at least two frames.
    Segmentation and keypoint models, `batch_size="auto"`, and the `"auto"` and `"gpu"` augmentation backends are not
    supported. Tracking inference requires eager PyTorch; optimized and exported models do not expose recurrent state.

## Arrange the dataset

Use one COCO-video annotation file in each split. With the default split convention, RF-DETR resolves both annotations
and frame paths beneath the split directory:

```text
my-video-dataset/
├── train/
│   ├── _annotations.coco.json
│   └── frames/
│       ├── 000000.jpg
│       └── 000001.jpg
└── val/
    ├── _annotations.coco.json
    └── frames/
        ├── 000000.jpg
        └── 000001.jpg
```

The JSON retains ordinary COCO `images`, `annotations`, and `categories`, and adds explicit chronology and identity:

```json
{
  "images": [
    {
      "id": 10,
      "file_name": "frames/000000.jpg",
      "width": 1280,
      "height": 720,
      "sequence_id": "procedure-001",
      "frame_index": 0,
      "timestamp": 0.0
    },
    {
      "id": 11,
      "file_name": "frames/000001.jpg",
      "width": 1280,
      "height": 720,
      "sequence_id": "procedure-001",
      "frame_index": 1,
      "timestamp": 0.033
    }
  ],
  "annotations": [
    {
      "id": 100,
      "image_id": 10,
      "category_id": 0,
      "bbox": [120, 80, 40, 90],
      "track_id": 7,
      "identity_provenance": "human"
    },
    {
      "id": 101,
      "image_id": 11,
      "category_id": 0,
      "bbox": [124, 82, 40, 90],
      "track_id": 7,
      "identity_provenance": "human"
    }
  ],
  "categories": [{"id": 0, "name": "tool"}]
}
```

`bbox` uses COCO `[x, y, width, height]` pixel coordinates and must lie inside the declared image dimensions.
`timestamp` is optional. Chronology never comes from filenames: every image must declare a non-negative `frame_index`
and a string or integer `sequence_id`. The aliases `video_id` and `source_frame_index`/`frame_id` are also accepted.

Follow these identity rules:

- Include `track_id` on every annotation. Use JSON `null` when identity is unknown; do not invent a sentinel identity.
- Use non-negative track IDs. IDs are local to a sequence, so another sequence may reuse the same number.
- A known track ID may appear at most once per frame and must keep the same `category_id` throughout its sequence.
- Include a non-empty `identity_provenance` on every annotation, such as `human`, `pseudo`, or `synthetic`.

Frames are ordered and validated before workers load them. Training uses sliding clips at `clip_stride`; validation uses
non-overlapping complete clips so a represented source frame is scored once. Sequences shorter than `clip_length` and
incomplete tails do not form clips.

## Configure and train the model

Enable the recurrent architecture when constructing the model, then select the public `video` dataset adapter during
training:

```python
from rfdetr import RFDETRMedium
from rfdetr.config import TrackingConfig

model = RFDETRMedium(
    group_detr=1,
    tracking=TrackingConfig(enabled=True),
)

model.train(
    dataset_dir="/path/to/my-video-dataset",
    dataset_file="video",
    epochs=50,
    batch_size=2,
    grad_accum_steps=8,
    augmentation_backend="cpu",
    tracking={
        "clip_length": 4,
        "clip_stride": 1,
        "detach_state_between_frames": False,
        "lifecycle_mode": "assignment_guided",
    },
    output_dir="output/video-tracking",
)
```

`batch_size` counts complete clips, not individual frames. Each clip remains intact through ordinary DataLoader
sampling and distributed sharding, and the trainer recurrently unrolls its frames in chronological order. Spatial
augmentation choices are shared across a clip, while pixel-level augmentation may vary by frame.

The default `detach_state_between_frames=False` permits gradients through recurrent state across the clip. Set it to
`True` only when you intentionally want truncated temporal gradients. Increase `clip_stride` to create fewer overlapping
training clips.

## Initialize from an image checkpoint

Published image checkpoints are structurally compatible with a tracking-enabled model and can initialize temporal
fine-tuning through the normal `pretrain_weights` path. Compatibility is only initialization: image-only training has
not taught the model to preserve query identity through time. Use the resulting temporally fine-tuned checkpoint for
useful persistent tracking; do not describe an unchanged image checkpoint as tracking-trained.

After training, construct the same tracking-enabled architecture with the temporal checkpoint and run it through a
`TrackingSession`. Tracking sessions are supported only by eager PyTorch inference; `optimize_for_inference()` and
exported ONNX or TensorRT graphs are stateless. See [Track Objects Across Video Frames](../run/tracking.md) for the
inference workflow and lifecycle controls.
