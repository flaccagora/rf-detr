---
description: Track objects across video frames with RF-DETR persistent queries and session-local identities.
---

# Track Objects Across Video Frames

RF-DETR persistent-query tracking carries selected decoder queries from one frame to the next. A tracking session returns the usual boxes, class IDs, and confidence scores, plus a session-local `tracker_id` for each visible object.

Use this path when frames arrive in chronological order and identity must persist across frames. Ordinary `model.predict()` remains stateless and does not update a tracking session.

## Current constraints

Persistent-query tracking currently:

- supports detection models only;
- requires two-stage proposals and `group_detr=1`;
- processes one video stream per session;
- uses eager PyTorch inference;
- does not support `optimize_for_inference()` or exported models; and
- needs a tracking-trained checkpoint for useful temporal behavior. Image checkpoints are structurally compatible, but image-only training does not teach query identity through time.

## Create a tracking model

Enable tracking when constructing the model. For a surgical-tool application with at most four live identities, set `max_active_tracks=4`. Active and temporarily suspended tracks both count toward this limit; all other query positions remain available for current-frame discovery.

```python
from rfdetr import RFDETRMedium
from rfdetr.config import TrackingConfig

model = RFDETRMedium(
    pretrain_weights="path/to/tracking-checkpoint.pth",
    group_detr=1,
    tracking=TrackingConfig(
        enabled=True,
        max_active_tracks=4,
        discovery_reserve=1,
    ),
)
```

The active-track capacity plus `discovery_reserve` must not exceed the model's fixed `num_queries`. Tracking configuration is disabled by default, so existing image-inference code is unaffected.

## Process a video

Create one session for each independent video or stream. The session owns recurrent query state and the next track ID; the shared model remains stateless.

```python
import cv2
import supervision as sv

from rfdetr.config import TrackingSessionConfig

session = model.create_tracking_session(
    TrackingSessionConfig(
        activation_threshold=0.5,
        continuation_threshold=0.3,
        duplicate_iou_threshold=0.7,
        max_missed_frames=30,
    )
)

capture = cv2.VideoCapture("path/to/video.mp4")
if not capture.isOpened():
    raise RuntimeError("Could not open video")

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
frame_index = 0

while True:
    ok, frame_bgr = capture.read()
    if not ok:
        break

    # NumPy input must be RGB in HWC layout.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections = session.update(frame_rgb, frame_index=frame_index)

    labels = [
        f"track={track_id} {model.class_names[class_id]} {confidence:.2f}"
        for track_id, class_id, confidence in zip(
            detections.tracker_id,
            detections.class_id,
            detections.confidence,
            strict=True,
        )
    ]
    annotated = box_annotator.annotate(frame_bgr.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels)

    cv2.imshow("RF-DETR tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_index += 1

capture.release()
cv2.destroyAllWindows()
```

`session.update()` accepts a path or URL, a PIL image, an RGB NumPy array in HWC layout, or a normalized floating-point tensor in CHW layout. Tensor values must be finite and in `[0, 1]`.

## Interpret the result

`update()` returns a Supervision `Detections` object. Its row-aligned fields include:

- `xyxy`: boxes in the original frame's pixel coordinates;
- `confidence`: the current confidence for each visible track;
- `class_id`: the current predicted class;
- `tracker_id`: a non-negative identity local to this session; and
- `metadata`: the frame index, optional timestamp, and lifecycle events.

Only tracks whose current observation passes `continuation_threshold` are returned. A missing track may remain suspended internally until it is recovered or exceeds `max_missed_frames`.

```python
for event in session.last_events:
    print(event.kind, event.track_id, event.slot, event.frame_index)

for track in session.active_tracks:
    print(track.track_id, track.status, track.missed_frames)
```

Lifecycle event kinds are `activated`, `suspended`, `recovered`, `terminated`, `duplicate_suppressed`, and `capacity_suppressed`.

## Frame indices, dropped frames, and resets

Frame indices must increase strictly within a session. When omitted, RF-DETR increments them by one. Supplying source-frame indices is preferable when frames may be dropped because expiry is based on the distance from the last reliable source frame.

Call `reset()` before reusing a session for a different video:

```python
session.reset()
```

Resetting clears recurrent state and restarts track IDs at zero. Never carry one session across unrelated videos.

## Track multiple streams

Create a separate session for every logical stream. Sessions may share one model without sharing identities or recurrent state:

```python
left_camera = model.create_tracking_session()
right_camera = model.create_tracking_session()

left_detections = left_camera.update(left_frame)
right_detections = right_camera.update(right_frame)
```

The current public session API processes each stream independently. It does not batch multiple sessions into one model call.

## Tune lifecycle behavior

`TrackingSessionConfig` controls host-side track lifecycle rather than model architecture:

| Setting                   | Effect                                                                                                                                  |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `activation_threshold`    | Minimum confidence required to create a track from a discovery query. Raise it to reduce false births.                                  |
| `continuation_threshold`  | Minimum confidence required to commit a new observation for an existing track. Raise it to avoid updating state from weak observations. |
| `duplicate_iou_threshold` | Suppresses an overlapping, same-class discovery that would duplicate an existing track.                                                 |
| `max_missed_frames`       | Source-frame distance tolerated since the last reliable observation before termination.                                                 |
| `collect_timing`          | Adds synchronized per-frame latency measurements. Leave disabled during normal inference.                                               |

Tune these values on complete sequences, not isolated frames. In particular, lowering activation thresholds can improve tool discovery while increasing duplicate tracks, and increasing `max_missed_frames` can improve occlusion recovery while occupying one of the four track slots longer.

## Measure streaming latency

Enable timing only while benchmarking because CUDA synchronization adds measurement overhead:

```python
from rfdetr.config import TrackingSessionConfig

session = model.create_tracking_session(TrackingSessionConfig(collect_timing=True))
session.update(frame_rgb)

timing = session.last_timing
if timing is not None:
    print(f"model: {timing.model_ms:.2f} ms")
    print(f"tracking overhead: {timing.tracking_overhead_ms:.2f} ms")
    print(f"total: {timing.total_ms:.2f} ms")
```

State memory and query count remain fixed across a sequence. Latency should therefore be evaluated as a per-frame distribution and checked with a long-sequence run.

## Train for temporal behavior

The public video dataset adapter connects sequence indexing, shared transforms, clip collation, recurrent training, and
identity-aware loss to `model.train(dataset_file="video")`. See [Train RF-DETR for Video Tracking](../train/video.md) for
the on-disk COCO-video schema, identity rules, model configuration, and complete training invocation.

Video annotations use COCO-style `images` and `annotations` arrays with additional explicit temporal fields:

```json
{
  "images": [
    {
      "id": 10,
      "file_name": "frames/000010.jpg",
      "sequence_id": "procedure-001",
      "frame_index": 10,
      "timestamp": 0.333
    }
  ],
  "annotations": [
    {
      "id": 100,
      "image_id": 10,
      "category_id": 2,
      "bbox": [
        120,
        80,
        40,
        90
      ],
      "track_id": 0,
      "identity_provenance": "human"
    }
  ]
}
```

Required rules include:

- every image has an explicit `sequence_id` (or `video_id`) and frame index;
- every annotation has `track_id`, using JSON `null` when identity is unknown;
- every annotation names non-empty identity provenance, such as `human`, `pseudo`, or `synthetic`;
- track IDs are sequence-local and non-negative; and
- one track ID cannot change category within a sequence.

`build_video_clip_index()` validates and orders this metadata. `SharedSequenceTransform` reuses spatial randomness across clip frames, and `sequence_collate_fn()` creates time-major batches for recurrent unrolling.

```python
import json

from rfdetr.datasets import build_video_clip_index

with open("annotations.json", encoding="utf-8") as stream:
    annotations = json.load(stream)

clips = build_video_clip_index(
    annotations,
    clip_length=4,
    stride=1,
)
```

Tracking training requires architecture tracking to be enabled with `group_detr=1` and a `TrackingTrainConfig` whose
`clip_length` is greater than one. The public adapter constructs complete clips of that length.

## Unsupported operations

Do not call `optimize_for_inference()` before creating or updating a tracking session. Exported ONNX, TensorRT, and other optimized paths currently expose stateless image inputs and outputs, so they cannot preserve persistent queries. RF-DETR raises an explicit error instead of silently running stateless inference.

If deployment requires an exported graph, the export contract must first be extended with explicit recurrent state inputs and outputs.

## Troubleshooting

| Symptom                                                       | Check                                                                                                                       |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `TrackingSession requires model_config.tracking.enabled=True` | Construct the model with `TrackingConfig(enabled=True, ...)`.                                                               |
| Configuration rejects `group_detr`                            | Set `group_detr=1`.                                                                                                         |
| IDs change frequently                                         | Use a tracking-trained checkpoint, inspect `suspended`/`terminated` events, and validate the continuation threshold.        |
| New tools are not assigned IDs                                | Check `activation_threshold`, the four-track capacity, and `capacity_suppressed` events. Suspended tracks consume capacity. |
| Duplicate IDs appear around one tool                          | Inspect `duplicate_suppressed` events and tune `duplicate_iou_threshold`.                                                   |
| A session rejects a frame index                               | Ensure source-frame indices increase strictly and reset between videos.                                                     |
| Optimized inference raises an error                           | Use the eager PyTorch model; tracking export is not implemented.                                                            |
