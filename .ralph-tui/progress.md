# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

- Cross-config constraints belong on a public validation boundary that receives both `ModelConfig` and `TrainConfig`;
  call it when constructing model and data modules so invalid combinations fail before model or dataset setup.
- A validated sequence index should retain explicit image paths, declared dimensions, and box geometry so dataset
  workers consume normalized records without rejoining or reinterpreting raw annotation JSON.
- Sequence augmentation should decompose existing detection pipelines by stage: replay declared geometric randomness
  across frames, while allowing declared pixel-level transforms to draw independently.
- Keep one complete clip behind each video-dataset index; ordinary DataLoader samplers, distributed sharding, batch
  sizing, and gradient-accumulation alignment then operate in clip units without temporal sampler special cases.
- Give validation its own clip-index boundary: default its stride to the clip length so source frames are scored once,
  while retaining duplicate `(sequence_id, frame_index)` rejection at the evaluation accumulator boundary.
- A public dataset format is complete only when `build_dataset` resolves its on-disk split convention and constructs
  the normalized dataset; configuration literals and standalone dataset classes do not make `model.train()` usable.

---

## 2026-07-27 - US-007
- Added a task-oriented video-tracking training guide covering the conventional on-disk split layout, complete
  COCO-video schema, chronology and identity invariants, recurrent model configuration, and the public
  `model.train(dataset_file="video")` invocation.
- Documented the detection-only, CPU-augmentation, explicit-batch-size, multi-frame, `group_detr=1`, and eager-inference
  limitations, including the distinction between structural image-checkpoint compatibility and useful temporal
  fine-tuning.
- Added the guide to the MkDocs **Train Model** navigation and updated the tracking inference guide to point to the now
  connected public video dataset adapter instead of describing it as unavailable.
- Files changed: `docs/learn/train/video.md`, `docs/learn/run/tracking.md`, `mkdocs.yaml`,
  `.ralph-tui/progress.md`.
- Verification:
  - `HF_HOME=/tmp/clevis-hf docker compose run --rm -v
    "$PWD/submodules/rf-detr:/app/submodules/rf-detr:ro" clevis -lc 'python -m pip install -q pytest pytest-timeout
    pytest-doctestplus && cd /app/submodules/rf-detr && PYTHONPATH=src python -m pytest -q
    tests/models/test_video_train_config.py tests/datasets/test_video.py tests/training/test_module_data.py
    tests/training/test_module_model.py tests/training/test_trainer_smoke.py tests/evaluation/test_sequence.py'` (from the
    Clevis root) — blocked because access to `/var/run/docker.sock` was denied.
  - `pre-commit run --all-files` — unavailable (`pre-commit: command not found`).
  - `UV_CACHE_DIR=/tmp/rfdetr-uv-cache uv run --no-sync mkdocs build --strict` — unavailable because `mkdocs` is not
    installed in the repository environment. The first attempt with uv's default cache was additionally blocked by its
    read-only host cache.
  - Changed-document local-link validation — passed; every relative link resolves on disk.
  - Stale public-adapter wording search and `git diff --check` — passed.
- **Learnings:**
  - Training and streaming inference are best documented as neighboring tasks: training owns the COCO-video data and
    recurrent fine-tuning contract, while inference owns session lifecycle and eager-state constraints.
  - Image-checkpoint compatibility must be described as initialization compatibility, not temporal capability; useful
    persistent identities require fine-tuning on chronological identity annotations.
---

## 2026-07-27 - US-006
- Registered the video format at the public dataset factory, including conventional `train/` and `val/` annotation
  discovery, explicit annotation-path support, training sliding clips, non-overlapping validation clips, and existing
  RF-DETR detection transforms wrapped for sequence-consistent geometry.
- Added a tiny on-disk public-training smoke that invokes `RFDETR.train(dataset_file="video")`, executes Lightning
  training and validation through real data/module boundaries, checks empty-to-predicted recurrent state flow, verifies
  image-checkpoint initialization uses the canonical loader, and checks serialized video/tracking configuration.
- Kept the existing ordinary detection smoke intact and added focused public dataset-factory coverage for both video
  splits.
- Files changed: `src/rfdetr/datasets/__init__.py`, `tests/datasets/test_video.py`,
  `tests/training/test_trainer_smoke.py`, `.ralph-tui/progress.md`.
- Verification:
  - `UV_CACHE_DIR=/tmp/rfdetr-uv-cache uv run --no-sync pytest -q tests/datasets/test_video.py -k
    public_dataset_factory_builds_on_disk_video_splits` — blocked during collection because the host Transformers
    installation lacks `BackboneConfigMixin`.
  - `.venv/bin/python -c 'import transformers; print(transformers.__version__); from transformers import
    BackboneConfigMixin'` — blocked because the repository-local environment has no `transformers` installation.
  - `HF_HOME=/tmp/clevis-hf docker compose run --rm -v
    "$PWD/submodules/rf-detr:/app/submodules/rf-detr:ro" clevis -lc 'python -m pip install -q pytest pytest-timeout
    pytest-doctestplus && cd /app/submodules/rf-detr && PYTHONPATH=src python -m pytest -q
    tests/datasets/test_video.py tests/training/test_trainer_smoke.py tests/training/test_module_model.py
    tests/training/test_load_pretrain_weights.py'` (from Clevis root) — blocked because access to
    `/var/run/docker.sock` was denied.
  - `pre-commit run --all-files` — unavailable (`pre-commit: command not found`).
  - `ruff check` (including its safe import fixes), `ruff format --check`, `python -m compileall`, and
    `git diff --check` for the affected source and tests — passed.
- **Learnings:**
  - The recurrent model loop and clip DataLoader were already present, but the missing `build_dataset("video", ...)`
    registration made the entire public feature unreachable; public-boundary smoke tests catch this integration gap.
  - Using the ordinary detection transform builder at the format adapter preserves image-training preprocessing while
    `VideoSequenceDataset` supplies the sequence-specific shared-randomness semantics.
  - A tiny recurrent network can exercise Lightning optimization and checkpoint initialization without weakening the
    on-disk dataset boundary: only heavyweight model construction needs substitution.
---

## 2026-07-27 - US-005
- Added a public validation clip-index boundary that defaults to complete, non-overlapping clips while leaving the
  training sliding-window index unchanged.
- Verified the existing sequence output keeps identity-free detection records available independently, records reset
  boundaries on sequence changes, rejects duplicate source frames deterministically, and isolates MOT exports by
  sequence. Ordinary image COCO evaluation code and behavior were not modified.
- Added focused coverage proving the default validation index contains no repeated `(sequence_id, frame_index)` keys.
- Files changed: `src/rfdetr/datasets/video.py`, `src/rfdetr/datasets/__init__.py`,
  `tests/datasets/test_video.py`, `.ralph-tui/progress.md`.
- Verification:
  - `docker compose run --rm -v "$PWD/submodules/rf-detr:/app/submodules/rf-detr:ro" clevis -lc 'python -m pip
    install -q pytest pytest-timeout pytest-doctestplus && cd /app/submodules/rf-detr && PYTHONPATH=src python -m
    pytest -q tests/datasets/test_video.py tests/evaluation/test_sequence.py'` (from Clevis root) — blocked before
    startup because Compose requires `HF_HOME`.
  - `HF_HOME=/tmp/clevis-hf docker compose run --rm -v
    "$PWD/submodules/rf-detr:/app/submodules/rf-detr:ro" clevis -lc 'python -m pip install -q pytest pytest-timeout
    pytest-doctestplus && cd /app/submodules/rf-detr && PYTHONPATH=src python -m pytest -q
    tests/datasets/test_video.py tests/evaluation/test_sequence.py'` (from Clevis root) — blocked because access to
    `/var/run/docker.sock` was denied.
  - `pre-commit run --all-files` — unavailable (`pre-commit: command not found`).
  - Dependency-isolated sequence validation checks for non-overlap, detection-only records, reset boundaries, and
    duplicate rejection — passed.
  - `ruff check` and focused `ruff format --check` for the affected dataset/evaluation files, `python -m compileall`,
    and `git diff --check` — passed.
- **Learnings:**
  - Training and evaluation need different default clip strides: sliding windows improve training coverage, but
    validation must advance by a full clip unless its downstream accumulator explicitly handles repeated provenance.
  - Using `(sequence_id, frame_index)` as the evaluated-frame key preserves sequence-local frame numbering and prevents
    both cross-clip metric inflation and accidental identity merging across sequence resets.
---

## 2026-07-27 - US-004
- Selected `make_sequence_collate_fn` at the DataModule boundary only when `dataset_file="video"`; all image datasets
  retain the existing ordinary image collator and loader behavior.
- Kept sampling, replacement sampling, distributed alignment, batch size, and gradient accumulation unchanged because
  each sequence-dataset item is already one complete clip; video batches now expose chronological time-major image and
  target tuples.
- Added behavioral coverage for complete-clip training batches and deterministic multi-worker validation ordering that
  cannot mix clip membership across time steps.
- Files changed: `src/rfdetr/training/module_data.py`, `tests/training/test_module_data.py`,
  `.ralph-tui/progress.md`.
- Verification:
  - `UV_CACHE_DIR=/tmp/rfdetr-uv-cache uv run --no-sync pytest -q tests/training/test_module_data.py -k
    video_batch_is_time_major_and_keeps_complete_clips` — blocked during collection because the host Transformers
    installation lacks `BackboneConfigMixin`.
  - `HF_HOME=/tmp/clevis-hf docker compose run --rm -v
    "$PWD/submodules/rf-detr:/app/submodules/rf-detr:ro" clevis -lc 'python -m pip install -q pytest pytest-timeout
    pytest-doctestplus && cd /app/submodules/rf-detr && PYTHONPATH=src python -m pytest -q
    tests/training/test_module_data.py tests/datasets/test_video.py'` (from Clevis root) — blocked because access to
    `/var/run/docker.sock` was denied.
  - `pre-commit run --all-files` — unavailable (`pre-commit: command not found`).
  - `ruff check src/rfdetr/training/module_data.py tests/training/test_module_data.py`, focused
    `ruff format --check`, `python -m compileall`, and `git diff --check` — passed.
- **Learnings:**
  - Treating clips as dataset samples lets PyTorch's existing samplers and Lightning's distributed sampler injection
    shard complete temporal units; only collation needs to become sequence-aware.
  - DataLoader worker prefetch preserves sampler order, so a sequential validation sampler plus clip-atomic
    `__getitem__` maintains deterministic frame and sequence boundaries without worker-specific sequencing logic.
---

## 2026-07-27 - US-003
- Wrapped ordinary RF-DETR detection pipelines at the `VideoSequenceDataset` boundary and refined
  `SharedSequenceTransform` to replay resize/crop/flip randomness stage-by-stage without changing ordinary image
  transform construction.
- Kept photometric augmentation frame-local and documented that contract separately from shared spatial randomness.
- Verified crop removal synchronizes boxes, labels, nullable `track_ids`, identity provenance, and arbitrary
  tensor/sequence instance fields; added coverage for shared flips, filtering, empty frames, unknown identities,
  photometric draws, and unchanged image pipelines.
- Files changed: `src/rfdetr/datasets/video.py`, `tests/datasets/test_video.py`, `.ralph-tui/progress.md`.
- Verification:
  - `UV_CACHE_DIR=/tmp/rfdetr-uv-cache uv run --no-sync pytest -q tests/datasets/test_video.py -k
    'shared_sequence_transform or crop or ordinary_image'` — blocked during collection because the host Transformers
    installation lacks `BackboneConfigMixin`.
  - `HF_HOME=/tmp/clevis-hf docker compose run --rm -v "$PWD/tests:/app/tests:ro" -v
    "$PWD/submodules/rf-detr:/app/submodules/rf-detr:ro" clevis -lc 'python -m pip install -q pytest && python -m pytest
    -q /app/submodules/rf-detr/tests/datasets/test_video.py'` (from Clevis root) — blocked because access to
    `/var/run/docker.sock` was denied.
  - `pre-commit run --all-files` — unavailable (`pre-commit: command not found`).
  - `ruff check src/rfdetr/datasets/video.py src/rfdetr/datasets/__init__.py tests/datasets/test_video.py`,
    focused `ruff format --check`, `python -m compileall`, and `git diff --check` — passed.
  - Dependency-isolated shared-spatial/frame-local-photometric behavioral check — passed.
- **Learnings:**
  - Replaying a whole composed transform also couples blur/color/noise; decomposing the existing Compose pipeline lets
    geometric wrappers share decisions while pixel-level wrappers retain realistic frame-local variation.
  - Albumentations' retained original indices are the stable boundary for filtering all instance-aligned fields,
    including Python sequences that preserve JSON-null identities.
---

## 2026-07-27 - US-002
- Added `VideoSequenceDataset`, which consumes the validated clip index, loads each frame by its explicit indexed path,
  verifies decoded dimensions, and returns equal-length chronological image/target tuples with recurrent metadata.
- Extended indexed frame/object records to retain declared image paths and sizes plus validated COCO box geometry;
  targets expose boxes, labels, nullable track IDs, image/original sizes, sequence/frame/timestamp metadata, and aligned
  identity provenance without synthesizing identities for JSON `null`.
- Added focused behavioral coverage for real image loading, chronology, metadata alignment, unknown identity preservation,
  missing files, decoded image/annotation mismatches, and invalid/non-finite/out-of-bounds boxes.
- Files changed: `src/rfdetr/datasets/video.py`, `src/rfdetr/datasets/__init__.py`,
  `tests/datasets/test_video.py`, `.ralph-tui/progress.md`.
- Verification:
  - `UV_CACHE_DIR=/tmp/rfdetr-uv-cache uv run --no-sync pytest -q tests/datasets/test_video.py -k chronological_clip`
    — blocked during collection because the host Transformers installation lacks `BackboneConfigMixin`.
  - `HF_HOME=/tmp/clevis-hf docker compose run --rm -v "$PWD/tests:/app/tests:ro" -v
    "$PWD/submodules/rf-detr:/app/submodules/rf-detr:ro" clevis -lc 'python -m pip install -q pytest && cd
    /app/submodules/rf-detr && PYTHONPATH=src python -m pytest -q tests/datasets/test_video.py'` (from Clevis root) —
    blocked because access to `/var/run/docker.sock` was denied.
  - `pre-commit run --all-files` — unavailable (`pre-commit: command not found`).
  - `ruff check src/rfdetr/datasets/video.py src/rfdetr/datasets/__init__.py tests/datasets/test_video.py`,
    focused `ruff format --check`, `python -m compileall`, and `git diff --check` — passed.
  - Dependency-isolated real-PNG behavioral check of `build_video_clip_index` → `VideoSequenceDataset` — passed,
    including explicit chronology, valid ID zero, JSON-null identity, and xywh-to-xyxy box conversion.
- **Learnings:**
  - Preserving unknown identity as a Python `None` sequence is necessary; tensor conversion would introduce a synthetic
    sentinel ID and blur the distinction between unknown identity and valid track ID zero.
  - Image dimensions and box bounds are most actionable at the public indexing/loading boundaries, before stochastic
    transforms can change geometry or hide source annotation mismatches.
---

## 2026-07-27 - US-001
- Added `dataset_file="video"` while preserving the existing `"roboflow"` default, plus serialized clip stride and
  annotation path settings in `TrackingTrainConfig`.
- Added actionable pre-data-loading validation for architecture tracking, `group_detr=1`, multi-frame clips, explicit
  integer batch size, and CPU augmentation, with focused behavioral coverage that leaves image configuration tests
  unchanged.
- Files changed: `src/rfdetr/config.py`, `src/rfdetr/training/module_model.py`,
  `src/rfdetr/training/module_data.py`, `tests/models/test_video_train_config.py`, `.ralph-tui/progress.md`.
- Verification:
  - `UV_CACHE_DIR=/tmp/rfdetr-uv-cache uv run --no-sync pytest -q tests/models/test_video_train_config.py` — blocked
    during collection because the host Transformers installation lacks `BackboneConfigMixin`.
  - `HF_HOME=/tmp/clevis-hf docker compose run --rm -v "$PWD/submodules/rf-detr:/app/submodules/rf-detr:ro" clevis
    -lc 'python -m pip install -q pytest pytest-timeout pytest-doctestplus && cd /app/submodules/rf-detr &&
    PYTHONPATH=src python -m pytest -q tests/models/test_video_train_config.py
    tests/models/test_tracking_config.py'` (from Clevis root) — blocked because access to `/var/run/docker.sock` was
    denied. The first invocation without `HF_HOME` also reported Compose's required-variable interpolation error.
  - `pre-commit run --all-files` — unavailable (`pre-commit: command not found`);
    `UV_CACHE_DIR=/tmp/rfdetr-uv-cache uv run --no-sync pre-commit run --all-files` likewise could not spawn the missing
    executable.
  - Dependency-isolated configuration checks, `python -m compileall`, focused `ruff check`,
    `ruff format --check`, and focused `git diff --check` — passed.
- **Learnings:**
  - `TrainConfig` cannot independently validate architecture tracking; the paired configuration boundary must run
    before either Lightning module begins building runtime state.
  - Nested Pydantic settings flow through the existing `model_dump()` namespace/checkpoint path, and using a normalized
    string for annotation paths keeps both JSON training configuration and checkpoint metadata serializable.
  - The host environment cannot execute the authoritative gates: its Python dependencies are incompatible, Docker is
    inaccessible, and `pre-commit` is not installed.
---
