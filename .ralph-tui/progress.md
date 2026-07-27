# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

- Cross-config constraints belong on a public validation boundary that receives both `ModelConfig` and `TrainConfig`;
  call it when constructing model and data modules so invalid combinations fail before model or dataset setup.

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
