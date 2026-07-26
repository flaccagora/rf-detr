# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

- Optional capabilities belong in nested Pydantic configs with `default_factory`; parent `model_validator(mode="after")`
  methods enforce cross-field capability constraints while leaving legacy defaults unchanged.
- Neural tracking boundaries use frozen dataclasses that validate `[batch, query, ...]` slot alignment, normalized finite
  boxes, boolean role maps, and device/dtype compatibility at construction time.
- Recurrent boxes cross into the decoder through `normalized_boxes_to_refpoints`: reparameterized models keep normalized
  `cxcywh`, while legacy additive models receive numerically stable inverse-sigmoid references.
- Two-stage discovery selection is isolated in `_initialize_two_stage_discovery`; it returns one frozen, slot-aligned
  bundle containing detached decoder references plus differentiable query features and encoder-loss boxes.

---

## 2026-07-27 - US-001
- Added explicit, independently typed persistent-query architecture capacity, video-training clip, and inference
  lifecycle configuration.
- Kept tracking disabled by default and added construction-time validation for detection-only, two-stage,
  single-query-group tracking plus fixed query-capacity/discovery-reserve constraints.
- Added focused behavioral coverage for defaults, configuration separation, unsupported capabilities, actionable
  capacity errors, and serialization round trips.
- Files changed: `src/rfdetr/config.py`, `tests/models/test_tracking_config.py`, `.ralph-tui/progress.md`.
- **Learnings:**
  - Existing configuration uses strict Pydantic models with assignment validation, so nested capability schemas retain
    typo detection and serialize through the established `model_dump()` / `model_validate()` path.
  - Cross-field tracking restrictions must be conditional on `tracking.enabled`; existing segmentation, keypoint,
    one-stage, and grouped-query image configurations must remain valid while tracking is disabled.
  - Local pytest collection is blocked in this environment by an incompatible global Transformers installation, and
    the PRD Compose fallback is unavailable because access to the Docker socket is denied. Isolated configuration
    checks, Ruff lint/format checks, and `git diff --check` passed; `pre-commit` is not installed.
---

## 2026-07-27 - US-002
- Added frozen, validated `TrackQueryState` and `TrackingFrameOutput` contracts with an explicit inactive-state factory,
  normalized box validation, and fixed query-slot alignment.
- Added behavioral tests for empty-state semantics, immutability-style field assignment, device/dtype propagation,
  invalid state rejection, structured frame outputs, and cross-output slot alignment.
- Files changed: `src/rfdetr/models/tracking.py`, `tests/models/test_tracking_types.py`,
  `.ralph-tui/progress.md`.
- **Learnings:**
  - The neural state active mask is the sole validity indicator; zero-filled inactive feature and box storage is padding,
    not historical track state.
  - Candidate neural state and the input role map must remain distinct in the frame contract because lifecycle policy,
    outside the model graph, decides which candidates become committed state.
  - Local pytest collection remains blocked before the focused tests by the host Transformers package lacking
    `BackboneConfigMixin`. Isolated runtime contract checks, Python compilation, Ruff lint/format, and
    `git diff --check` pass; `pre-commit` is unavailable.
---

## 2026-07-27 - US-003
- Added one normalized-box-to-decoder-reference boundary supporting both RF-DETR box parameterizations.
- Added focused behavioral coverage for direct normalized references, stable legacy inverse-sigmoid references, and
  dtype/device preservation.
- Files changed: `src/rfdetr/models/transformer.py`, `tests/models/test_transformer.py`,
  `.ralph-tui/progress.md`.
- **Learnings:**
  - `bbox_reparam=True` uses normalized `cxcywh` references directly; the legacy additive refinement path expects
    inverse-sigmoid references because decoder attention applies `sigmoid()` before using them.
  - Reusing the existing `inverse_sigmoid` utility keeps exact zero/one state boxes finite at the conversion boundary.
  - Focused pytest collection remains blocked by the host Transformers package lacking `BackboneConfigMixin`.
    Isolated runtime conversion checks, Python compilation, Ruff lint/format, and `git diff --check` pass;
    `pre-commit` is unavailable.
---

## 2026-07-27 - US-004
- Extracted current two-stage proposal generation, per-group scoring, box refinement, top-k selection, and feature
  gathering into a testable transformer step while preserving the existing `forward()` outputs.
- Added focused behavioral coverage for ranked discovery initialization in both box parameterizations, including
  decoder-reference detachment and differentiable encoder state.
- Files changed: `src/rfdetr/models/transformer.py`, `tests/models/test_transformer.py`,
  `.ralph-tui/progress.md`.
- **Learnings:**
  - The proposal boundary carries three aligned tensors with deliberately different gradient semantics: decoder
    references are detached, while selected query features and encoder boxes remain connected for training.
  - Evaluation uses only the first proposal group, while training processes and concatenates every configured DETR
    group; keeping that decision inside the extracted step preserves current image-model behavior.
  - Focused pytest collection remains blocked by the host Transformers package lacking `BackboneConfigMixin`; after a
    narrow compatibility shim, collection reaches a second missing host dependency (`deprecate`). Python compilation,
    Ruff lint/format, and `git diff --check` pass.
---
