# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

- Optional capabilities belong in nested Pydantic configs with `default_factory`; parent `model_validator(mode="after")`
  methods enforce cross-field capability constraints while leaving legacy defaults unchanged.
- Neural tracking boundaries use frozen dataclasses that validate `[batch, query, ...]` slot alignment, normalized finite
  boxes, boolean role maps, and device/dtype compatibility at construction time.

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
