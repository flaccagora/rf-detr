# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

- Optional capabilities belong in nested Pydantic configs with `default_factory`; parent `model_validator(mode="after")`
  methods enforce cross-field capability constraints while leaving legacy defaults unchanged.

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
