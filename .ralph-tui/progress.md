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
- Persistent/discovery query composition is a fixed-shape `torch.where` selection over a per-item active mask; prior
  normalized boxes cross the parameterization boundary before selection, while inactive slots remain untouched.
- Tracking forward paths should delegate to the ordinary detection graph and inject explicit state only at decoder
  query composition; final decoder features and predicted boxes then form the aligned candidate state.
- Host lifecycle updates are pure immutable transitions over an aligned slot table and committed neural state; weak
  continuations preserve trusted tensors, and terminated slots become discovery queries only on the following frame.
- Public streaming inference keeps all recurrent neural and host state inside an independent session; output detections
  are built directly from committed active slots so boxes, classes, confidences, and tracker IDs remain slot-aligned.
- Video clip indexes are built from explicit sequence/frame metadata, never filename order; unknown object identities
  remain `None`, while known sequence-local identities are validated for per-frame uniqueness and category stability.

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

## 2026-07-27 - US-005
- Added a validated transformer composition boundary that replaces active decoder slots with prior query features and
  parameterization-correct prior box references while retaining current-frame discovery initialization in inactive
  slots.
- Added behavioral coverage for mixed per-batch role maps and both reparameterized and legacy decoder references.
- Files changed: `src/rfdetr/models/transformer.py`, `tests/models/test_transformer.py`,
  `.ralph-tui/progress.md`.
- **Learnings:**
  - Query persistence is a masked fixed-slot selection, so batches may have different active counts without changing
    decoder tensor shapes or concatenating variable-length state.
  - Recurrent normalized boxes must be converted before masked composition; discovery references are already in the
    decoder's internal parameterization and must remain unchanged.
  - Focused pytest collection remains blocked by the incompatible host Transformers package and then missing
    `supervision`, even with narrow compatibility shims. Isolated runtime checks pass in both box modes, as do Python
    compilation, Ruff lint/format, and `git diff --check`; `mypy` and `pre-commit` are unavailable.
---

## 2026-07-27 - US-006
- Added `LWDETR.forward_tracking`, accepting explicit or omitted prior state and returning standard frame predictions,
  aligned final-decoder candidate state, input slot roles, auxiliary predictions, and encoder predictions.
- Routed recurrent state through the unchanged backbone, decoder, and detection heads while preserving the ordinary
  `forward()` interface and stateless transformer call.
- Added focused behavioral coverage for structured candidate output, role preservation, auxiliary/encoder output, and
  empty-state parity with stateless evaluation.
- Files changed: `src/rfdetr/models/lwdetr.py`, `src/rfdetr/models/transformer.py`,
  `tests/models/test_lwdetr_tracking.py`, `.ralph-tui/progress.md`.
- **Learnings:**
  - The final decoder hidden state is already slot-aligned with final detection boxes, so it can become candidate neural
    state without a second decoder path or lifecycle decisions inside the model.
  - Keeping the input active mask separate from an all-candidate output state preserves the distinction between query
    roles entering the frame and candidates that the external lifecycle may commit.
  - Focused pytest collection remains blocked by the host Transformers package lacking `BackboneConfigMixin` and then
    the missing `deprecate` package. Python compilation, Ruff lint/format, and `git diff --check` pass; `mypy` and
    `pre-commit` are unavailable.
---

## 2026-07-27 - US-007
- Added frozen host-side slot metadata, ordered lifecycle diagnostics, and a pure single-stream transition boundary for
  activation, suspension, recovery, termination, duplicate suppression, capacity enforcement, and slot recycling.
- Committed reliable candidate tensors while preserving the last trusted neural state during suspension; termination
  clears both host and neural state, and monotonic IDs prevent recycled slots from inheriting old identities.
- Added focused behavioral coverage for activation, elapsed-frame suspension, same-ID recovery, expiry and recycling,
  overlap suppression, and strict host/neural role-map validation.
- Files changed: `src/rfdetr/tracking/__init__.py`, `src/rfdetr/tracking/lifecycle.py`,
  `tests/models/test_tracking_lifecycle.py`, `.ralph-tui/progress.md`.
- **Learnings:**
  - Lifecycle expiry should use source-frame deltas from the last reliable observation, rather than update-call counts,
    so sampled or dropped frames age tracks deterministically.
  - A slot terminated from a persistent role must remain cleared for the current transition and return as a discovery
    query on the next model call; activating its current persistent candidate would cross role semantics.
  - The host Transformers installation still blocks ordinary package collection. The focused lifecycle tests pass
    through an isolated package-loading harness; Python compilation, Ruff lint/format, and `git diff --check` pass,
    while `mypy` and `pre-commit` are unavailable.
---

## 2026-07-27 - US-008
- Added a public `TrackingSession` plus `RFDETR.create_tracking_session()` for independent single-stream eager-PyTorch
  inference with `update()` and `reset()` operations.
- Added predict-compatible single-frame preprocessing for paths/URLs, PIL images, NumPy arrays, and tensors; returned
  visible `supervision.Detections` carry committed slot-aligned tracker IDs and lifecycle metadata.
- Added read-only active/suspended track inspection, lifecycle event inspection, automatic or explicit frame indexing,
  session-local ID reset, and clear rejection of optimized/export-style inference without explicit state I/O.
- Files changed: `src/rfdetr/tracking/session.py`, `src/rfdetr/tracking/__init__.py`, `src/rfdetr/detr.py`,
  `tests/models/test_tracking_session.py`, `.ralph-tui/progress.md`.
- **Learnings:**
  - Session results should be projected from committed active lifecycle slots instead of generic top-k postprocessing;
    this guarantees each public tracker ID refers to the same fixed query slot as its box, class, and confidence.
  - Lazy model device placement must occur before allocating empty recurrent state, and normalized input tensors must be
    cast to the model/state dtype before the tracking forward path for half-precision eager inference.
  - The focused session suite passes through the isolated dependency-shim harness (6 tests). Ordinary host collection
    remains blocked by incompatible/missing `transformers`, `deprecate`, and `supervision` packages. Python compilation,
    Ruff lint/format, and `git diff --check` pass; `mypy` and `pre-commit` are unavailable.
---

## 2026-07-27 - US-009
- Added frozen video object, frame, and clip schemas plus deterministic fixed-length sliding-window indexing over
  explicit sequence IDs and source-frame indices.
- Validated required identity/provenance fields, preserved unknown identities as `None`, rejected ambiguous frame
  positions and duplicate per-frame identities, and enforced category stability for known sequence-local tracks.
- Supported both project-facing and COCO-video metadata names without consulting filenames, and exposed the indexing
  boundary through the datasets package.
- Files changed: `src/rfdetr/datasets/video.py`, `src/rfdetr/datasets/__init__.py`,
  `tests/datasets/test_video.py`, `.ralph-tui/progress.md`.
- **Learnings:**
  - Explicit JSON `null` is a useful distinction from an absent `track_id`: the former means identity is unknown, while
    the latter is a schema error that could otherwise invite accidental filename- or frame-based identity synthesis.
  - Sequence-local category consistency catches identity reuse mistakes early without incorrectly requiring track IDs
    to be globally unique across independent sequences.
  - The focused suite passes through a dependency-isolated package harness (9 tests). Ordinary project collection is
    blocked first by the incompatible host Transformers package and, with package isolation, by missing `supervision`.
    Python compilation, Ruff lint/format, and `git diff --check` pass.
---
