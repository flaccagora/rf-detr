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
- Tracking remains checkpoint-compatible by keeping recurrence parameter-free: image and tracking paths share one
  strict state dict, and ordinary stateless `forward()` remains the image-inference boundary.
- Host lifecycle updates are pure immutable transitions over an aligned slot table and committed neural state; weak
  continuations preserve trusted tensors, and terminated slots become discovery queries only on the following frame.
- Public streaming inference keeps all recurrent neural and host state inside an independent session; output detections
  are built directly from committed active slots so boxes, classes, confidences, and tracker IDs remain slot-aligned.
- Video clip indexes are built from explicit sequence/frame metadata, never filename order; unknown object identities
  remain `None`, while known sequence-local identities are validated for per-frame uniqueness and category stability.
- Sequence augmentation replays one transform state across every chronological frame, and collation remains time-major
  as one ordinary `NestedTensor` batch per step so the existing backbone can be reused unchanged during unroll.
- Sequence assignment first binds visible known identities to their persistent slots, then runs the ordinary Hungarian
  matcher on the residual target/discovery-slot subproblem; unknown identities remain non-persistent.
- Tracking criteria reuse one precomputed sequence assignment for the final and every auxiliary decoder layer; encoder
  proposal losses remain frame-local and use the ordinary matcher, while unmatched absent slots are classification-only.
- Recurrent Lightning steps consume time-major frame batches, create state per clip, average already-normalized frame
  losses, and commit predicted candidate tensors under assignment-guided or confidence-gated lifecycle policy.
- Sequence evaluation retains one rich provenance record while deriving identity-free detection views and per-sequence
  MOTChallenge streams; sequence grouping is the reset boundary that scopes otherwise reusable session-local track IDs.
- Optional performance instrumentation belongs at the public session boundary and stays disabled by default; CUDA
  timings synchronize at phase boundaries, while stateless acceleration/export entry points reject tracking before
  performing imports, copies, tracing, or mutation.

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

## 2026-07-27 - US-010
- Added a shared sequence-transform adapter that replays Python, NumPy, PyTorch, and transform-owned random state across
  every frame while advancing ambient randomness as a single augmentation draw.
- Added chronological, time-major sequence collation with ordinary per-step `NestedTensor` batches, aligned targets,
  backbone block-size padding, validation, and a picklable DataLoader factory.
- Files changed: `src/rfdetr/datasets/video.py`, `src/rfdetr/datasets/__init__.py`,
  `tests/datasets/test_video.py`, `.ralph-tui/progress.md`.
- **Learnings:**
  - Replaying only process-global generators is insufficient for Albumentations-style transforms that own private RNGs;
    cloning the pre-call transform state makes those decisions shared while the live pipeline advances once per clip.
  - Time-major collation provides a small interface over the existing image collator: each recurrent step still receives
    the same padded `[batch, C, H, W]` contract and its targets remain in matching batch order.
  - Ordinary pytest collection remains blocked by the host Transformers package lacking `BackboneConfigMixin`, and `uv`
    cannot use its read-only cache. Dependency-isolated behavioral checks, Python compilation, Ruff lint/format, and
    `git diff --check` pass.
---

## 2026-07-27 - US-011
- Added immutable per-frame sequence assignment results that separate fixed continuing correspondences, residual
  discovery matches, absent persistent slots, and the updated slot identity table.
- Bound visible continuing IDs directly to prior query positions and restricted ordinary Hungarian matching to
  identity-free discovery slots plus targets not represented by any persistent slot.
- Covered synthetic geometry crossings, absent continuations, newborn activation, uniqueness, and exclusion of
  continuing targets from discovery matching.
- Files changed: `src/rfdetr/models/matcher.py`, `tests/models/test_matcher.py`, `.ralph-tui/progress.md`.
- **Learnings:**
  - Residual matching can reuse all existing RF-DETR cost terms by slicing query- and instance-aligned tensors before
    invoking the ordinary matcher, keeping identity policy independent from detection costs.
  - Unknown target identities may receive per-frame discovery supervision but must remain `None` in the next slot table
    so missing identity knowledge is never synthesized into a temporal correspondence.
  - Ordinary pytest collection remains blocked by the host Transformers package lacking `BackboneConfigMixin`, and `uv`
    cannot use its read-only cache. Dependency-isolated assignment checks, Python compilation, Ruff lint/format, and
    `git diff --check` pass.
---

## 2026-07-27 - US-012
- Added `TrackingSetCriterion`, which evaluates existing RF-DETR classification, box, GIoU, and auxiliary decoder
  losses with precomputed identity-aware sequence assignments while retaining ordinary encoder proposal matching.
- Kept absent/suspended persistent slots unmatched so existing focal classification supervises them as negatives
  without box regression, and preserved the existing clamped target-count normalization for empty/varying frames.
- Files changed: `src/rfdetr/models/criterion.py`, `src/rfdetr/models/__init__.py`,
  `tests/models/test_criterion.py`, `.ralph-tui/progress.md`.
- **Learnings:**
  - One decoder assignment must be shared across auxiliary layers; rematching auxiliary predictions would silently
    weaken the fixed-slot identity constraint even if the final decoder layer remained identity-aware.
  - Existing unmatched-query focal supervision already provides the required absent-slot classification treatment;
    excluding absent slots from assignment is sufficient to prevent regression against nonexistent targets.
  - The dependency-isolated focused criterion suite passes (12 tests). Ordinary collection remains blocked by the host
    Transformers package lacking `BackboneConfigMixin`; Python compilation, Ruff lint/format, and `git diff --check`
    pass, while `uv` cannot use its read-only global cache.
---

## 2026-07-27 - US-013
- Added causal time-major clip unrolling to Lightning training and validation, with an explicit empty state at each clip
  boundary, per-frame identity-aware assignment, frame-mean loss aggregation, and flattened validation postprocessing.
- Carried predicted decoder features and boxes between frames, preserved trusted state for absent/weak continuations,
  supported explicit temporal detachment, and made validation use inference-like confidence gating.
- Selected `TrackingSetCriterion` for tracking-enabled builds and replayed one multi-scale resize across every frame in a
  clip while leaving ordinary image steps unchanged.
- Files changed: `src/rfdetr/training/module_model.py`, `src/rfdetr/models/lwdetr.py`,
  `tests/training/test_module_model.py`, `tests/models/test_builders.py`, `.ralph-tui/progress.md`.
- **Learnings:**
  - A clip recurrent state must be a local variable initialized explicitly from the first time-major batch; using model
    or Lightning-module attributes would leak graphs and identities across independent clips.
  - Assignment-guided lifecycle still carries model-predicted features and boxes: ground truth controls only slot role
    and identity, while absent slots preserve the preceding trusted prediction.
  - The transitional builder namespace deliberately omits nested capability configs, so the tracking criterion switch
    belongs in `build_criterion_from_config` rather than the shared namespace compatibility surface.
  - Focused pytest remains blocked by the host environment: the global Transformers lacks its public backbone exports,
    `deprecate` is absent, and the training environment lacks `pytorch_lightning`; the checked-in `.venv` also lacks
    pytest. Python compilation, Ruff lint/format, and `git diff --check` pass; `pre-commit` and typecheck tools are not
    installed.
---

## 2026-07-27 - US-014
- Verified that recurrent tracking introduces no checkpoint-only parameters: existing detection state dicts load
  strictly into the tracking path, and tracking state dicts load strictly back into ordinary image inference.
- Added a regression test that round-trips weights in both directions and confirms stateless image predictions match
  empty-state tracking predictions.
- Files changed: `tests/models/test_lwdetr_tracking.py`, `.ralph-tui/progress.md`.
- **Learnings:**
  - Checkpoint compatibility follows from keeping recurrent tensors explicit runtime inputs and candidate outputs,
    rather than learned model members; detection and tracking therefore retain one identical parameter namespace.
  - Image compatibility is preserved by leaving `LWDETR.forward()` stateless while `forward_tracking()` delegates to
    the same detection graph with an explicit empty state.
  - Focused pytest collection remains blocked by the host Transformers package lacking `BackboneConfigMixin`, even
    with a writable isolated uv cache. Python compilation, Ruff lint/format, and `git diff --check` pass;
    `pre-commit` and `mypy` are unavailable.
---

## 2026-07-27 - US-015
- Added validated sequence prediction records containing explicit sequence, source frame, absolute box, class, score,
  and sequence-local track identity.
- Added an evaluation accumulator with explicit reset boundaries, identity-free per-frame detection records,
  per-sequence TrackEval-compatible MOTChallenge export, and lifecycle/false-track diagnostics.
- Files changed: `src/rfdetr/evaluation/sequence.py`, `src/rfdetr/evaluation/__init__.py`,
  `tests/evaluation/test_sequence.py`, `.ralph-tui/progress.md`.
- **Learnings:**
  - Track IDs are session-local, so MOT export must produce a separate stream for every sequence instead of flattening
    records into one namespace; equal numeric IDs after a reset are then unambiguous.
  - Association identity can be removed from the same rich records to preserve ordinary frame-level detection scoring
    without maintaining a second prediction path.
  - False-track creation is evaluator-derived: an unmatched lifecycle birth is recorded separately from the host
    lifecycle event, while recovery, suspension, termination, and duplicate suppression remain policy diagnostics.
  - Ordinary pytest collection remains blocked by the host Transformers package lacking `BackboneConfigMixin`.
    Dependency-isolated behavioral checks, Python compilation, Ruff lint/format, and `git diff --check` pass;
    `pre-commit` and typecheck tools are unavailable.
---

## 2026-07-27 - US-017
- Added opt-in synchronized tracking-session timing with preprocessing, neural forward, lifecycle, output, total, and
  derived host tracking-overhead measurements exposed both on the session and returned detection metadata.
- Rejected tracking-enabled `optimize_for_inference()` and ONNX/TFLite `export()` calls before any stateless graph
  construction or model mutation, with actionable eager-session guidance.
- Kept timing disabled by default so existing streaming behavior has no clock or CUDA-synchronization overhead.
- Files changed: `src/rfdetr/config.py`, `src/rfdetr/detr.py`, `src/rfdetr/tracking/session.py`,
  `src/rfdetr/tracking/__init__.py`, `tests/models/test_tracking_session.py`,
  `tests/inference/test_optimize_for_inference.py`, `tests/export/test_export.py`, `.ralph-tui/progress.md`.
- **Learnings:**
  - Accurate GPU phase measurements require explicit device synchronization because CUDA kernels are asynchronous;
    making measurement opt-in prevents that synchronization from distorting normal deployment latency.
  - Unsupported stateful capabilities must be rejected at the public acceleration boundary, not only when a session is
    later updated, so export/optimization cannot produce a seemingly valid stateless artifact.
  - Ordinary pytest collection remains blocked by the host Transformers package lacking `BackboneConfigMixin`.
    Python compilation, Ruff lint/format, and `git diff --check` pass; `pre-commit` and `mypy` are unavailable.
---
