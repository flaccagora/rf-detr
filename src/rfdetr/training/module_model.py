# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LightningModule for RF-DETR training and validation."""

from __future__ import annotations

import math
import random
import time
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, get_args

import torch
import torch.nn.functional as F  # noqa: N812 -- project-conventional alias (see AGENTS.md)
from pytorch_lightning import LightningModule, seed_everything

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import ClassSchema, ModelConfig, TrainConfig
from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
from rfdetr.models.matcher import SequenceAssignment, identity_aware_sequence_assignment
from rfdetr.models.tracking import TrackingFrameOutput, TrackQueryState
from rfdetr.models.weights import apply_lora, interpolate_position_embeddings, load_pretrain_weights
from rfdetr.tracking.lifecycle import EventKind, TrackSlotTable, transition_lifecycle
from rfdetr.training.checkpoint import authoritative_checkpoint_metadata, source_checkpoint_hash
from rfdetr.training.param_groups import get_param_dict
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy, box_iou
from rfdetr.utilities.logger import get_logger

logger = get_logger()

_FALSE_POSITIVE_INJECTION_SALT = 1
_QUERY_DROPOUT_SALT = 2
_ERROR_EXPOSURE_LOGIT_MAGNITUDE = 8.0

# Duplicate-FP PRD US-011 sampling composition: a supervised persistent transition counts as a
# "partial visibility" event when a track that stays present across the transition shrinks below
# this fraction of the largest normalized box area it reached earlier in the clip's supervised
# suffix -- an annotation-free occlusion proxy for datasets (DanceTrack) that carry no visibility
# field.
_PARTIAL_VISIBILITY_AREA_FRACTION = 0.5

# Every sampling-composition transition category reported per training step (US-011 AC).
_SAMPLING_TRANSITION_KINDS: tuple[str, ...] = (
    "departure",
    "re_entry",
    "partial_visibility",
    "injected_fp",
    "query_dropout",
)

# Every prediction-driven lifecycle transition kind, logged per training step so
# incorrect births, missed continuations, suspensions, confirmations, and
# terminations are visible during training (duplicate-FP PRD US-009).
_LIFECYCLE_EVENT_KINDS: tuple[str, ...] = get_args(EventKind)

_TRAIN_PROGRESS_LOSS_ALIASES: dict[str, str] = {
    "loss_ce": "loss_cls",
    "loss_bbox": "loss_box",
    "loss_giou": "loss_giou",
    "loss_mask_ce": "mask_ce",
    "loss_mask_dice": "mask_dice",
    "loss_keypoints_l1": "kp_l1",
    "loss_keypoints_findable": "kp_find",
    "loss_keypoints_visible": "kp_vis",
    "loss_keypoints_nll": "kp_nll",
}


class RFDETRModelModule(LightningModule):
    """LightningModule wrapping the RF-DETR model and training loop.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        train_config.validate_for_model(model_config)
        # Manual optimization is enabled only for keypoint models so that the box-count
        # normalizer can be accumulated across grad-accum microbatches. Detection and
        # segmentation use Lightning's automatic optimization (PTL handles accumulation,
        # AMP, and gradient clipping), which keeps their step semantics unchanged from
        # the pre-fix/scaling behaviour.
        self._use_manual_optimization: bool = bool(getattr(model_config, "use_grouppose_keypoints", False))
        self.automatic_optimization = not self._use_manual_optimization
        self._accumulated_box_normalizer: torch.Tensor | None = None
        # Decoded-frame LR progress (PRD Section 7.6): frames forwarded since the last
        # scheduler step, the running total, and the per-optimizer-step frame reference
        # used to convert that total into the step-equivalent domain configure_optimizers()'
        # lr_lambda already expects. See _decoded_frame_count / _step_lr_scheduler.
        self._pending_decoded_frames: int = 0
        self._decoded_frames_seen: float = 0.0
        self._decoded_frames_per_optimizer_step: float = 1.0
        # Allow partial state-dict loading when resuming from a .pth checkpoint
        # (which contains only model weights, not criterion/postprocess state).
        self.strict_loading = False

        # Duplicate-FP PRD US-009 lifecycle-commitment diagnostics, recomputed
        # per training step. ``_lifecycle_commit_frames`` counts supervised
        # frame-item commits by curriculum stage; ``_lifecycle_transition_counts``
        # tallies every prediction-driven lifecycle event kind. Both are reset at
        # the start of every ``_unroll_tracking_clip`` and logged by
        # ``_training_step_tracking``.
        self._lifecycle_commit_frames: dict[str, int] = {"oracle": 0, "prediction_driven": 0}
        self._lifecycle_transition_counts: Counter[str] = Counter()

        # Duplicate-FP PRD US-010 error-exposure diagnostics, recomputed per training step.
        # ``_error_exposure_diagnostics`` accumulates attempted/applied counts plus running
        # sums for score, ground-truth overlap, capacity pressure, and lifetime;
        # ``_injected_track_records`` follows each injected false-positive track across the
        # clip so its lifetime and cancellation outcome can be reported. Both are reset at
        # the start of every ``_unroll_tracking_clip`` and logged by ``_training_step_tracking``.
        self._error_exposure_diagnostics: dict[str, float] = self._empty_error_exposure_diagnostics()
        self._injected_track_records: dict[tuple[int, int], dict[str, Any]] = {}

        # Duplicate-FP PRD US-011 sampling composition, recomputed per training step. Counts how
        # many of this step's supervised persistent transitions (one per supervised frame-step
        # after the first, aggregated over batch items) contained a track departure, a track
        # re-entry, a partial-visibility shrink, an injected false-positive slot, or a dropped
        # genuine query -- so a longer-horizon clip's realized error exposure is measurable, not
        # assumed. Reset at the start of every ``_unroll_tracking_clip``; logged by
        # ``_training_step_tracking``.
        self._sampling_transition_diagnostics: dict[str, float] = self._empty_sampling_transition_diagnostics()

        # Model, criterion, and postprocessor.
        self.model = build_model_from_config(model_config, train_config)
        if model_config.pretrain_weights is not None:
            # Canonical loader handles PE interpolation, PTL .ckpt normalisation,
            # per-group query slicing, class-name extraction, partial-load warnings,
            # and writes any auto-aligned ``num_classes`` back onto ``model_config``.
            load_pretrain_weights(self.model, self.model_config)
            if model_config.use_grouppose_keypoints:
                # Older model shims may omit the keypoint reset hook; call it only when implemented.
                reset_keypoint_gaussian_parameters = getattr(self.model, "reset_keypoint_gaussian_parameters", None)
                if callable(reset_keypoint_gaussian_parameters):
                    reset_keypoint_gaussian_parameters()
                    logger.info(
                        "Reset keypoint Gaussian precision outputs to unit values after pretrained weight load."
                    )
        if model_config.backbone_lora:
            apply_lora(self.model)

        # Build criterion/postprocessors after potential num_classes alignment so
        # they are constructed with a config that matches the current model head.
        self.criterion, self.postprocess = build_criterion_from_config(self.model_config, self.train_config)
        self._source_checkpoint_hash = source_checkpoint_hash(self.model_config)

        # torch.compile is opt-in: set model_config.compile=True to enable.
        # Only enabled on CUDA; MPS and CPU do not benefit from compilation.
        # Use the fork-safe DEVICE constant instead of torch.cuda.is_available(),
        # which creates a CUDA driver context that breaks fork-based DDP.
        from rfdetr.config import DEVICE

        accelerator = str(train_config.accelerator).lower()
        uses_cuda_accelerator = accelerator in {"auto", "gpu", "cuda"}
        compile_enabled = (
            model_config.compile and DEVICE == "cuda" and uses_cuda_accelerator and not train_config.multi_scale
        )
        if model_config.compile and train_config.multi_scale:
            logger.info("Disabling torch.compile because multi_scale=True introduces dynamic input shapes.")
        if compile_enabled:
            # dynamic=True: one compiled graph handles all multi-scale input sizes instead
            # of recompiling per (H, W) pair. suppress_errors=True: if inductor can't
            # compile a subgraph (e.g. bicubic backward with symbolic shapes), it falls
            # back to eager mode for that subgraph rather than crashing.
            # capture_scalar_outputs=True: include Tensor.item() calls
            # (gen_encoder_output_proposals / ms_deform_attn use spatial-shape .item()
            # as Python slice indices). Safe with dynamic=True because item() results
            # are backed symbols derived from input shapes — not unbacked symbols that
            # would cause PendingUnbackedSymbolNotFound (which only occurs without dynamic).
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.capture_scalar_outputs = True
            self.model = torch.compile(self.model, dynamic=True)

    # ------------------------------------------------------------------
    # PTL lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Seed RNGs at fit start when ``TrainConfig.seed`` is set.

        This avoids hidden global side-effects in ``build_trainer`` while still preserving deterministic training
        behaviour for actual fit runs.
        """
        if self.train_config.seed is not None:
            seed_everything(self.train_config.seed + self.global_rank, workers=True)

    def on_train_batch_start(self, batch: tuple, batch_idx: int) -> None:
        """Apply optional multi-scale resize to the incoming batch.

        Modifications to ``batch`` (in-place on ``NestedTensor``) are visible in ``training_step`` because they share
        the same object.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Index of the current batch within the epoch.
        """
        tc = self.train_config
        mc = self.model_config

        if tc.multi_scale and not tc.do_random_resize_via_padding:
            samples, _ = batch
            scales = compute_multi_scale_scales(mc.resolution, tc.expanded_scales, mc.patch_size, mc.num_windows)
            step = self.trainer.global_step
            # Use a step-local generator so the scale choice is deterministic and DDP-consistent
            # without reseeding the process-global RNG on every batch.
            scale = random.Random(step).choice(scales)
            with torch.no_grad():
                sample_batches = samples if isinstance(samples, (tuple, list)) else (samples,)
                for frame_samples in sample_batches:
                    frame_samples.tensors = F.interpolate(
                        frame_samples.tensors, size=scale, mode="bilinear", align_corners=False
                    )
                    frame_samples.mask = (
                        F.interpolate(frame_samples.mask.unsqueeze(1).float(), size=scale, mode="nearest")
                        .squeeze(1)
                        .bool()
                    )

    def on_train_epoch_start(self) -> None:
        """Reset the accumulated box normalizer at the start of every training epoch.

        Lightning may reuse the module across epochs without calling ``_step_optimizer`` at the boundary (for example
        when an epoch ends mid-accumulation window with a non-divisible batch count). Clearing the accumulator here
        guarantees the manual-optimization path always starts each epoch from a known state, so the first microbatch's
        gradients are scaled by its own box count and not by a stale previous-epoch denominator.

        This is a no-op for non-keypoint models because they use Lightning's automatic optimization path and never
        populate ``self._accumulated_box_normalizer``.

        Note: on finite datasets the final-batch fallback in ``_should_step_optimizer`` always flushes a partial
        trailing window, so this reset is the only change needed.  On IterableDatasets (infinite
        ``num_training_batches``) a partial window may survive epoch end with un-stepped gradients; those are
        discarded here and the optimizer is zeroed so the first microbatch of the new epoch starts from a clean state.
        """
        if self._accumulated_box_normalizer is not None:
            # Discard any partial accumulation window that survived the epoch boundary
            # (only possible for IterableDatasets where num_training_batches is infinite).
            try:
                opts = self.optimizers()
                for opt in opts if isinstance(opts, list) else [opts]:
                    opt.zero_grad()
            except RuntimeError:
                pass  # Not attached to Trainer (unit-test context); nothing to zero.
        self._accumulated_box_normalizer = None

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor | dict[str, Any]:
        """Compute loss for one training step and log metrics.

        PTL handles AMP (``precision``) without a manual ``GradScaler``. Keypoint models perform manual optimization so
        box-count loss normalization is based on the full accumulated effective batch rather than each microbatch
        independently; detection and segmentation models keep Lightning's automatic optimization path.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the epoch.

        Returns:
            Scalar loss tensor by default. When ``compute_train_metrics=True``,
            returns a Lightning-compatible dict containing ``loss`` plus
            detached postprocessed predictions for train mAP logging.
        """
        samples, targets = batch
        self._pending_decoded_frames += self._decoded_frame_count(samples, targets)
        if self.model_config.tracking.enabled and isinstance(samples, (tuple, list)):
            return self._training_step_tracking(samples, targets, batch_idx)
        batch_size = len(targets)
        outputs = self.model(samples, targets)
        if self._use_manual_optimization:
            loss_dict, raw_loss, normalizer = self._compute_train_losses(outputs, targets)
            loss_for_backward = self._scale_loss_for_accumulation(raw_loss, normalizer)
        else:
            loss_dict = self.criterion(outputs, targets)
            loss_for_backward = None
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        # Automatic optimization path: divide by accumulate_grad_batches so the accumulated
        # gradient matches a single large batch, matching the legacy engine.  PTL accumulates
        # full-scale gradients by default; dividing here keeps the effective LR identical.
        accumulate_grad_batches = max(1, int(self.trainer.accumulate_grad_batches))
        loss_for_return = loss if self._use_manual_optimization else loss / accumulate_grad_batches
        train_log_sync_dist = bool(self.train_config.train_log_sync_dist)
        train_log_on_step = bool(self.train_config.train_log_on_step)
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        self.log(
            "train/loss",
            loss,
            prog_bar=False,
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        self._log_train_progress_metrics(loss, loss_dict, batch_size=batch_size)
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        # Optimizer may have multiple param groups with different LRs (e.g., backbone/decoder).
        # Preserve the first group's LR for backward compatibility, but also log the
        # min/max across all groups so the progress bar reflects the full schedule.
        group_lrs = [pg["lr"] for pg in optimizer.param_groups if "lr" in pg]
        if group_lrs:
            base_lr = group_lrs[0]
            min_lr = min(group_lrs)
            max_lr = max(group_lrs)
            self.log("train/lr", base_lr, prog_bar=False, on_step=True, on_epoch=False)
            self.log("train/lr_min", min_lr, prog_bar=False, on_step=True, on_epoch=False)
            self.log("train/lr_max", max_lr, prog_bar=False, on_step=True, on_epoch=False)
        if self._use_manual_optimization:
            self.manual_backward(loss_for_backward)
            if self._should_step_optimizer(batch_idx):
                self._step_optimizer(optimizer)
        if self.train_config.compute_train_metrics:
            with torch.no_grad():
                orig_sizes = torch.stack([t["orig_size"] for t in targets])
                # Slice to group-0 queries only — mirrors the eval-mode path in
                # lwdetr.py that trims refpoint_embed to [:num_queries]. Without
                # this, training mode emits group_detr×num_queries queries (e.g.
                # 13×300=3900) and postprocess top-k selection draws from all
                # groups, producing OKS/mAP values ~50× below true accuracy.
                nq = self.model_config.num_queries
                # Only include tensor-valued keys — pred_masks is a dict in
                # train mode (sparse_forward) and postprocess cannot handle it.
                inference_outputs = {
                    k: v[:, :nq] if v.ndim >= 2 else v
                    for k, v in outputs.items()
                    if k in ("pred_logits", "pred_boxes", "pred_masks", "pred_keypoints")
                    and isinstance(v, torch.Tensor)
                }
                results = self.postprocess(inference_outputs, orig_sizes)
            return {
                "loss": loss_for_return.detach() if self._use_manual_optimization else loss_for_return,
                "results": self._detach_results(results),
                "targets": targets,
            }
        return loss_for_return.detach() if self._use_manual_optimization else loss_for_return

    @staticmethod
    def _tracking_outputs(frame: TrackingFrameOutput) -> dict[str, Any]:
        """Convert a structured tracking frame into the existing criterion contract."""
        outputs: dict[str, Any] = {
            "pred_logits": frame.pred_logits,
            "pred_boxes": frame.pred_boxes,
        }
        if frame.aux_outputs:
            outputs["aux_outputs"] = list(frame.aux_outputs)
        if frame.enc_outputs is not None:
            outputs["enc_outputs"] = frame.enc_outputs
        return outputs

    @staticmethod
    def _slice_frame_output(frame: TrackingFrameOutput, batch_index: int) -> TrackingFrameOutput:
        """Select one batch item's tracking output as a single-stream frame.

        ``transition_lifecycle`` operates on exactly one stream at a time (it is the same
        pure function ``TrackingSession`` calls during deployment), so a batched training
        clip must be split into per-item single-stream frames before each lifecycle update.
        """
        candidate = frame.candidate_state
        return TrackingFrameOutput(
            pred_logits=frame.pred_logits[batch_index : batch_index + 1],
            pred_boxes=frame.pred_boxes[batch_index : batch_index + 1],
            candidate_state=TrackQueryState(
                candidate.query_features[batch_index : batch_index + 1],
                candidate.reference_boxes[batch_index : batch_index + 1],
                candidate.active_mask[batch_index : batch_index + 1],
            ),
            input_active_mask=frame.input_active_mask[batch_index : batch_index + 1],
        )

    def _commit_tracking_state_assignment_guided(
        self,
        frame: TrackingFrameOutput,
        assignments: list[SequenceAssignment],
        prior_state: TrackQueryState,
        prior_track_ids: list[tuple[int | None, ...]],
    ) -> tuple[TrackQueryState, list[tuple[int | None, ...]]]:
        """Commit predicted tensors with ground-truth-driven identity.

        This is the matched control described by PRD US-018/US-022: whether a query keeps
        or gains a track identity follows ground-truth assignment rather than the model's
        own confidence, so it isolates the effect of inference-like state commitment in
        head-to-head training comparisons.
        """
        candidate = frame.candidate_state
        features = candidate.query_features.clone()
        boxes = candidate.reference_boxes.clone()
        next_ids: list[tuple[int | None, ...]] = []
        active_masks: list[torch.Tensor] = []
        capacity = self.model_config.tracking.active_capacity(self.model_config.num_queries)

        for batch_index, (assignment, old_ids) in enumerate(zip(assignments, prior_track_ids, strict=True)):
            committed_ids = list(old_ids)
            visible_queries = set(assignment.decoder_indices[0].tolist())
            discovery_pairs = zip(
                assignment.discovery_indices[0].tolist(), assignment.discovery_indices[1].tolist(), strict=True
            )
            target_ids = list(assignment.slot_track_ids)

            for query_index in range(len(old_ids)):
                if old_ids[query_index] is not None and query_index not in visible_queries:
                    features[batch_index, query_index] = prior_state.query_features[batch_index, query_index]
                    boxes[batch_index, query_index] = prior_state.reference_boxes[batch_index, query_index]

            for query_index, _ in discovery_pairs:
                proposed_id = target_ids[query_index]
                can_activate = proposed_id is not None and sum(value is not None for value in committed_ids) < capacity
                if can_activate:
                    committed_ids[query_index] = proposed_id

            next_ids.append(tuple(committed_ids))
            active_masks.append(torch.tensor([value is not None for value in committed_ids], device=features.device))

        return TrackQueryState(features, boxes, torch.stack(active_masks).bool()), next_ids

    def _error_exposure_draws(
        self, *, salt: int, frame_index: int, batch_index: int, query_indices: Sequence[int]
    ) -> list[float]:
        """One deterministic uniform draw per query index (duplicate-FP PRD US-010).

        Each draw is seeded independently from ``TrackingTrainConfig.error_exposure_seed``
        plus ``(salt, frame_index, batch_index, query_index)`` -- never from a shared
        per-frame vector or from global RNG state -- so a slot's draw is identical
        regardless of how many other slots are eligible that frame, in what order they are
        considered, or what randomness earlier training code consumed.
        """
        base_seed = self.train_config.tracking.error_exposure_seed
        prefix = ((base_seed * 1_000_003 + salt) * 1_000_003 + frame_index) * 1_000_003 + batch_index
        draws: list[float] = []
        for query_index in query_indices:
            generator = torch.Generator(device="cpu")
            generator.manual_seed((prefix * 1_000_003 + query_index) % (2**63 - 1))
            draws.append(float(torch.rand(1, generator=generator).item()))
        return draws

    def _false_positive_injection_plan(
        self,
        table: TrackSlotTable,
        assignment: SequenceAssignment,
        *,
        frame_index: int,
        batch_index: int,
        probability: float,
        max_count: int,
    ) -> tuple[list[int], list[int], list[int]]:
        """Return ``(eligible, attempted, selected)`` query indices for false-positive injection.

        ``eligible`` are query positions with no correspondence at all in ``assignment``
        (neither continuing nor discovery -- genuine "unmatched query states"), restricted to
        currently inactive slots so injection creates a new synthetic false-positive track
        rather than altering an already-tracked identity. ``attempted`` are the eligible slots
        whose per-query draw falls under ``probability``; ``selected`` keeps the lowest-drawn
        ``max_count`` of those (the remaining per-sample budget). All three are deterministic
        given the seed.
        """
        matched = set(assignment.decoder_indices[0].tolist())
        eligible = [
            index for index, slot in enumerate(table.slots) if slot.status == "inactive" and index not in matched
        ]
        if not eligible:
            return [], [], []
        draws = self._error_exposure_draws(
            salt=_FALSE_POSITIVE_INJECTION_SALT,
            frame_index=frame_index,
            batch_index=batch_index,
            query_indices=eligible,
        )
        ranked = sorted(zip(eligible, draws), key=lambda pair: pair[1])
        attempted = [index for index, draw in ranked if draw < probability]
        selected = attempted[:max_count] if max_count > 0 else []
        return eligible, attempted, selected

    def _select_false_positive_injection_indices(
        self,
        table: TrackSlotTable,
        assignment: SequenceAssignment,
        *,
        frame_index: int,
        batch_index: int,
        probability: float,
        max_count: int,
    ) -> list[int]:
        """Select ground-truth-unmatched inactive slots to force-activate this frame."""
        return self._false_positive_injection_plan(
            table,
            assignment,
            frame_index=frame_index,
            batch_index=batch_index,
            probability=probability,
            max_count=max_count,
        )[2]

    def _query_dropout_plan(
        self,
        table: TrackSlotTable,
        *,
        frame_index: int,
        batch_index: int,
        probability: float,
    ) -> tuple[list[int], list[int], list[int]]:
        """Return ``(candidates, attempted, selected)`` query indices for query dropout.

        ``candidates`` are currently active slots; ``attempted`` are those whose per-query
        draw falls under ``probability``; ``selected`` is ``attempted`` minus the guard-spared
        survivor (the highest-drawn candidate) whenever every candidate was attempted, so a
        sample can never lose all active identities to dropout in one frame.
        """
        candidates = [index for index, slot in enumerate(table.slots) if slot.status == "active"]
        if not candidates:
            return [], [], []
        draws = self._error_exposure_draws(
            salt=_QUERY_DROPOUT_SALT,
            frame_index=frame_index,
            batch_index=batch_index,
            query_indices=candidates,
        )
        paired = list(zip(candidates, draws))
        attempted = [index for index, draw in paired if draw < probability]
        selected = list(attempted)
        if len(selected) == len(candidates):
            survivor_index = max(paired, key=lambda pair: pair[1])[0]
            selected = [index for index in selected if index != survivor_index]
        return candidates, attempted, selected

    def _select_query_dropout_indices(
        self,
        table: TrackSlotTable,
        *,
        frame_index: int,
        batch_index: int,
        probability: float,
    ) -> list[int]:
        """Select active slots to force a missed-detection score this frame."""
        return self._query_dropout_plan(
            table, frame_index=frame_index, batch_index=batch_index, probability=probability
        )[2]

    def _inject_false_positive_scores(
        self, frame: TrackingFrameOutput, indices: Sequence[int], class_schema: ClassSchema
    ) -> TrackingFrameOutput:
        """Force selected inactive slots' foreground scores well above any activation threshold."""
        if not indices:
            return frame
        pred_logits = frame.pred_logits.clone()
        target_class = class_schema.foreground_class_ids[0]
        for index in indices:
            pred_logits[0, index, :] = -_ERROR_EXPOSURE_LOGIT_MAGNITUDE
            pred_logits[0, index, target_class] = _ERROR_EXPOSURE_LOGIT_MAGNITUDE
        return replace(frame, pred_logits=pred_logits)

    def _apply_query_dropout_scores(
        self, frame: TrackingFrameOutput, indices: Sequence[int], class_schema: ClassSchema
    ) -> TrackingFrameOutput:
        """Force selected active slots' foreground scores well below any continuation threshold."""
        if not indices:
            return frame
        pred_logits = frame.pred_logits.clone()
        for index in indices:
            for class_id in class_schema.foreground_class_ids:
                pred_logits[0, index, class_id] = -_ERROR_EXPOSURE_LOGIT_MAGNITUDE
            if class_schema.background_logit_index is not None:
                pred_logits[0, index, class_schema.background_logit_index] = _ERROR_EXPOSURE_LOGIT_MAGNITUDE
        return replace(frame, pred_logits=pred_logits)

    @staticmethod
    def _empty_error_exposure_diagnostics() -> dict[str, float]:
        """Zeroed accumulator for one training step's error-exposure diagnostics (US-010)."""
        return {
            "fp_injection_attempted": 0.0,
            "fp_injection_applied": 0.0,
            "fp_injection_tracked": 0.0,
            "fp_injection_cancelled": 0.0,
            "fp_injection_score_sum": 0.0,
            "fp_injection_target_iou_sum": 0.0,
            "fp_injection_capacity_pressure_sum": 0.0,
            "fp_injection_lifetime_sum": 0.0,
            "query_dropout_attempted": 0.0,
            "query_dropout_applied": 0.0,
            "query_dropout_score_sum": 0.0,
            "query_dropout_target_iou_sum": 0.0,
            "query_dropout_lifetime_sum": 0.0,
        }

    def _error_exposure_curriculum_factor(self) -> float:
        """Current warm-up-then-ramp multiplier for the error-exposure sampling rates (US-010).

        Deterministic from ``trainer.current_epoch`` / ``trainer.global_step`` so repeated runs
        and resumes agree. ``mode="disabled"`` (the default) always returns ``1.0``, leaving
        every pre-existing configuration unchanged.
        """
        curriculum = self.train_config.tracking.error_exposure_curriculum
        trainer = getattr(self, "_trainer", None)
        return curriculum.factor_at(
            epoch=int(getattr(trainer, "current_epoch", 0) or 0),
            step=int(getattr(trainer, "global_step", 0) or 0),
        )

    def _error_exposure_slot_metrics(
        self,
        base_frame: TrackingFrameOutput,
        query_index: int,
        gt_boxes_xyxy: torch.Tensor | None,
        class_schema: ClassSchema,
    ) -> tuple[float, float]:
        """Return ``(foreground_score, max_ground_truth_iou)`` for one perturbed slot (US-010).

        The score and box are read from ``base_frame`` -- the model's *un-perturbed* output for
        this slot -- so the logged diagnostics describe the genuine prediction that injection or
        dropout overrode, not the synthetic magnitude written on top of it.
        """
        probs = base_frame.pred_logits[0, query_index].detach().softmax(-1)
        foreground_score = float(probs[list(class_schema.foreground_class_ids)].max())
        if gt_boxes_xyxy is None or gt_boxes_xyxy.numel() == 0:
            return foreground_score, 0.0
        pred_xyxy = box_cxcywh_to_xyxy(base_frame.pred_boxes[0, query_index].detach().unsqueeze(0))
        iou, _ = box_iou(pred_xyxy, gt_boxes_xyxy)
        return foreground_score, float(iou.max())

    def _commit_tracking_state_inference_like(
        self,
        frame: TrackingFrameOutput,
        assignments: list[SequenceAssignment],
        prior_state: TrackQueryState,
        tables: list[TrackSlotTable],
        *,
        frame_index: int,
        fp_injection_remaining: list[int] | None = None,
        record_diagnostics: bool = False,
        frame_targets: Sequence[Any] | None = None,
    ) -> tuple[TrackQueryState, list[tuple[int | None, ...]], list[TrackSlotTable]]:
        """Commit state through the exact deployment lifecycle state machine.

        Ground-truth assignment plays no role in the committed lifecycle decisions themselves:
        every activation, suspension, recovery, and termination follows
        :func:`transition_lifecycle`, the same pure function ``TrackingSession`` calls at
        inference, driven only by the model's own foreground scores. This is what lets training
        see -- and learn to recover from -- the model's own false positives, false negatives,
        and stale suspended references instead of having ground truth silently repair them.

        ``assignments`` is consulted only by the optional error-exposure pilots
        (``false_positive_injection_enabled`` / ``query_dropout_enabled``, duplicate-FP PRD
        US-010), which perturb the *candidate scores* fed into ``transition_lifecycle`` -- never
        the lifecycle decision itself -- to select which ground-truth-unmatched candidates get
        force-activated or which active slots get force-dropped this frame, deterministically.
        ``frame_targets`` supplies ground-truth boxes for the target-overlap diagnostic only.
        """
        if self.model_config.class_schema is None:
            raise ValueError("prediction-driven tracking requires an authoritative class_schema")
        class_schema = self.model_config.class_schema
        tracking_config = self.train_config.tracking
        lifecycle_config = tracking_config.lifecycle
        capacity = self.model_config.tracking.active_capacity(self.model_config.num_queries)
        exposure_factor = self._error_exposure_curriculum_factor()
        fp_probability = tracking_config.false_positive_injection_probability * exposure_factor
        dropout_probability = tracking_config.query_dropout_probability * exposure_factor
        diagnostics = self._error_exposure_diagnostics

        next_tables: list[TrackSlotTable] = []
        next_ids: list[tuple[int | None, ...]] = []
        features: list[torch.Tensor] = []
        boxes: list[torch.Tensor] = []
        active_masks: list[torch.Tensor] = []

        for batch_index, table in enumerate(tables):
            item_state = TrackQueryState(
                prior_state.query_features[batch_index : batch_index + 1],
                prior_state.reference_boxes[batch_index : batch_index + 1],
                prior_state.active_mask[batch_index : batch_index + 1],
            )
            base_frame = self._slice_frame_output(frame, batch_index)
            item_frame = base_frame
            gt_boxes_xyxy: torch.Tensor | None = None
            if frame_targets is not None:
                gt_boxes = frame_targets[batch_index]["boxes"].detach()
                gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes) if gt_boxes.numel() else gt_boxes.new_zeros((0, 4))
            active_count = sum(1 for slot in table.slots if slot.status == "active")
            capacity_pressure = active_count / capacity if capacity else 0.0
            injected: list[int] = []

            if tracking_config.false_positive_injection_enabled and fp_injection_remaining is not None:
                eligible, attempted, injected = self._false_positive_injection_plan(
                    table,
                    assignments[batch_index],
                    frame_index=frame_index,
                    batch_index=batch_index,
                    probability=fp_probability,
                    max_count=fp_injection_remaining[batch_index],
                )
                if record_diagnostics:
                    diagnostics["fp_injection_attempted"] += len(attempted)
                if injected:
                    item_frame = self._inject_false_positive_scores(item_frame, injected, class_schema)
                    fp_injection_remaining[batch_index] -= len(injected)
                    if record_diagnostics:
                        diagnostics["fp_injection_applied"] += len(injected)
                        for query_index in injected:
                            score, overlap = self._error_exposure_slot_metrics(
                                base_frame, query_index, gt_boxes_xyxy, class_schema
                            )
                            diagnostics["fp_injection_score_sum"] += score
                            diagnostics["fp_injection_target_iou_sum"] += overlap
                            diagnostics["fp_injection_capacity_pressure_sum"] += capacity_pressure

            if tracking_config.query_dropout_enabled:
                _candidates, dropout_attempted, dropped = self._query_dropout_plan(
                    table,
                    frame_index=frame_index,
                    batch_index=batch_index,
                    probability=dropout_probability,
                )
                if record_diagnostics:
                    diagnostics["query_dropout_attempted"] += len(dropout_attempted)
                if dropped:
                    item_frame = self._apply_query_dropout_scores(item_frame, dropped, class_schema)
                    if record_diagnostics:
                        diagnostics["query_dropout_applied"] += len(dropped)
                        for query_index in dropped:
                            score, overlap = self._error_exposure_slot_metrics(
                                base_frame, query_index, gt_boxes_xyxy, class_schema
                            )
                            diagnostics["query_dropout_score_sum"] += score
                            diagnostics["query_dropout_target_iou_sum"] += overlap
                            diagnostics["query_dropout_lifetime_sum"] += float(table.slots[query_index].age)

            transition = transition_lifecycle(
                table,
                item_state,
                item_frame,
                lifecycle_config,
                class_schema,
                max_active_tracks=capacity,
                frame_index=frame_index,
            )
            if record_diagnostics:
                self._lifecycle_transition_counts.update(event.kind for event in transition.events)
                self._update_injected_track_records(
                    batch_index, transition.table, injected if fp_injection_remaining is not None else [], frame_index
                )
            next_tables.append(transition.table)
            next_ids.append(transition.table.slot_track_ids)
            features.append(transition.state.query_features)
            boxes.append(transition.state.reference_boxes)
            active_masks.append(transition.state.active_mask)

        state = TrackQueryState(torch.cat(features, dim=0), torch.cat(boxes, dim=0), torch.cat(active_masks, dim=0))
        return state, next_ids, next_tables

    def _update_injected_track_records(
        self,
        batch_index: int,
        table: TrackSlotTable,
        injected_query_indices: Sequence[int],
        frame_index: int,
    ) -> None:
        """Follow each injected false-positive track across the clip for the US-010 diagnostics.

        A record is opened for every injected slot that the lifecycle actually activated. On
        every later frame the record is either extended (the slot is still that active track)
        or closed as *cancelled* -- the lifecycle suspended, terminated, or recycled the
        synthetic track. Records still open at the clip boundary are survivors, not
        cancellations.
        """
        for (record_batch, query_index), record in self._injected_track_records.items():
            if record_batch != batch_index or record["closed"] or record["birth_frame"] >= frame_index:
                continue
            slot = table.slots[query_index]
            if slot.status == "active" and slot.track_id == record["track_id"]:
                record["last_seen_frame"] = frame_index
            else:
                record["closed"] = True
                record["cancelled"] = True

        for query_index in injected_query_indices:
            slot = table.slots[query_index]
            if slot.status == "active" and slot.track_id is not None:
                self._injected_track_records[(batch_index, query_index)] = {
                    "track_id": slot.track_id,
                    "birth_frame": frame_index,
                    "last_seen_frame": frame_index,
                    "closed": False,
                    "cancelled": False,
                }

    def _finalize_error_exposure_diagnostics(self) -> None:
        """Fold the per-clip injected-track records into the logged lifetime/cancellation totals."""
        diagnostics = self._error_exposure_diagnostics
        for record in self._injected_track_records.values():
            diagnostics["fp_injection_tracked"] += 1.0
            diagnostics["fp_injection_lifetime_sum"] += record["last_seen_frame"] - record["birth_frame"] + 1
            if record["cancelled"]:
                diagnostics["fp_injection_cancelled"] += 1.0

    @staticmethod
    def _empty_sampling_transition_diagnostics() -> dict[str, float]:
        """Zeroed accumulator for one training step's sampling-composition report (US-011)."""
        counts = {"supervised_transition_steps": 0.0}
        counts.update({f"{kind}_steps": 0.0 for kind in _SAMPLING_TRANSITION_KINDS})
        return counts

    def _accumulate_sampling_transition_composition(
        self,
        *,
        supervised_targets: Sequence[Sequence[Any]],
        injection_flags: Sequence[bool],
        dropout_flags: Sequence[bool],
    ) -> None:
        """Tally which supervised persistent transitions this clip actually exercised (US-011 AC).

        A supervised persistent transition is one supervised frame-step after the first: the clip
        carries recurrent state from the previous supervised frame into it. For every such step
        this records whether -- across any batch item -- the ground truth shows a track
        **departure** (an identity present last step is gone), a track **re_entry** (an identity
        absent last step reappears having been seen earlier in the suffix), or a
        **partial_visibility** shrink (a surviving identity drops below
        :data:`_PARTIAL_VISIBILITY_AREA_FRACTION` of its earlier peak box area), and whether the
        error-exposure pilots committed an **injected_fp** slot or a **query_dropout** on that
        step. Denominator and numerators are logged so a longer-horizon clip's realized exposure
        to disappearance/re-entry/occlusion is measurable rather than assumed.

        Args:
            supervised_targets: Time-major per-supervised-frame target-dict lists (one list per
                batch item), i.e. ``target_batches`` restricted to the supervised suffix.
            injection_flags: Per-supervised-frame flag, ``True`` when the lifecycle committed at
                least one injected false-positive slot on that frame.
            dropout_flags: Per-supervised-frame flag, ``True`` when the lifecycle forced at least
                one active slot to look dropped on that frame.
        """
        diagnostics = self._sampling_transition_diagnostics
        num_steps = len(supervised_targets)
        if num_steps == 0:
            return
        batch_size = len(supervised_targets[0])

        def _frame_maps(target: Any) -> tuple[set[int], dict[int, float]]:
            boxes = target["boxes"]
            present: set[int] = set()
            areas: dict[int, float] = {}
            for row, raw_id in enumerate(target["track_ids"]):
                if raw_id is None:
                    continue
                track_id = int(raw_id)
                present.add(track_id)
                if row < len(boxes):
                    area = float(boxes[row, 2]) * float(boxes[row, 3])
                    areas[track_id] = max(areas.get(track_id, 0.0), area)
            return present, areas

        seen_ids: list[set[int]] = [set() for _ in range(batch_size)]
        peak_area: list[dict[int, float]] = [{} for _ in range(batch_size)]
        for item in range(batch_size):
            present, areas = _frame_maps(supervised_targets[0][item])
            seen_ids[item].update(present)
            for track_id, area in areas.items():
                peak_area[item][track_id] = max(peak_area[item].get(track_id, 0.0), area)

        for step in range(1, num_steps):
            diagnostics["supervised_transition_steps"] += 1.0
            step_flags = {"departure": False, "re_entry": False, "partial_visibility": False}
            for item in range(batch_size):
                prev_present, _ = _frame_maps(supervised_targets[step - 1][item])
                cur_present, cur_areas = _frame_maps(supervised_targets[step][item])
                if prev_present - cur_present:
                    step_flags["departure"] = True
                if (cur_present - prev_present) & seen_ids[item]:
                    step_flags["re_entry"] = True
                for track_id in prev_present & cur_present:
                    baseline = peak_area[item].get(track_id, 0.0)
                    if baseline > 0.0 and cur_areas.get(track_id, 0.0) < _PARTIAL_VISIBILITY_AREA_FRACTION * baseline:
                        step_flags["partial_visibility"] = True
                seen_ids[item].update(cur_present)
                for track_id, area in cur_areas.items():
                    peak_area[item][track_id] = max(peak_area[item].get(track_id, 0.0), area)
            for kind, hit in step_flags.items():
                if hit:
                    diagnostics[f"{kind}_steps"] += 1.0
            if step < len(injection_flags) and injection_flags[step]:
                diagnostics["injected_fp_steps"] += 1.0
            if step < len(dropout_flags) and dropout_flags[step]:
                diagnostics["query_dropout_steps"] += 1.0

    def _log_sampling_transition_diagnostics(self, *, batch_size: int) -> None:
        """Log this step's supervised-transition composition (US-011 AC).

        ``train/sampling_transition_<kind>_fraction`` is the share of the step's supervised
        persistent transitions that contained at least one event of that kind (departure,
        re-entry, partial visibility, injected FP, query dropout); the raw
        ``train/sampling_transition_<kind>_steps`` counts and the
        ``train/sampling_transition_supervised_steps`` denominator are logged alongside so the
        proportions are never reported without the counts that produced them.
        """
        diagnostics = self._sampling_transition_diagnostics
        total = diagnostics["supervised_transition_steps"]
        log_kwargs = dict(
            on_step=self.train_config.train_log_on_step,
            on_epoch=True,
            sync_dist=self.train_config.train_log_sync_dist,
            batch_size=batch_size,
        )
        self.log("train/sampling_transition_supervised_steps", float(total), **log_kwargs)
        for kind in _SAMPLING_TRANSITION_KINDS:
            steps = diagnostics[f"{kind}_steps"]
            self.log(f"train/sampling_transition_{kind}_steps", float(steps), **log_kwargs)
            self.log(
                f"train/sampling_transition_{kind}_fraction",
                float(steps / total) if total else 0.0,
                **log_kwargs,
            )

    def _log_tracking_step_perf_metrics(
        self,
        *,
        unroll_seconds: float,
        peak_vram_bytes: int,
        processed_frames: int,
        gradient_bearing_frames: int,
        batch_size: int,
    ) -> None:
        """Log this step's throughput, step time, peak VRAM, and frame budget (US-011 AC).

        ``unroll_seconds`` times the causal clip unroll (every forwarded frame plus its lifecycle
        commit) -- the tracking-specific work of the step, excluding the optimizer/backward that
        Lightning runs after ``training_step`` returns. ``processed_frames`` is every forwarded
        frame including burn-in (``batch_size * clip_length``); ``gradient_bearing_frames`` is the
        loss-producing suffix only (``batch_size * supervised_frames``); throughput is
        ``processed_frames / unroll_seconds``. ``peak_vram_bytes`` is ``0`` off CUDA.
        """
        throughput = processed_frames / unroll_seconds if unroll_seconds > 0 else 0.0
        log_kwargs = dict(
            on_step=self.train_config.train_log_on_step,
            on_epoch=True,
            sync_dist=self.train_config.train_log_sync_dist,
            batch_size=batch_size,
        )
        self.log("train/perf_step_time_seconds", float(unroll_seconds), **log_kwargs)
        self.log("train/perf_processed_frames", float(processed_frames), **log_kwargs)
        self.log("train/perf_gradient_bearing_frames", float(gradient_bearing_frames), **log_kwargs)
        self.log("train/perf_throughput_frames_per_second", float(throughput), **log_kwargs)
        self.log("train/perf_peak_vram_bytes", float(peak_vram_bytes), **log_kwargs)

    def _commit_tracking_state(
        self,
        frame: TrackingFrameOutput,
        assignments: list[SequenceAssignment],
        prior_state: TrackQueryState,
        prior_track_ids: list[tuple[int | None, ...]],
        tables: list[TrackSlotTable] | None,
        *,
        frame_index: int,
        inference_like: bool,
        fp_injection_remaining: list[int] | None = None,
        record_diagnostics: bool = False,
        frame_targets: Sequence[Any] | None = None,
    ) -> tuple[TrackQueryState, list[tuple[int | None, ...]], list[TrackSlotTable] | None]:
        """Commit one frame's recurrent state under the configured clip lifecycle.

        Ground-truth assignment always remains available to :meth:`_compute_train_losses`
        for loss construction, but it is deliberately not consulted here in inference-like
        mode: the committed identities, tensors, and lifecycle transitions come only from
        the model's own predictions, exactly as they would at deployment. The one exception is
        the optional PRD US-019 error-exposure pilots, which use ``assignments`` only to pick
        *which* candidates to perturb before the lifecycle decision runs -- see
        :meth:`_commit_tracking_state_inference_like`.

        Args:
            frame: Slot-aligned candidate output for the current frame.
            assignments: Ground-truth identity assignment for this frame, used by the
                assignment-guided control and by the inference-like error-exposure pilots.
            prior_state: Recurrent state committed after the preceding frame.
            prior_track_ids: Per-item slot identity table entering this frame.
            tables: Per-item lifecycle host state entering this frame. Required and updated
                in inference-like mode; unused and passed through as ``None`` otherwise.
            frame_index: Monotonically increasing clip-local frame index.
            inference_like: Selects the prediction-driven lifecycle over the assignment-guided
                control.
            fp_injection_remaining: Per-item remaining false-positive-injection budget for the
                whole clip, mutated in place. ``None`` when injection is disabled.
            record_diagnostics: When ``True`` (supervised frames only), every prediction-driven
                lifecycle event kind is tallied into ``self._lifecycle_transition_counts`` and
                the error-exposure diagnostics are accumulated, for per-step logging
                (duplicate-FP PRD US-009/US-010). Has no effect in assignment-guided mode,
                which emits no lifecycle events.
            frame_targets: Per-item target dicts for this frame, used only to compute the
                ground-truth-overlap error-exposure diagnostic. ``None`` disables that metric.

        Returns:
            Committed state, the next slot identity table, and the next lifecycle host state
            (``None`` in assignment-guided mode).
        """
        if inference_like:
            if tables is None:
                raise ValueError("inference-like tracking requires per-item lifecycle tables")
            state, next_ids, next_tables = self._commit_tracking_state_inference_like(
                frame,
                assignments,
                prior_state,
                tables,
                frame_index=frame_index,
                fp_injection_remaining=fp_injection_remaining,
                record_diagnostics=record_diagnostics,
                frame_targets=frame_targets,
            )
        else:
            state, next_ids = self._commit_tracking_state_assignment_guided(
                frame, assignments, prior_state, prior_track_ids
            )
            next_tables = None

        if self.train_config.tracking.detach_state_between_frames:
            state = TrackQueryState(
                state.query_features.detach(),
                state.reference_boxes.detach(),
                state.active_mask,
            )
        return state, next_ids, next_tables

    def _unroll_tracking_clip(
        self,
        frame_batches: tuple | list,
        target_batches: tuple | list,
        *,
        inference_like: bool,
        compute_losses: bool = True,
        tracking_model: Any | None = None,
        burn_in_frames: int = 0,
        tbptt_chunk_frames: int | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[tuple[dict[str, Any], tuple]]]:
        """Causally unroll one time-major clip and return frame-mean losses and outputs.

        Implements the PRD Section 7.6 long-horizon curriculum: ``burn_in_frames`` leading
        frames are unrolled under ``torch.no_grad()`` through the prediction-driven
        inference-like lifecycle (regardless of ``inference_like``) so the model enters the
        supervised suffix with a causally realistic state history it never backpropagates
        through. The remaining supervised frames are grouped into truncated-backpropagation
        chunks of ``tbptt_chunk_frames`` frames (``None`` keeps the whole supervised suffix as
        one chunk, matching pre-curriculum behavior); recurrent state is detached after every
        chunk so no gradient crosses a chunk boundary. Losses are averaged only over supervised
        frames, so the loss scale does not grow with the burn-in or supervised suffix length.

        Args:
            frame_batches: Time-major per-frame ``NestedTensor`` samples for the whole clip.
            target_batches: Time-major per-frame target dictionaries for the whole clip.
            inference_like: Selects the prediction-driven lifecycle over the assignment-guided
                control for the *supervised* frames. Burn-in frames always use the
                prediction-driven lifecycle.
            compute_losses: Whether to compute per-frame supervised losses.
            tracking_model: Optional model override (defaults to ``self.model``).
            burn_in_frames: Leading frames of the clip excluded from the loss and run without
                gradient. Must be in ``[0, len(frame_batches))``.
            tbptt_chunk_frames: Number of consecutive supervised frames backpropagated together
                before recurrent state is detached. ``None`` means one chunk for the whole
                supervised suffix.

        Returns:
            Frame-mean supervised losses and the per-frame ``(outputs, targets)`` pairs for
            every frame that received a forward pass (burn-in included).
        """
        if len(frame_batches) != len(target_batches) or not frame_batches:
            raise ValueError("tracking frame and target batches must have the same nonzero clip length")
        if len(frame_batches) != self.train_config.tracking.clip_length:
            raise ValueError(
                f"received clip length {len(frame_batches)}, expected {self.train_config.tracking.clip_length}"
            )
        if not 0 <= burn_in_frames < len(frame_batches):
            raise ValueError(
                f"burn_in_frames ({burn_in_frames}) must be in [0, clip_length) for a clip of length "
                f"{len(frame_batches)}"
            )
        if tbptt_chunk_frames is not None and tbptt_chunk_frames < 1:
            raise ValueError(f"tbptt_chunk_frames ({tbptt_chunk_frames}) must be at least one frame")

        # Reset the per-step lifecycle-commitment diagnostics (duplicate-FP PRD US-009) and the
        # error-exposure diagnostics (US-010): only supervised frames of this clip contribute
        # counts, so burn-in and prior clips leave nothing behind.
        self._lifecycle_commit_frames = {"oracle": 0, "prediction_driven": 0}
        self._lifecycle_transition_counts = Counter()
        self._error_exposure_diagnostics = self._empty_error_exposure_diagnostics()
        self._injected_track_records = {}
        self._sampling_transition_diagnostics = self._empty_sampling_transition_diagnostics()

        batch_size = len(target_batches[0])
        slot_track_ids = [tuple(None for _ in range(self.model_config.num_queries)) for _ in range(batch_size)]
        needs_tables = inference_like or burn_in_frames > 0
        tables: list[TrackSlotTable] | None = (
            [TrackSlotTable.empty(self.model_config.num_queries) for _ in range(batch_size)] if needs_tables else None
        )
        fp_injection_remaining: list[int] | None = (
            [self.train_config.tracking.false_positive_injection_max_per_sample] * batch_size
            if inference_like and self.train_config.tracking.false_positive_injection_enabled
            else None
        )
        first_tensors = frame_batches[0].tensors
        state: TrackQueryState = TrackQueryState.empty(
            batch_size=batch_size,
            num_queries=self.model_config.num_queries,
            hidden_dim=self.model_config.hidden_dim,
            device=first_tensors.device,
            dtype=first_tensors.dtype,
        )
        frame_losses: list[dict[str, torch.Tensor]] = []
        frame_outputs: list[tuple[dict[str, Any], tuple]] = []
        active_model = self.model if tracking_model is None else tracking_model

        def _run_frame(
            frame_index: int, *, frame_inference_like: bool, record_loss: bool, injection_remaining: list[int] | None
        ) -> None:
            nonlocal state, slot_track_ids, tables
            samples, targets = frame_batches[frame_index], target_batches[frame_index]
            if len(targets) != batch_size:
                raise ValueError("every sequence time step must have the same batch size")
            frame = active_model.forward_tracking(samples, state)
            outputs = self._tracking_outputs(frame)
            assignments = identity_aware_sequence_assignment(
                self.criterion.matcher, outputs, list(targets), slot_track_ids
            )
            if record_loss:
                frame_losses.append(self.criterion(outputs, list(targets), assignments))
            state, slot_track_ids, tables = self._commit_tracking_state(
                frame,
                assignments,
                state,
                slot_track_ids,
                tables,
                frame_index=frame_index,
                inference_like=frame_inference_like,
                fp_injection_remaining=injection_remaining,
                record_diagnostics=record_loss,
                frame_targets=list(targets),
            )
            if record_loss:
                stage = "prediction_driven" if frame_inference_like else "oracle"
                self._lifecycle_commit_frames[stage] += batch_size
            frame_outputs.append((outputs, targets))

        # Burn-in: prediction-driven recurrence under no gradient (PRD Section 7.6). Never
        # contributes to the loss, never injects synthetic errors, and always commits through
        # the inference-like lifecycle regardless of the supervised suffix's lifecycle_mode.
        if burn_in_frames > 0:
            with torch.no_grad():
                for frame_index in range(burn_in_frames):
                    _run_frame(frame_index, frame_inference_like=True, record_loss=False, injection_remaining=None)
            state = TrackQueryState(state.query_features.detach(), state.reference_boxes.detach(), state.active_mask)

        # Supervised suffix, causally unrolled in truncated-backpropagation-through-time chunks.
        # Detaching state after every chunk means no gradient flows across a chunk boundary --
        # and, since the loop ends at the clip boundary, no gradient graph survives past the clip.
        supervised_indices = range(burn_in_frames, len(frame_batches))
        chunk_size = tbptt_chunk_frames if tbptt_chunk_frames is not None else max(1, len(supervised_indices))
        supervised_injection_flags: list[bool] = []
        supervised_dropout_flags: list[bool] = []
        for offset, frame_index in enumerate(supervised_indices):
            fp_applied_before = self._error_exposure_diagnostics["fp_injection_applied"]
            dropout_applied_before = self._error_exposure_diagnostics["query_dropout_applied"]
            _run_frame(
                frame_index,
                frame_inference_like=inference_like,
                record_loss=compute_losses,
                injection_remaining=fp_injection_remaining,
            )
            supervised_injection_flags.append(
                self._error_exposure_diagnostics["fp_injection_applied"] > fp_applied_before
            )
            supervised_dropout_flags.append(
                self._error_exposure_diagnostics["query_dropout_applied"] > dropout_applied_before
            )
            if (offset + 1) % chunk_size == 0:
                state = TrackQueryState(
                    state.query_features.detach(), state.reference_boxes.detach(), state.active_mask
                )

        self._finalize_error_exposure_diagnostics()
        self._accumulate_sampling_transition_composition(
            supervised_targets=[target_batches[index] for index in supervised_indices],
            injection_flags=supervised_injection_flags,
            dropout_flags=supervised_dropout_flags,
        )

        loss_names = set().union(*(losses.keys() for losses in frame_losses)) if frame_losses else set()
        mean_losses = {
            name: torch.stack([losses[name] for losses in frame_losses if name in losses]).mean() for name in loss_names
        }
        return mean_losses, frame_outputs

    def _resolve_supervised_lifecycle_inference_like(self) -> bool:
        """Resolve the assignment-guided -> prediction-driven curriculum for this training step.

        Returns whether the *supervised* frames of this step's clip commit recurrent state
        through the exact deployment lifecycle (:func:`transition_lifecycle`) rather than the
        assignment-guided control. ``lifecycle_commitment_curriculum.mode == "disabled"``
        defers entirely to the static ``lifecycle_mode`` so pre-existing runs are unchanged;
        ``"epoch"`` / ``"step"`` switch from the control to prediction-driven commitment once
        the configured warm-up has elapsed, deterministically from ``trainer.current_epoch`` /
        ``trainer.global_step`` (so repeated runs and resumes agree). Burn-in frames always
        run prediction-driven regardless of this result.
        """
        tracking_config = self.train_config.tracking
        curriculum = tracking_config.lifecycle_commitment_curriculum
        if curriculum.mode == "disabled":
            return tracking_config.lifecycle_mode == "inference_like"
        trainer = getattr(self, "_trainer", None)
        if curriculum.mode == "epoch":
            return int(getattr(trainer, "current_epoch", 0) or 0) >= curriculum.warmup_epochs
        return int(getattr(trainer, "global_step", 0) or 0) >= curriculum.warmup_steps

    def _log_lifecycle_commitment_diagnostics(self, *, batch_size: int) -> None:
        """Log this step's oracle/prediction commitment fractions and every lifecycle event kind.

        ``train/lifecycle_prediction_driven_fraction`` is the share of the step's supervised
        frame-item commits that ran through the deployment lifecycle rather than the
        assignment-guided control; ``on_epoch=True`` means the epoch-level value shows exactly
        where the US-009 curriculum crossed over. ``train/lifecycle_event_<kind>`` reports the
        per-step count of every prediction-driven :class:`LifecycleEvent` kind
        (``activated``, ``suspended``, ``recovered``, ``terminated``, ``confirmed``,
        ``tentative_started``, ...), so incorrect births and missed continuations produced by
        the model's own state are visible during training.
        """
        commit_frames = self._lifecycle_commit_frames
        total = commit_frames["oracle"] + commit_frames["prediction_driven"]
        prediction_fraction = commit_frames["prediction_driven"] / total if total else 0.0
        oracle_fraction = commit_frames["oracle"] / total if total else 0.0
        log_kwargs = dict(
            on_step=self.train_config.train_log_on_step,
            on_epoch=True,
            sync_dist=self.train_config.train_log_sync_dist,
            batch_size=batch_size,
        )
        self.log("train/lifecycle_prediction_driven_fraction", prediction_fraction, **log_kwargs)
        self.log("train/lifecycle_oracle_fraction", oracle_fraction, **log_kwargs)
        for kind in _LIFECYCLE_EVENT_KINDS:
            self.log(
                f"train/lifecycle_event_{kind}",
                float(self._lifecycle_transition_counts.get(kind, 0)),
                **log_kwargs,
            )

    def _log_error_exposure_diagnostics(self, *, batch_size: int) -> None:
        """Log this step's false-positive-injection and query-dropout diagnostics (US-010).

        Covers every field the story's acceptance criteria call for: attempted vs. applied
        counts for both mechanisms, the mean genuine foreground score that was overridden, the
        mean ground-truth box overlap of the perturbed slots (``0`` when a slot overlaps no
        real object), the mean active-capacity pressure at injection time, the mean lifetime of
        injected tracks (frames survived within the clip), and how many injected tracks the
        lifecycle later cancelled. ``train/error_exposure_curriculum_factor`` records the
        warm-up-then-ramp multiplier currently applied to both sampling rates.
        """
        diagnostics = self._error_exposure_diagnostics
        fp_applied = diagnostics["fp_injection_applied"]
        fp_tracked = diagnostics["fp_injection_tracked"]
        dropout_applied = diagnostics["query_dropout_applied"]

        def _mean(total: float, count: float) -> float:
            return total / count if count else 0.0

        log_kwargs = dict(
            on_step=self.train_config.train_log_on_step,
            on_epoch=True,
            sync_dist=self.train_config.train_log_sync_dist,
            batch_size=batch_size,
        )
        values = {
            "train/error_exposure_curriculum_factor": self._error_exposure_curriculum_factor(),
            "train/error_exposure_fp_injection_attempted": diagnostics["fp_injection_attempted"],
            "train/error_exposure_fp_injection_applied": fp_applied,
            "train/error_exposure_fp_injection_cancelled": diagnostics["fp_injection_cancelled"],
            "train/error_exposure_fp_injection_mean_score": _mean(diagnostics["fp_injection_score_sum"], fp_applied),
            "train/error_exposure_fp_injection_mean_target_iou": _mean(
                diagnostics["fp_injection_target_iou_sum"], fp_applied
            ),
            "train/error_exposure_fp_injection_mean_capacity_pressure": _mean(
                diagnostics["fp_injection_capacity_pressure_sum"], fp_applied
            ),
            "train/error_exposure_fp_injection_mean_lifetime": _mean(
                diagnostics["fp_injection_lifetime_sum"], fp_tracked
            ),
            "train/error_exposure_query_dropout_attempted": diagnostics["query_dropout_attempted"],
            "train/error_exposure_query_dropout_applied": dropout_applied,
            "train/error_exposure_query_dropout_mean_score": _mean(
                diagnostics["query_dropout_score_sum"], dropout_applied
            ),
            "train/error_exposure_query_dropout_mean_target_iou": _mean(
                diagnostics["query_dropout_target_iou_sum"], dropout_applied
            ),
            "train/error_exposure_query_dropout_mean_lifetime": _mean(
                diagnostics["query_dropout_lifetime_sum"], dropout_applied
            ),
        }
        for name, value in values.items():
            self.log(name, float(value), **log_kwargs)

    def _training_step_tracking(
        self, frame_batches: tuple | list, target_batches: tuple | list, batch_idx: int
    ) -> torch.Tensor:
        """Run one causal video-training step with clip-local recurrent state."""
        tracking_config = self.train_config.tracking
        cuda_device = self._tracking_clip_cuda_device(frame_batches)
        if cuda_device is not None:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        unroll_start = time.perf_counter()
        loss_dict, _ = self._unroll_tracking_clip(
            frame_batches,
            target_batches,
            inference_like=self._resolve_supervised_lifecycle_inference_like(),
            burn_in_frames=tracking_config.burn_in_frames,
            tbptt_chunk_frames=tracking_config.tbptt_chunk_frames,
        )
        unroll_seconds = time.perf_counter() - unroll_start
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(cuda_device)) if cuda_device is not None else 0
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[name] * weight_dict[name] for name in loss_dict if name in weight_dict)
        batch_size = len(target_batches[0])
        self.log_dict(
            {f"train/{name}": value for name, value in loss_dict.items()},
            on_step=self.train_config.train_log_on_step,
            on_epoch=True,
            sync_dist=self.train_config.train_log_sync_dist,
            batch_size=batch_size,
        )
        self.log(
            "train/loss",
            loss,
            on_step=self.train_config.train_log_on_step,
            on_epoch=True,
            sync_dist=self.train_config.train_log_sync_dist,
            batch_size=batch_size,
        )
        self._log_train_progress_metrics(loss, loss_dict, batch_size=batch_size)
        self._log_lifecycle_commitment_diagnostics(batch_size=batch_size)
        self._log_error_exposure_diagnostics(batch_size=batch_size)
        self._log_sampling_transition_diagnostics(batch_size=batch_size)
        self._log_tracking_step_perf_metrics(
            unroll_seconds=unroll_seconds,
            peak_vram_bytes=peak_vram_bytes,
            processed_frames=batch_size * tracking_config.clip_length,
            gradient_bearing_frames=batch_size * tracking_config.supervised_frames,
            batch_size=batch_size,
        )
        return loss / max(1, int(self.trainer.accumulate_grad_batches))

    @staticmethod
    def _tracking_clip_cuda_device(frame_batches: tuple | list) -> torch.device | None:
        """Return the CUDA device a tracking clip lives on, or ``None`` when it is not on CUDA.

        Peak-VRAM accounting (duplicate-FP PRD US-011 AC) is only meaningful on CUDA; on CPU (the
        unit-test path) this returns ``None`` and the caller reports ``0`` bytes rather than
        calling ``torch.cuda`` APIs that would raise.
        """
        if not torch.cuda.is_available() or not frame_batches:
            return None
        tensors = getattr(frame_batches[0], "tensors", None)
        device = getattr(tensors, "device", None)
        return device if device is not None and device.type == "cuda" else None

    def _compute_train_losses(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Compute normalized losses for logging and raw weighted loss for backward.

        Args:
            outputs: Model output dictionary.
            targets: Target dictionaries for the current batch.

        Returns:
            A tuple of normalized loss dictionary, unnormalized weighted loss numerator, and box normalizer.
        """
        weight_dict = self.criterion.weight_dict
        if not getattr(self.criterion, "supports_loss_normalizer_override", False):
            raise ValueError(
                f"{type(self.criterion).__name__}.supports_loss_normalizer_override is False; "
                "manual optimization (keypoint models) requires a criterion that accepts a "
                "num_boxes keyword argument. Set supports_loss_normalizer_override = True on "
                "your criterion subclass and implement the num_boxes parameter in forward()."
            )
        normalizer = self.criterion.num_boxes_for_targets(outputs, targets)
        numerator_loss_dict = self.criterion(outputs, targets, num_boxes=torch.ones_like(normalizer))
        # Keys in weight_dict are loss terms whose criterion implementation divides by num_boxes
        # (so passing num_boxes=1.0 yields raw numerators that we divide by normalizer here).
        # Keys outside weight_dict (e.g. "class_error", "cardinality_error") are diagnostics
        # that do NOT divide by num_boxes internally — they are passed through unchanged.
        # If a future loss term divides by num_boxes AND is omitted from weight_dict, its
        # logged value will be on a different scale than the keypoint path; verify when adding
        # new criterion terms.
        loss_dict = {
            key: value / normalizer if key in weight_dict else value for key, value in numerator_loss_dict.items()
        }
        raw_loss = sum(numerator_loss_dict[k] * weight_dict[k] for k in numerator_loss_dict if k in weight_dict)
        return loss_dict, raw_loss, normalizer

    def _scale_loss_for_accumulation(
        self,
        raw_loss: torch.Tensor,
        normalizer: torch.Tensor,
    ) -> torch.Tensor:
        """Scale the current numerator loss by the accumulated box denominator.

        Args:
            raw_loss: Current microbatch weighted loss numerator.
            normalizer: Current microbatch box denominator.

        Returns:
            Loss scalar to pass to ``manual_backward``.
        """
        normalizer = normalizer.detach()
        previous_normalizer = self._accumulated_box_normalizer
        accumulated_normalizer = normalizer if previous_normalizer is None else previous_normalizer + normalizer
        if previous_normalizer is not None:
            self._rescale_accumulated_gradients(previous_normalizer / accumulated_normalizer)
        self._accumulated_box_normalizer = accumulated_normalizer.detach()
        return raw_loss / accumulated_normalizer

    def _rescale_accumulated_gradients(self, scale: torch.Tensor) -> None:
        """Rescale gradients already accumulated in the current optimizer window.

        Args:
            scale: Multiplicative factor that converts previous gradients from the old denominator to the new one.
        """
        for parameter in self.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale.to(device=parameter.grad.device, dtype=parameter.grad.dtype))

    def _should_step_optimizer(self, batch_idx: int) -> bool:
        """Return whether the current batch closes an optimizer accumulation window.

        The optimizer steps when either:

        - The current batch closes a complete ``grad_accum_steps`` window
          (``(batch_idx + 1) % grad_accum_steps == 0``), or
        - This is the final batch of the epoch and a partial accumulation window
          is still open, so the trailing microbatches are not silently dropped.

        Lightning's ``Trainer.num_training_batches`` may be reported as ``float('inf')``
        for iterable / streaming datasets where the epoch length is unknown. In that case
        only the modulo path can ever close the window — the final-batch fallback is
        skipped because ``batch_idx + 1`` can never reach infinity.

        Args:
            batch_idx: Batch index within the epoch.

        Returns:
            ``True`` when the optimizer should step after this batch.
        """
        accum_steps = max(1, int(self.train_config.grad_accum_steps))
        if (batch_idx + 1) % accum_steps == 0:
            return True
        num_training_batches = getattr(self.trainer, "num_training_batches", None)
        return (
            isinstance(num_training_batches, (int, float))
            and math.isfinite(num_training_batches)
            and batch_idx + 1 >= num_training_batches
        )

    def _step_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Clip gradients, step optimizer and scheduler, then reset accumulation state.

        Args:
            optimizer: Optimizer returned by Lightning.
        """
        trainer_gradient_clip_val = getattr(self.trainer, "gradient_clip_val", None)
        if trainer_gradient_clip_val is None:
            gradient_clip_val = self.train_config.clip_max_norm
        elif isinstance(trainer_gradient_clip_val, (int, float)):
            gradient_clip_val = trainer_gradient_clip_val
        else:
            gradient_clip_val = None
        gradient_clip_algorithm = getattr(self.trainer, "gradient_clip_algorithm", None)
        if not isinstance(gradient_clip_algorithm, str):
            gradient_clip_algorithm = None
        if gradient_clip_val is not None and gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )
        optimizer.step()
        optimizer.zero_grad()
        self._step_lr_scheduler()
        self._accumulated_box_normalizer = None

    def _decoded_frame_count(self, samples: Any, targets: Any) -> int:
        """Return the number of model frame-forwards this microbatch performs.

        A multi-frame tracking clip (``clip_length > 1``) forwards every frame in the
        clip -- including burn-in frames, which still run the model even though they
        carry no gradient -- so its contribution is ``batch_size * clip_length`` (PRD
        Section 7.6 "decoded-video-frame budget"). Ordinary image batches and
        degenerate length-1 clips report a flat 1 per microbatch instead of their
        (possibly auto-probed) sample count, so non-curriculum training's LR schedule
        stays byte-for-byte identical to the pre-curriculum step-counted schedule; see
        ``configure_optimizers``'s matching ``_decoded_frames_per_optimizer_step``
        reference.

        Args:
            samples: Either a NestedTensor image batch (stateless) or a sequence of
                per-frame sample batches (a tracking clip).
            targets: Either a list of per-image target dicts (stateless) or a sequence
                of per-frame target-dict lists (a tracking clip), matching ``samples``.
        """
        if (
            self.model_config.tracking.enabled
            and isinstance(samples, (tuple, list))
            and self.train_config.tracking.clip_length > 1
        ):
            return len(targets[0]) * len(samples)
        return 1

    def _advance_decoded_frame_progress(self) -> float:
        """Commit pending decoded frames and return the step-equivalent schedule position.

        PRD Section 7.6: "Learning-rate schedules advance in decoded-frame units, not
        dataloader-step units." ``configure_optimizers()`` still computes ``lr_lambda``
        in the step-equivalent domain (warmup_steps/total_steps derived from
        ``estimated_stepping_batches``), so the frames accumulated since the last call
        are converted back into that domain via ``_decoded_frames_per_optimizer_step``.
        For a batch composition with a constant frame count per optimizer step (every
        existing single-clip-length training run), this produces the exact same
        +1-per-call progression as before; a curriculum or dataloader that mixes clip
        lengths (or stateless images) within one run instead advances the schedule by
        each step's true share of decoded compute.
        """
        self._decoded_frames_seen += self._pending_decoded_frames
        self._pending_decoded_frames = 0
        return self._decoded_frames_seen / self._decoded_frames_per_optimizer_step

    def _step_lr_scheduler(self) -> None:
        """Step Lightning's scheduler object using decoded-frame progress (manual-optimization path)."""
        try:
            scheduler = self.lr_schedulers()
        except (AttributeError, RuntimeError):
            return
        if scheduler is None:
            return
        equivalent_step = self._advance_decoded_frame_progress()
        schedulers = scheduler if isinstance(scheduler, list) else [scheduler]
        for scheduler_item in schedulers:
            scheduler_item.step(equivalent_step)

    def lr_scheduler_step(self, scheduler: Any, metric: Any = None) -> None:
        """Step using decoded-frame progress under automatic optimization (PRD Section 7.6).

        Lightning calls this hook once per optimizer step for ``interval: "step"``
        schedulers under automatic optimization (the detection/segmentation/tracking
        path; the manual-optimization keypoint path steps explicitly via
        ``_step_lr_scheduler`` instead and never reaches this hook). Delegating to the
        same decoded-frame-aware progress keeps both optimization paths on identical
        schedule semantics.
        """
        del metric
        scheduler.step(self._advance_decoded_frame_progress())

    @staticmethod
    def _detach_results(results: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Detach postprocessed result tensors before handing them to callbacks.

        Args:
            results: Per-image postprocessed prediction dictionaries.

        Returns:
            Per-image dictionaries with tensor values detached from the graph.
        """
        return [
            {key: value.detach() if torch.is_tensor(value) else value for key, value in result.items()}
            for result in results
        ]

    def _log_train_progress_metrics(
        self,
        loss: torch.Tensor,
        loss_dict: dict[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> None:
        """Log compact per-step convergence metrics for the progress bar only.

        Args:
            loss: Unscaled aggregate training loss.
            loss_dict: Raw criterion loss dictionary.
            batch_size: Current batch size used by Lightning for metric reduction metadata.
        """
        self.log(
            "loss",
            loss,
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        for loss_name, progress_name in _TRAIN_PROGRESS_LOSS_ALIASES.items():
            value = loss_dict.get(loss_name)
            if value is None:
                continue
            self.log(
                progress_name,
                value,
                prog_bar=True,
                logger=False,
                on_step=True,
                on_epoch=False,
                batch_size=batch_size,
            )

    def _log_val_loss_metrics(
        self,
        loss: torch.Tensor,
        loss_dict: dict[str, torch.Tensor],
        *,
        batch_size: int,
    ) -> None:
        """Log aggregate and component validation losses.

        Args:
            loss: Aggregate weighted validation loss.
            loss_dict: Raw criterion loss dictionary.
            batch_size: Current batch size used by Lightning for metric reduction metadata.
        """
        self.log_dict(
            {f"val/{k}": v for k, v in loss_dict.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

    def validation_step(self, batch: tuple, batch_idx: int) -> dict[str, Any]:
        """Run forward pass and postprocess for one validation step.

        Returns raw results and targets so ``COCOEvalCallback`` can accumulate them across the epoch via
        ``on_validation_batch_end``.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the validation epoch.

        Returns:
            Dict with ``results`` (postprocessed predictions) and ``targets``.
        """
        samples, targets = batch
        if self.model_config.tracking.enabled and isinstance(samples, (tuple, list)):
            loss_dict, frame_outputs = self._unroll_tracking_clip(
                samples,
                targets,
                inference_like=True,
                compute_losses=self.train_config.compute_val_loss,
            )
            if self.train_config.compute_val_loss:
                weight_dict = self.criterion.weight_dict
                loss = sum(loss_dict[name] * weight_dict[name] for name in loss_dict if name in weight_dict)
                self._log_val_loss_metrics(loss, loss_dict, batch_size=len(targets[0]))
            results: list[dict[str, torch.Tensor]] = []
            flat_targets: list[dict[str, torch.Tensor]] = []
            for outputs, frame_targets in frame_outputs:
                orig_sizes = torch.stack([target["orig_size"] for target in frame_targets])
                results.extend(self.postprocess(outputs, orig_sizes))
                flat_targets.extend(frame_targets)
            return {"results": results, "targets": flat_targets}
        outputs = self.model(samples)
        if self.train_config.compute_val_loss:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            self._log_val_loss_metrics(loss, loss_dict, batch_size=len(targets))

        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        results = self.postprocess(outputs, orig_sizes)
        return {"results": results, "targets": targets}

    @property
    def _use_fused_optimizer(self) -> bool:
        """Return whether fused AdamW should be used for the current training configuration.

        Fused AdamW is only safe when the trainer's actual precision is a BF16 variant.  Checking GPU capability alone
        (``is_bf16_supported()``) is
        insufficient: on Ampere+ hardware that flag is always ``True`` even when
        the trainer is configured for ``32-true``, which causes a ``params, grads, exp_avgs, and exp_avg_sqs must have
        same dtype, device, and layout`` crash in DDP because gradient bucket views have non-matching strides in FP32.

        Returns:
            ``True`` when fused AdamW is both requested and safe to use.

        Examples:
            >>> from unittest.mock import patch
            >>> module = RFDETRModelModule.__new__(RFDETRModelModule)
            >>> module.model_config = type("Cfg", (), {"fused_optimizer": True})()
            >>> with patch("torch.cuda.is_available", return_value=False):
            ...     module._use_fused_optimizer
            False
        """
        return (
            self.model_config.fused_optimizer
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and str(self.trainer.precision) in {"bf16-mixed", "bf16", "bf16-true"}
        )

    def configure_optimizers(self) -> dict[str, Any]:
        """Build AdamW optimizer with layer-wise LR decay and LambdaLR scheduler.

        Uses ``trainer.estimated_stepping_batches`` for total step count so cosine annealing covers the full training
        run regardless of dataset size or accumulation settings.

        Returns:
            PTL optimizer config dict with optimizer and step-interval scheduler.
        """
        tc = self.train_config
        ns = _namespace_from_configs(self.model_config, tc)

        # Unwrap torch.compile's OptimizedModule so get_param_dict sees the
        # original module's named_parameters() — compiled wrapper can cause
        # name-prefix mismatches that put the same tensor in multiple groups.
        model_for_params = getattr(self.model, "_orig_mod", self.model)
        param_dicts = get_param_dict(ns, model_for_params)
        param_dicts = [p for p in param_dicts if p["params"].requires_grad]
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=tc.lr,
            weight_decay=tc.weight_decay,
            fused=self._use_fused_optimizer,
        )

        # ``trainer.estimated_stepping_batches`` is reported in *microbatch* units when
        # the keypoint path runs with ``Trainer(accumulate_grad_batches=1)`` and manages
        # accumulation manually. ``LambdaLR.step()`` is called once per optimizer-step
        # (i.e. every ``grad_accum_steps`` microbatches), so the schedule must be sized
        # in optimizer-step units rather than microbatches; otherwise warmup and cosine
        # decay finish ``grad_accum_steps``× too early. Detection / segmentation models
        # still rely on Lightning's automatic optimization, where PTL already accounts
        # for ``accumulate_grad_batches`` inside ``estimated_stepping_batches`` and the
        # division below is a no-op (``grad_accum_steps`` would be 1 in that path).
        grad_accum_steps = max(1, int(tc.grad_accum_steps))
        microbatches = int(self.trainer.estimated_stepping_batches)
        # _should_step_optimizer steps the final partial window at epoch end, so the true
        # number of optimizer steps is ceil(microbatches / grad_accum_steps).  Using floor
        # would undercount when the epoch is not evenly divisible, causing warmup / cosine
        # schedules to finish one step earlier than the last actual step fires.
        total_steps = (
            max(1, math.ceil(microbatches / grad_accum_steps)) if self._use_manual_optimization else microbatches
        )
        steps_per_epoch = max(1, total_steps // tc.epochs)
        warmup_steps = int(steps_per_epoch * tc.warmup_epochs)

        # Decoded-frame LR progress (PRD Section 7.6): lr_lambda below still operates in
        # the step-equivalent domain computed above, so _step_lr_scheduler /
        # lr_scheduler_step convert accumulated decoded frames back into that domain via
        # this reference. Only genuine multi-frame clips (clip_length > 1) carry a
        # per-frame reference derived from the configured batch size (tracking rejects
        # batch_size="auto", so this is always a concrete int there); every other run
        # (including auto-batch-sized image training) uses grad_accum_steps as the
        # reference, which -- paired with _decoded_frame_count returning a flat 1 per
        # non-curriculum microbatch -- reproduces the exact +1-per-call progression the
        # unscaled step counter always used.
        effective_clip_length = tc.tracking.clip_length if self.model_config.tracking.enabled else 1
        self._decoded_frames_per_optimizer_step = float(
            int(tc.batch_size) * effective_clip_length * grad_accum_steps
            if effective_clip_length > 1
            else grad_accum_steps
        )
        self._decoded_frames_seen = 0.0
        self._pending_decoded_frames = 0

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if tc.lr_scheduler == "cosine":
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return tc.lr_min_factor + (1 - tc.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
            # Step decay: drop by 10× after lr_drop epochs.
            if current_step < tc.lr_drop * steps_per_epoch:
                return 1.0
            return 0.1

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Override PTL gradient clipping to support fused AdamW.

        PTL's AMP precision plugin refuses to clip gradients when the optimizer declares it handles unscaling internally
        (fused=True).  When fused is active we are on BF16 (no GradScaler) so ``clip_grad_norm_`` is correct.  For the
        non-fused path (FP16 + GradScaler or FP32) we delegate to ``super()`` to preserve scaler-aware unscaling.

        Args:
            optimizer: The current optimizer.
            gradient_clip_val: Maximum gradient norm.
            gradient_clip_algorithm: Clipping algorithm; forwarded to super()
                for the non-fused path.
        """
        if self._use_fused_optimizer:
            if gradient_clip_val and gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
        else:
            super().clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )

    def test_step(self, batch: tuple, batch_idx: int) -> dict[str, Any]:
        """Run forward pass and postprocess for one test step.

        Mirrors :meth:`validation_step` so ``COCOEvalCallback`` can accumulate results via ``on_test_batch_end`` when
        ``trainer.test()`` is called (e.g. from :class:`~rfdetr.training.callbacks.BestModelCallback` at end of
        training).

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the test epoch.

        Returns:
            Dict with ``results`` (postprocessed predictions) and ``targets``.
        """
        samples, targets = batch
        outputs = self.model(samples)
        if self.train_config.compute_test_loss:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            self.log("test/loss", loss, sync_dist=True, batch_size=len(targets))

        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        results = self.postprocess(outputs, orig_sizes)
        return {"results": results, "targets": targets}

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Run inference on a preprocessed batch and return postprocessed results.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index.
            dataloader_idx: Index of the predict dataloader.

        Returns:
            Postprocessed detection results from ``PostProcess``.
        """
        samples, targets = batch
        with torch.no_grad():
            outputs = self.model(samples)
        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        return self.postprocess(outputs, orig_sizes)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Auto-detect legacy formats and reconcile PE shapes at checkpoint load time.

        PTL calls this hook before applying ``checkpoint["state_dict"]`` to the module.  Three normalisation steps are
        applied in order:

        1. **Raw legacy format** — a ``*.pth`` file loaded directly by
           ``Trainer`` (e.g. via ``ckpt_path=``).  Recognised by the presence of ``"model"`` without ``"state_dict"``.
           The state dict is rewritten in-place with the ``"model."`` prefix so PTL can apply it normally.

        2. **Positional-embedding interpolation** — when the checkpoint was
           saved at a different image resolution than the current model, the DINOv2 ``position_embeddings`` tensor shape
           will mismatch. :func:`~rfdetr.models.weights.interpolate_position_embeddings` is called to bicubic-resize the
           PE to ``model_config.positional_encoding_size`` before PTL applies the state dict.  Regression fix for
           :issue:`998`.

        3. **Converted format** — a file produced by
           :func:`~rfdetr.training.checkpoint.convert_legacy_checkpoint` that already has ``"state_dict"`` but also
           carries ``"legacy_ema_state_dict"``.  The EMA weights are stashed on ``self._pending_legacy_ema_state`` for
           optional restoration by :class:`~rfdetr.training.callbacks.ema.RFDETREMACallback`.

        Note:
            This hook only fires on ``Trainer(ckpt_path=...)`` resume paths. Fresh-train bootstrap from a
            ``pretrain_weights`` checkpoint runs through :func:`~rfdetr.models.weights.load_pretrain_weights` during
            ``__init__`` instead — that helper performs its own PTL ``.ckpt`` normalisation (``state_dict`` → ``model``
            key, ``_orig_mod`` strip) and PE interpolation, so the two code paths intentionally do not share state.

        Args:
            checkpoint: Checkpoint dict passed in by PTL (mutated in-place).
        """
        # Raw legacy .pth: no "state_dict" key — build it from "model".
        if "model" in checkpoint and "state_dict" not in checkpoint:
            checkpoint["state_dict"] = {"model." + k: v for k, v in checkpoint["model"].items()}

        # Interpolate DINOv2 positional embeddings when the checkpoint was saved
        # at a different resolution than the current model.  PTL applies
        # checkpoint["state_dict"] immediately after this hook, so the shapes
        # must already match at this point.  Regression: #998.
        if "state_dict" in checkpoint:
            interpolate_position_embeddings(
                checkpoint["state_dict"],
                self.model_config.positional_encoding_size,
            )

        # Stash legacy EMA weights for RFDETREMACallback.setup(), which restores
        # them into AveragedModel when resuming from converted legacy checkpoints.
        if "legacy_ema_state_dict" in checkpoint:
            self._pending_legacy_ema_state = checkpoint["legacy_ema_state_dict"]
            warnings.warn(
                "Checkpoint contains legacy EMA weights (`legacy_ema_state_dict`). "
                "Add RFDETREMACallback to your trainer callbacks to restore them; "
                "without it the stashed weights will be ignored.",
                UserWarning,
                stacklevel=2,
            )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Attach the authoritative architecture contract to native Lightning checkpoints."""
        raw_model = getattr(self.model, "_orig_mod", self.model)
        model_state = raw_model.state_dict()
        checkpoint.update(
            authoritative_checkpoint_metadata(
                model_config=self.model_config,
                train_config=self.train_config,
                state_dict=model_state,
                epoch=int(checkpoint.get("epoch", self.current_epoch)),
                weight_flavor="regular",
                source_checkpoint_hash_value=self._source_checkpoint_hash,
            )
        )
        checkpoint["hyper_parameters"] = checkpoint["train_config"]

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Reinitialize the detection head for a new class count.

        Args:
            num_classes: New number of classes (excluding background).
        """
        self.model.reinitialize_detection_head(num_classes)
