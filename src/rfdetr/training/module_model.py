# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""LightningModule for RF-DETR training and validation."""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

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
from rfdetr.tracking.lifecycle import TrackSlotTable, transition_lifecycle
from rfdetr.training.checkpoint import authoritative_checkpoint_metadata, source_checkpoint_hash
from rfdetr.training.param_groups import get_param_dict
from rfdetr.utilities.logger import get_logger

logger = get_logger()

_FALSE_POSITIVE_INJECTION_SALT = 1
_QUERY_DROPOUT_SALT = 2
_ERROR_EXPOSURE_LOGIT_MAGNITUDE = 8.0

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
            active_masks.append(
                torch.tensor([value is not None for value in committed_ids], device=features.device)
            )

        return TrackQueryState(features, boxes, torch.stack(active_masks).bool()), next_ids

    def _error_exposure_draws(
        self, *, salt: int, frame_index: int, batch_index: int, count: int
    ) -> list[float]:
        """Deterministic uniform draws for one frame/batch-item/mechanism triple (PRD US-019).

        Seeded only from ``TrackingTrainConfig.error_exposure_seed`` plus the salt, frame, and
        batch indices -- never from global RNG state -- so injection/dropout decisions are
        reproducible across repeated runs regardless of dataloader shuffling or other
        randomness consumed earlier in the step.
        """
        if count <= 0:
            return []
        base_seed = self.train_config.tracking.error_exposure_seed
        combined_seed = ((base_seed * 1_000_003 + salt) * 1_000_003 + frame_index) * 1_000_003 + batch_index
        generator = torch.Generator(device="cpu")
        generator.manual_seed(combined_seed % (2**63 - 1))
        return torch.rand(count, generator=generator).tolist()

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
        """Select ground-truth-unmatched inactive slots to force-activate this frame.

        Candidates are query positions with no correspondence at all in ``assignment`` (neither
        continuing nor discovery) -- genuine "unmatched query states" -- restricted to currently
        inactive slots so injection creates a new synthetic false-positive track rather than
        altering an already-tracked identity. Selection is capped at ``max_count`` (the
        remaining per-sample injection budget) by keeping the lowest-drawn eligible candidates,
        which is deterministic given the seeded draws.
        """
        if max_count <= 0:
            return []
        matched = set(assignment.decoder_indices[0].tolist())
        candidates = [index for index, slot in enumerate(table.slots) if slot.status == "inactive" and index not in matched]
        if not candidates:
            return []
        draws = self._error_exposure_draws(
            salt=_FALSE_POSITIVE_INJECTION_SALT, frame_index=frame_index, batch_index=batch_index, count=len(candidates)
        )
        ranked = sorted(zip(candidates, draws), key=lambda pair: pair[1])
        return [index for index, draw in ranked if draw < probability][:max_count]

    def _select_query_dropout_indices(
        self,
        table: TrackSlotTable,
        *,
        frame_index: int,
        batch_index: int,
        probability: float,
    ) -> list[int]:
        """Select active slots to force a missed-detection score this frame.

        Never selects every currently active slot: if every draw qualifies, the
        least-confidently-dropped index (highest draw) is spared so a sample can never lose
        all active identities to dropout in one frame.
        """
        candidates = [index for index, slot in enumerate(table.slots) if slot.status == "active"]
        if not candidates:
            return []
        draws = self._error_exposure_draws(
            salt=_QUERY_DROPOUT_SALT, frame_index=frame_index, batch_index=batch_index, count=len(candidates)
        )
        paired = list(zip(candidates, draws))
        selected = [index for index, draw in paired if draw < probability]
        if len(selected) == len(candidates):
            survivor_index = max(paired, key=lambda pair: pair[1])[0]
            selected = [index for index in selected if index != survivor_index]
        return selected

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

    def _commit_tracking_state_inference_like(
        self,
        frame: TrackingFrameOutput,
        assignments: list[SequenceAssignment],
        prior_state: TrackQueryState,
        tables: list[TrackSlotTable],
        *,
        frame_index: int,
        fp_injection_remaining: list[int] | None = None,
    ) -> tuple[TrackQueryState, list[tuple[int | None, ...]], list[TrackSlotTable]]:
        """Commit state through the exact deployment lifecycle state machine.

        Ground-truth assignment plays no role in the committed lifecycle decisions themselves:
        every activation, suspension, recovery, and termination follows
        :func:`transition_lifecycle`, the same pure function ``TrackingSession`` calls at
        inference, driven only by the model's own foreground scores. This is what lets training
        see -- and learn to recover from -- the model's own false positives, false negatives,
        and stale suspended references instead of having ground truth silently repair them.

        ``assignments`` is consulted only by the optional PRD US-019 error-exposure pilots
        (``false_positive_injection_enabled`` / ``query_dropout_enabled``), which perturb the
        *candidate scores* fed into ``transition_lifecycle`` -- never the lifecycle decision
        itself -- to select which ground-truth-unmatched candidates get force-activated or which
        active slots get force-dropped this frame, deterministically.
        """
        if self.model_config.class_schema is None:
            raise ValueError("prediction-driven tracking requires an authoritative class_schema")
        class_schema = self.model_config.class_schema
        tracking_config = self.train_config.tracking
        lifecycle_config = tracking_config.lifecycle
        capacity = self.model_config.tracking.active_capacity(self.model_config.num_queries)

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
            item_frame = self._slice_frame_output(frame, batch_index)

            if tracking_config.false_positive_injection_enabled and fp_injection_remaining is not None:
                injected = self._select_false_positive_injection_indices(
                    table,
                    assignments[batch_index],
                    frame_index=frame_index,
                    batch_index=batch_index,
                    probability=tracking_config.false_positive_injection_probability,
                    max_count=fp_injection_remaining[batch_index],
                )
                if injected:
                    item_frame = self._inject_false_positive_scores(item_frame, injected, class_schema)
                    fp_injection_remaining[batch_index] -= len(injected)

            if tracking_config.query_dropout_enabled:
                dropped = self._select_query_dropout_indices(
                    table,
                    frame_index=frame_index,
                    batch_index=batch_index,
                    probability=tracking_config.query_dropout_probability,
                )
                if dropped:
                    item_frame = self._apply_query_dropout_scores(item_frame, dropped, class_schema)

            transition = transition_lifecycle(
                table,
                item_state,
                item_frame,
                lifecycle_config,
                class_schema,
                max_active_tracks=capacity,
                frame_index=frame_index,
            )
            next_tables.append(transition.table)
            next_ids.append(transition.table.slot_track_ids)
            features.append(transition.state.query_features)
            boxes.append(transition.state.reference_boxes)
            active_masks.append(transition.state.active_mask)

        state = TrackQueryState(torch.cat(features, dim=0), torch.cat(boxes, dim=0), torch.cat(active_masks, dim=0))
        return state, next_ids, next_tables

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

        Returns:
            Committed state, the next slot identity table, and the next lifecycle host state
            (``None`` in assignment-guided mode).
        """
        if inference_like:
            if tables is None:
                raise ValueError("inference-like tracking requires per-item lifecycle tables")
            state, next_ids, next_tables = self._commit_tracking_state_inference_like(
                frame, assignments, prior_state, tables, frame_index=frame_index, fp_injection_remaining=fp_injection_remaining
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
            )
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
        for offset, frame_index in enumerate(supervised_indices):
            _run_frame(
                frame_index,
                frame_inference_like=inference_like,
                record_loss=compute_losses,
                injection_remaining=fp_injection_remaining,
            )
            if (offset + 1) % chunk_size == 0:
                state = TrackQueryState(state.query_features.detach(), state.reference_boxes.detach(), state.active_mask)

        loss_names = set().union(*(losses.keys() for losses in frame_losses)) if frame_losses else set()
        mean_losses = {
            name: torch.stack([losses[name] for losses in frame_losses if name in losses]).mean() for name in loss_names
        }
        return mean_losses, frame_outputs

    def _training_step_tracking(
        self, frame_batches: tuple | list, target_batches: tuple | list, batch_idx: int
    ) -> torch.Tensor:
        """Run one causal video-training step with clip-local recurrent state."""
        tracking_config = self.train_config.tracking
        loss_dict, _ = self._unroll_tracking_clip(
            frame_batches,
            target_batches,
            inference_like=tracking_config.lifecycle_mode == "inference_like",
            burn_in_frames=tracking_config.burn_in_frames,
            tbptt_chunk_frames=tracking_config.tbptt_chunk_frames,
        )
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
        return loss / max(1, int(self.trainer.accumulate_grad_batches))

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
