# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Scheduled complete-sequence tracking evaluation callback for RF-DETR Lightning training."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from rfdetr.training.callbacks.best_model import BestModelCallback
from rfdetr.training.checkpoint import authoritative_checkpoint_metadata, serialize_train_config
from rfdetr.utilities.logger import get_logger

logger = get_logger()


@dataclass(frozen=True)
class TrackingEvalRequest:
    """One materialized checkpoint ready for external chronological tracking evaluation.

    Attributes:
        checkpoint_path: Self-describing ``.pth`` checkpoint written for this request. Loadable by the same generic
            checkpoint loader used everywhere else (PRD US-004).
        weight_flavor: ``"regular"`` or ``"ema"``.
        epoch: 1-indexed count of completed training epochs (matches ``checkpoint_epoch`` convention used elsewhere:
            the 0-indexed ``trainer.current_epoch`` plus one).
        is_final_epoch: Whether this request corresponds to the last training epoch.
    """

    checkpoint_path: Path
    weight_flavor: Literal["regular", "ema"]
    epoch: int
    is_final_epoch: bool


class TrackingEvalCallback(Callback):
    """Runs an externally supplied evaluation function on a fixed epoch schedule.

    Mirrors :class:`~rfdetr.training.callbacks.coco_eval.COCOEvalCallback`'s "every N epochs, always the final epoch"
    schedule (PRD Section 7.5), but for complete-sequence chronological tracking evaluation rather than stateless
    per-image mAP. This callback owns only the *schedule* and *checkpoint materialization*; the actual evaluation
    (which requires a dataset-specific ground-truth root, lifecycle policy, etc.) is supplied by the caller as
    ``eval_fn`` so this module has no dependency on any particular evaluation harness.

    On every required epoch (divisible by ``interval_epochs``, or the final epoch), this callback:

    1. Materializes the live (regular) model weights into a self-describing checkpoint under
       ``output_dir/tracking_eval/epoch_<N>/checkpoint_regular.pth`` and calls
       ``eval_fn(TrackingEvalRequest(..., weight_flavor="regular"))``.
    2. If an EMA callback is active, does the same for the averaged weights (``checkpoint_ema.pth``,
       ``weight_flavor="ema"``).

    Every ``eval_fn`` return value (or ``None``, meaning skipped by the caller) is recorded, tagged with epoch and
    weight flavor, in ``self.results``.

    Resume safety: ``state_dict()``/``load_state_dict()`` persist which epochs have already been evaluated and the
    accumulated results, so ``trainer.fit(ckpt_path=...)`` neither re-evaluates a completed epoch nor loses prior
    results. The schedule itself needs no separate epoch counter — it is a pure function of
    ``trainer.current_epoch``, which PTL already restores correctly on resume.

    Args:
        output_dir: Training run's output directory. Materialized checkpoints are written under
            ``<output_dir>/tracking_eval/``.
        interval_epochs: Run evaluation every N completed epochs. The final epoch is always evaluated regardless of
            this value. Must be >= 1.
        eval_fn: Called once per materialized checkpoint. Receives a :class:`TrackingEvalRequest` and returns a
            JSON-serializable report dict (or ``None`` to record nothing for that request). Implementations are
            responsible for scoping evaluation to the correct dataset partition (for example, calibration-only —
            this callback has no notion of dataset splits and cannot enforce that itself).
    """

    def __init__(
        self,
        output_dir: str,
        interval_epochs: int,
        eval_fn: Callable[[TrackingEvalRequest], dict[str, Any] | None],
    ) -> None:
        super().__init__()
        self._output_dir = Path(output_dir) / "tracking_eval"
        self._interval_epochs = max(1, int(interval_epochs))
        self._eval_fn = eval_fn
        self._evaluated_epochs: list[int] = []
        self.results: list[dict[str, Any]] = []

    def _required_epoch(self, trainer: Trainer) -> tuple[int, bool] | None:
        """Return ``(epoch, is_final_epoch)`` when this epoch requires evaluation, else ``None``."""
        current_epoch = int(trainer.current_epoch) + 1
        max_epochs = trainer.max_epochs
        is_final_epoch = isinstance(max_epochs, int) and max_epochs > 0 and current_epoch >= max_epochs
        if is_final_epoch or current_epoch % self._interval_epochs == 0:
            return current_epoch, is_final_epoch
        return None

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Materialize checkpoints and invoke ``eval_fn`` when this epoch is on the schedule.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``RFDETRModelModule`` being trained.
        """
        if not trainer.is_global_zero:
            return
        required = self._required_epoch(trainer)
        if required is None:
            return
        epoch, is_final_epoch = required
        if epoch in self._evaluated_epochs:
            # Already evaluated (e.g. a resume restarted exactly at this epoch after a
            # crash between the eval running and the next epoch boundary) — idempotent.
            return

        epoch_dir = self._output_dir / f"epoch_{epoch:04d}"
        regular_path = epoch_dir / "checkpoint_regular.pth"
        self._materialize_checkpoint(trainer, pl_module, regular_path, weight_flavor="regular")
        self._run_eval(regular_path, "regular", epoch, is_final_epoch)

        ema_state_dict = self._get_ema_state_dict(trainer)
        if ema_state_dict is not None:
            ema_path = epoch_dir / "checkpoint_ema.pth"
            self._materialize_checkpoint(
                trainer, pl_module, ema_path, weight_flavor="ema", state_dict_override=ema_state_dict
            )
            self._run_eval(ema_path, "ema", epoch, is_final_epoch)

        self._evaluated_epochs.append(epoch)

    def _run_eval(self, checkpoint_path: Path, weight_flavor: str, epoch: int, is_final_epoch: bool) -> None:
        """Call ``eval_fn`` for one materialized checkpoint and record its result."""
        request = TrackingEvalRequest(
            checkpoint_path=checkpoint_path,
            weight_flavor=weight_flavor,  # type: ignore[arg-type]
            epoch=epoch,
            is_final_epoch=is_final_epoch,
        )
        logger.info(
            "Running scheduled chronological tracking evaluation for epoch %d (%s weights).", epoch, weight_flavor
        )
        report = self._eval_fn(request)
        if report is not None:
            self.results.append(
                {
                    "epoch": epoch,
                    "weight_flavor": weight_flavor,
                    "is_final_epoch": is_final_epoch,
                    "report": report,
                }
            )

    @staticmethod
    def _get_ema_state_dict(trainer: Trainer) -> dict[str, torch.Tensor] | None:
        """Return the averaged EMA model weights, or ``None`` when no EMA callback is active/warmed up."""
        for callback in trainer.callbacks:
            getter = getattr(callback, "get_ema_model_state_dict", None)
            if callable(getter):
                return getter()
        return None

    def _materialize_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        path: Path,
        *,
        weight_flavor: str,
        state_dict_override: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Write a self-describing checkpoint for one weight flavor at the current epoch.

        Reuses :class:`~rfdetr.training.callbacks.best_model.BestModelCallback`'s payload-construction helpers so
        every checkpoint this callback produces round-trips through the same generic loader as
        ``checkpoint_best_regular.pth``/``checkpoint_best_ema.pth`` (PRD US-004).
        """
        state_dict = (
            state_dict_override
            if state_dict_override is not None
            else BestModelCallback._get_live_model_state_dict(pl_module)
        )
        train_config = pl_module.train_config
        dataset_class_names = getattr(trainer.datamodule, "class_names", None)
        if (
            dataset_class_names is not None
            and hasattr(train_config, "model_copy")
            and getattr(train_config, "class_names", None) is None
        ):
            train_config = train_config.model_copy(update={"class_names": dataset_class_names})
        args_dict = serialize_train_config(train_config)
        model_name = BestModelCallback._resolve_model_name(pl_module)
        model_config_dict = BestModelCallback._serialize_model_config(pl_module, state_dict)
        source_hash = getattr(pl_module, "_source_checkpoint_hash", None)
        if not isinstance(source_hash, str):
            source_hash = None
        payload = BestModelCallback._build_checkpoint_payload(
            state_dict,
            args_dict,
            trainer,
            model_name=model_name,
            model_config_dict=model_config_dict,
        )
        if model_config_dict is not None:
            payload.update(
                authoritative_checkpoint_metadata(
                    model_config=pl_module.model_config,
                    train_config=train_config,
                    state_dict=state_dict,
                    epoch=trainer.current_epoch,
                    weight_flavor=weight_flavor,
                    source_checkpoint_hash_value=source_hash,
                )
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def state_dict(self) -> dict[str, Any]:
        """Return callback state so resume neither duplicates nor skips a required epoch's evaluation."""
        return {
            "evaluated_epochs": list(self._evaluated_epochs),
            "results": copy.deepcopy(self.results),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore evaluated-epoch bookkeeping and accumulated results from a checkpoint."""
        self._evaluated_epochs = list(state_dict.get("evaluated_epochs", []))
        self.results = list(state_dict.get("results", []))
