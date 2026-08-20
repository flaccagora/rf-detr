# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for :class:`TrackingEvalCallback` (PRD Section 7.5 chronological eval schedule)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning.trainer.states import TrainerFn
from rfdetr.training.callbacks.tracking_eval import TrackingEvalCallback, TrackingEvalRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(
    current_epoch: int,
    max_epochs: int | None = None,
    is_global_zero: bool = True,
    callbacks: list[object] | None = None,
) -> MagicMock:
    """Create a minimal mock Trainer exposing everything the callback touches."""
    trainer = MagicMock()
    trainer.current_epoch = current_epoch
    trainer.max_epochs = max_epochs
    trainer.is_global_zero = is_global_zero
    trainer.callbacks = callbacks or []
    trainer.global_step = 1
    trainer.state.fn = TrainerFn.FITTING
    trainer.datamodule.class_names = None
    return trainer


def _make_pl_module() -> MagicMock:
    """Create a minimal mock RFDETRModule with a picklable model state_dict."""
    pl_module = MagicMock()
    pl_module.model.state_dict.return_value = {"w": torch.zeros(1)}
    pl_module.model_config = None
    pl_module.train_config = {"lr": 0.001}
    return pl_module


class _EMACallback:
    """Fake EMA callback exposing the ``get_ema_model_state_dict`` contract."""

    def __init__(self, state_dict: dict[str, torch.Tensor] | None) -> None:
        self._state_dict = state_dict

    def get_ema_model_state_dict(self) -> dict[str, torch.Tensor] | None:
        return self._state_dict


_DEFAULT_REPORT = {"HOTA": 1.0}


class _Recorder:
    """Records every ``TrackingEvalRequest`` passed to it."""

    def __init__(self, report: dict[str, Any] | None = _DEFAULT_REPORT) -> None:
        self.requests: list[TrackingEvalRequest] = []
        self._report = report

    def __call__(self, request: TrackingEvalRequest) -> dict[str, Any] | None:
        self.requests.append(request)
        return self._report


# ---------------------------------------------------------------------------
# TestTrackingEvalCallback
# ---------------------------------------------------------------------------


class TestTrackingEvalCallback:
    """Verify the interval + final-epoch schedule, EMA handling, and resume safety."""

    def test_skips_non_divisible_non_final_epoch(self, tmp_path: Path) -> None:
        """Epoch 1 (of an unbounded run) with interval=5 is not on the schedule."""
        recorder = _Recorder()
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=recorder)
        trainer = _make_trainer(current_epoch=0, max_epochs=None)

        callback.on_validation_end(trainer, _make_pl_module())

        assert recorder.requests == []
        assert callback.results == []

    def test_runs_on_divisible_epoch(self, tmp_path: Path) -> None:
        """Completed epoch 5 (trainer.current_epoch=4) with interval=5 is required."""
        recorder = _Recorder()
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=recorder)
        trainer = _make_trainer(current_epoch=4, max_epochs=None)

        callback.on_validation_end(trainer, _make_pl_module())

        assert [r.epoch for r in recorder.requests] == [5]
        assert recorder.requests[0].weight_flavor == "regular"
        assert recorder.requests[0].is_final_epoch is False
        assert recorder.requests[0].checkpoint_path.is_file()
        assert len(callback.results) == 1
        assert callback.results[0]["report"] == {"HOTA": 1.0}

    def test_runs_on_final_epoch_even_when_not_divisible(self, tmp_path: Path) -> None:
        """The final epoch is always evaluated regardless of the interval."""
        recorder = _Recorder()
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=recorder)
        trainer = _make_trainer(current_epoch=6, max_epochs=7)  # completed epoch 7 of 7

        callback.on_validation_end(trainer, _make_pl_module())

        assert [r.epoch for r in recorder.requests] == [7]
        assert recorder.requests[0].is_final_epoch is True

    def test_evaluates_both_regular_and_ema_weights_when_ema_active(self, tmp_path: Path) -> None:
        """PRD AC: EMA weights are evaluated periodically alongside regular weights."""
        recorder = _Recorder()
        ema_state = {"w": torch.ones(1)}
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=recorder)
        trainer = _make_trainer(current_epoch=4, callbacks=[_EMACallback(ema_state)])

        callback.on_validation_end(trainer, _make_pl_module())

        flavors = {r.weight_flavor for r in recorder.requests}
        assert flavors == {"regular", "ema"}
        assert all(r.epoch == 5 for r in recorder.requests)
        assert len(callback.results) == 2

    def test_final_epoch_evaluates_both_regular_and_ema(self, tmp_path: Path) -> None:
        """PRD AC: the final epoch evaluates both regular and EMA weights."""
        recorder = _Recorder()
        ema_state = {"w": torch.ones(1)}
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=recorder)
        trainer = _make_trainer(current_epoch=6, max_epochs=7, callbacks=[_EMACallback(ema_state)])

        callback.on_validation_end(trainer, _make_pl_module())

        assert {r.weight_flavor for r in recorder.requests} == {"regular", "ema"}
        assert all(r.is_final_epoch for r in recorder.requests)

    def test_no_ema_callback_present_only_evaluates_regular(self, tmp_path: Path) -> None:
        """Without an active EMA callback, only regular weights are evaluated."""
        recorder = _Recorder()
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=recorder)
        trainer = _make_trainer(current_epoch=4, callbacks=[])

        callback.on_validation_end(trainer, _make_pl_module())

        assert [r.weight_flavor for r in recorder.requests] == ["regular"]

    def test_ema_callback_present_but_not_warmed_up_skips_ema(self, tmp_path: Path) -> None:
        """An EMA callback that returns None (not yet warmed up) is treated as absent."""
        recorder = _Recorder()
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=recorder)
        trainer = _make_trainer(current_epoch=4, callbacks=[_EMACallback(None)])

        callback.on_validation_end(trainer, _make_pl_module())

        assert [r.weight_flavor for r in recorder.requests] == ["regular"]

    def test_skips_on_non_global_zero_rank(self, tmp_path: Path) -> None:
        """DDP: only global rank zero materializes checkpoints and calls eval_fn."""
        recorder = _Recorder()
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=recorder)
        trainer = _make_trainer(current_epoch=4, is_global_zero=False)

        callback.on_validation_end(trainer, _make_pl_module())

        assert recorder.requests == []

    def test_eval_fn_returning_none_records_nothing_but_still_marks_epoch_done(self, tmp_path: Path) -> None:
        """A caller that opts out of recording a report leaves ``results`` empty but is not re-run on retry."""
        recorder = _Recorder(report=None)
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=recorder)
        trainer = _make_trainer(current_epoch=4)

        callback.on_validation_end(trainer, _make_pl_module())
        callback.on_validation_end(trainer, _make_pl_module())  # simulate a duplicate hook call

        assert callback.results == []
        assert len(recorder.requests) == 1  # not called again for the same epoch

    def test_duplicate_call_for_same_epoch_is_idempotent(self, tmp_path: Path) -> None:
        """Two ``on_validation_end`` calls for the same completed epoch evaluate only once.

        This is the mechanism that keeps a resume from re-running an epoch's evaluation twice if
        ``on_validation_end`` fired before a crash and training resumes exactly at the next epoch boundary.
        """
        recorder = _Recorder()
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=recorder)
        trainer = _make_trainer(current_epoch=4)

        callback.on_validation_end(trainer, _make_pl_module())
        callback.on_validation_end(trainer, _make_pl_module())

        assert len(recorder.requests) == 1
        assert len(callback.results) == 1

    def test_state_dict_round_trip_restores_evaluated_epochs_and_results(self, tmp_path: Path) -> None:
        """Resume-safety: state_dict()/load_state_dict() preserve schedule bookkeeping across a fresh instance."""
        recorder = _Recorder()
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=recorder)
        trainer = _make_trainer(current_epoch=4)
        callback.on_validation_end(trainer, _make_pl_module())

        state = callback.state_dict()
        resumed = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=recorder)
        resumed.load_state_dict(state)

        assert resumed.results == callback.results
        assert resumed._evaluated_epochs == [5]

        # Resume lands back at the same completed epoch (e.g. a crash between this epoch's
        # eval and the next epoch boundary) — must not duplicate the evaluation.
        resumed.on_validation_end(_make_trainer(current_epoch=4), _make_pl_module())
        assert len(recorder.requests) == 1
        assert len(resumed.results) == 1

    def test_load_state_dict_defaults_when_keys_absent(self, tmp_path: Path) -> None:
        """Loading a checkpoint saved before this callback existed must not raise."""
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=_Recorder())

        callback.load_state_dict({})

        assert callback._evaluated_epochs == []
        assert callback.results == []

    def test_interval_epochs_is_clamped_to_at_least_one(self, tmp_path: Path) -> None:
        """A non-positive interval degrades to evaluating every epoch rather than never/crashing."""
        recorder = _Recorder()
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=0, eval_fn=recorder)
        trainer = _make_trainer(current_epoch=0)

        callback.on_validation_end(trainer, _make_pl_module())

        assert len(recorder.requests) == 1

    @pytest.mark.parametrize("completed_epoch", [1, 2, 3, 4])
    def test_only_the_fifth_epoch_in_a_five_epoch_run_is_final(self, tmp_path: Path, completed_epoch: int) -> None:
        """Epochs 1-4 of a 5-epoch run are neither divisible by 5 nor final."""
        recorder = _Recorder()
        callback = TrackingEvalCallback(output_dir=str(tmp_path), interval_epochs=5, eval_fn=recorder)
        trainer = _make_trainer(current_epoch=completed_epoch - 1, max_epochs=5)

        callback.on_validation_end(trainer, _make_pl_module())

        assert recorder.requests == []
