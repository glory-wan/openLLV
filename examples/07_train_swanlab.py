"""Train Zero-DCE with SwanLab recording at both batch and epoch level.

openLLV's ``Trainer`` has no built-in SwanLab support, so this example
subclasses it and hooks two points:

- Batch level: override ``train_one_epoch``/``validate`` to log the loss,
  learning rate, and gradient norm per batch.
- Epoch level: override ``_save_history`` (called once per epoch by the
  base loop) to log the epoch summary.

Console output is a single line per epoch, nothing else::

    Epoch 1/2 | train_loss=3.172775 | val_loss=3.119804

SwanLab is not an openLLV dependency; install it separately:

    pip install swanlab
"""

from __future__ import annotations

import time
from datetime import datetime

import swanlab
import torch
from _utils import create_tiny_common_dataset, ensure_results_dir

from openLLV.deepLearning import Trainer


class SwanLabTrainer(Trainer):
    """Trainer subclass that records epoch-level metrics with SwanLab.

    Args:
        swan_project: SwanLab project name passed to ``swanlab.init``.
        swan_experiment: SwanLab experiment name passed to
            ``swanlab.init``; defaults to the model class name.
    """

    def __init__(
        self,
        *args,
        swan_project: str = "openLLV",
        swan_experiment: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        swanlab.init(
            project=swan_project,
            experiment_name=(swan_experiment or None),
            config=self._to_yaml_safe(self.config),
        )

    def print_training_info(self) -> None:
        """Keep the console clean: skip the setup banner."""

    def _save_history(self) -> None:
        """Write the JSON history, then log the epoch summary to SwanLab."""
        super()._save_history()
        latest = self.history[-1]
        metrics = {
            "train/epoch_loss": latest["train_loss"],
            "epoch/seconds": latest["seconds"],
        }
        if latest.get("val_loss") is not None:
            metrics["val/epoch_loss"] = latest["val_loss"]
        swanlab.log(metrics, step=latest["epoch"])

    def train(self) -> dict:
        """Run the base training loop printing only the epoch summary.

        Mirrors ``Trainer.train`` while skipping its ``begin Training``
        banner, per-epoch blank lines, and the ``Finished training`` block;
        keep in sync with upstream changes in
        ``openLLV/deepLearning/trainer.py``.
        """
        epochs = int(self.config["train"]["epochs"])
        if self.training_started_at is None:
            self.training_started_at = (
                datetime.now().astimezone().isoformat(timespec="seconds")
            )
            self.training_ended_at = None
            self._save_training_config()

        try:
            for epoch in range(self.start_epoch, epochs + 1):
                epoch_start = time.time()
                train_loss = self.train_one_epoch(epoch)

                val_loss = None
                if (
                    self.val_loader is not None
                    and epoch % int(self.config["train"]["validate_every"]) == 0
                ):
                    val_loss = self.validate(epoch)

                self._step_scheduler(val_loss, train_loss)

                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "seconds": time.time() - epoch_start,
                }
                self.history.append(record)
                self._save_history()

                is_best = False
                if val_loss is not None and self.best_val_loss is not None:
                    is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                if epoch % int(self.config["train"]["save_every"]) == 0:
                    self.save_checkpoint("last.pt", epoch, val_loss)
                if is_best:
                    self.save_checkpoint("best.pt", epoch, val_loss)

                print(
                    f"Epoch {epoch}/{epochs} | train_loss={train_loss:.6f}"
                    + (f" | val_loss={val_loss:.6f}" if val_loss is not None else "")
                )
        finally:
            self.training_ended_at = (
                datetime.now().astimezone().isoformat(timespec="seconds")
            )
            self._save_training_config()
            swanlab.finish()

        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "checkpoint_dir": str(self.checkpoint_dir),
        }


class BatchSwanLabTrainer(SwanLabTrainer):
    """Trainer subclass that additionally logs per-batch metrics.

    Overriding ``train_one_epoch``/``validate`` duplicates the base-class
    loops; keep an eye on upstream changes in
    ``openLLV/deepLearning/trainer.py`` so the copies stay in sync.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Global step counters keep the curves continuous across epochs.
        self._train_step = 0
        self._val_step = 0

    def train_one_epoch(self, epoch: int) -> float:
        """Base training loop plus per-batch SwanLab logging."""
        self.model.train_mode()
        total_loss = 0.0
        total_samples = 0
        log_every = int(self.config["train"]["log_every"])
        grad_clip = self.config["train"].get("grad_clip")

        for batch in self.train_loader:
            batch_size = self._batch_size(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                enabled=self.amp_enabled,
            ):
                loss, _ = self._compute_batch_loss(batch)

            self.scaler.scale(loss).backward()

            # Gradient norm must be captured here, after backward and before
            # step (the base class zeroes gradients at the next batch).
            # Unscale first so AMP-scaled gradients report their true norm
            # (a no-op when AMP is disabled).
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                float("inf"),
            )
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(grad_clip),
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            self._train_step += 1

            if self._train_step % log_every == 0 or self._train_step == 1:
                swanlab.log(
                    {
                        "train/loss": loss.item(),
                        "train/grad_norm": float(grad_norm),
                    },
                    step=self._train_step,
                )

        if total_samples == 0:
            raise RuntimeError(
                "The training dataloader produced no samples. Check "
                "drop_last, batch_size, and dataset contents."
            )
        return total_loss / total_samples

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Base validation loop plus per-batch SwanLab logging."""
        self.model.eval_mode()
        total_loss = 0.0
        total_samples = 0

        if self.val_loader is None:
            raise RuntimeError("Validation was requested without a val_loader.")

        for batch in self.val_loader:
            loss, _ = self._compute_batch_loss(batch)

            batch_size = self._batch_size(batch)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            self._val_step += 1
            swanlab.log(
                {"val/loss": loss.item()},
                step=self._val_step,
            )

        if total_samples == 0:
            raise RuntimeError("The validation dataloader produced no samples.")
        return total_loss / total_samples


def main() -> None:
    """Train ZeroDCE for two epochs on a generated tiny dataset."""
    dataset_root = create_tiny_common_dataset(ensure_results_dir("tiny_dataset"))
    output_dir = ensure_results_dir("checkpoints", "ZeroDCE_swanlab")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = BatchSwanLabTrainer(
        swan_project="openLLV",
        model="ZeroDCE",
        model_params={"input_channels": 3},
        dataset="CommonDataset",
        root_dir=str(dataset_root),
        train_split="train",
        val_split="val",
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        loss="zerodce",
        optimizer="adam",
        lr=1e-4,
        epochs=100,
        device=device,
        output_dir=str(output_dir),
        log_every=1,
        save_every=1,
        validate_every=1,
        progress_bar=False,
    )

    trainer.train()


if __name__ == "__main__":
    main()
