"""Tests for the configuration-driven deep-learning Trainer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image

import openLLV.deepLearning as deep_learning_package
from openLLV.data.datasets import BaseDataset
from openLLV.deepLearning import Trainer
from openLLV.deepLearning.config import (
    DEFAULT_TRAIN_CONFIG,
    deep_update,
    get_default_device,
    get_default_train_config,
)
from openLLV.deepLearning.loss import BaseLoss
from openLLV.deepLearning.models import LLVModel, ZeroDCE


class TrainerTinyPairedDataset(BaseDataset):
    """In-memory paired dataset used to exercise the trainer."""

    name = "trainer-tiny-paired"
    aliases = ["trainer-paired"]

    def __init__(
        self,
        root_dir: str | Path = ".",
        split: str = "train",
        return_filename: bool = True,
        length: int = 4,
        image_size: int = 8,
        **kwargs,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.return_filename = return_filename
        self.length = int(length)
        self.image_size = int(image_size)

    def _resolve_pair_dirs(self, input_dir, target_dir):
        return Path("."), Path(".")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        value = 0.05 + 0.1 * (index + 1)
        input_tensor = torch.full(
            (3, self.image_size, self.image_size),
            value,
            dtype=torch.float32,
        )
        target = torch.clamp(input_tensor * 0.5 + 0.35, 0.0, 1.0)
        if self.return_filename:
            return input_tensor, target, f"{self.split}-{index}.png"
        return input_tensor, target


class TrainerTinyUnpairedDataset(TrainerTinyPairedDataset):
    """In-memory dataset whose target field is intentionally None."""

    name = "trainer-tiny-unpaired"
    aliases = ["trainer-unpaired"]

    def __getitem__(self, index: int):
        input_tensor, _, filename = super().__getitem__(index)
        return input_tensor, None, filename


class TrainerToyModel(LLVModel):
    """Small trainable model with the standard structured output contract."""

    task = "test"
    aliases = ["trainer-toy"]

    def _get_default_config(self):
        config = super()._get_default_config()
        config.update({"mode": "inference"})
        return config

    def _init_model(self) -> None:
        self.projection = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, image: torch.Tensor):
        prediction = torch.sigmoid(self.projection(image))
        if self.config.get("mode") == "train":
            return self._format_output(
                prediction,
                aux={"projection": prediction},
                meta={"mode": "train"},
            )
        return prediction


class TrainerGrandchildModel(TrainerToyModel):
    """Indirect LLVModel descendant used to verify registry construction."""

    aliases = ["trainer-grandchild"]


class TrainerReferenceFreeLoss(BaseLoss):
    """Simple differentiable loss for unpaired trainer tests."""

    name = "trainer-reference-free"
    requires_target = False

    def forward(self, input_tensor, model_output):
        prediction = model_output["pred"]
        return (prediction - input_tensor).abs().mean()


def trainer_config(
    output_dir: Path,
    *,
    model=TrainerGrandchildModel,
    dataset=TrainerTinyPairedDataset,
    epochs: int = 1,
    val_split: str | None = None,
):
    """Build a minimal CPU trainer configuration."""
    return {
        "model": {"name": model, "params": {}},
        "data": {
            "dataset": dataset,
            "root_dir": ".",
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
            "train_split": "train",
            "val_split": val_split,
            "return_filename": True,
            "params": {"length": 4, "image_size": 8},
            "train_params": {},
            "val_params": {},
        },
        "loss": {"name": "mse", "params": {}},
        "optimizer": {"name": "sgd", "lr": 0.1, "params": {}},
        "scheduler": {"name": None, "params": {}},
        "train": {
            "epochs": epochs,
            "output_dir": str(output_dir),
            "save_every": 1,
            "validate_every": 1,
            "log_every": 1,
            "grad_clip": None,
            "amp": False,
            "resume": None,
            "strict_resume": True,
            "seed": 7,
            "device": "cpu",
            "progress_bar": False,
        },
    }


class TrainerConfigTests(unittest.TestCase):
    def test_default_config_is_independent_and_uses_current_dataset(self):
        first = get_default_train_config()
        second = get_default_train_config()
        self.assertEqual(first, DEFAULT_TRAIN_CONFIG)
        self.assertEqual(first["data"]["dataset"], "CommonDataset")
        self.assertIsNone(first["data"]["resize"])
        self.assertEqual(first["optimizer"]["lr"], 1e-4)
        self.assertEqual(first["train"]["epochs"], 100)

        first["model"]["params"]["width"] = 4
        self.assertNotIn("width", second["model"]["params"])

    def test_default_device_prefers_cuda_then_cpu(self):
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(get_default_device(), "cuda")
        with patch("torch.cuda.is_available", return_value=False), patch.object(
            torch.backends,
            "mps",
            None,
            create=True,
        ):
            self.assertEqual(get_default_device(), "cpu")

    def test_deep_update_merges_nested_values_and_preserves_runtime_objects(self):
        model = TrainerToyModel()
        base = {"train": {"epochs": 10, "seed": 1}}
        updates = {"train": {"epochs": 2}, "model": {"name": model}}
        merged = deep_update(base, updates)
        self.assertIs(merged, base)
        self.assertEqual(merged["train"], {"epochs": 2, "seed": 1})
        self.assertIs(merged["model"]["name"], model)

    def test_with_defaults_and_flat_kwargs_follow_override_precedence(self):
        merged = Trainer._with_defaults(
            {"optimizer": {"lr": 0.02}, "train": {"epochs": 3}}
        )
        self.assertEqual(merged["optimizer"]["name"], "adam")
        self.assertEqual(merged["optimizer"]["lr"], 0.02)
        self.assertEqual(merged["train"]["epochs"], 3)
        self.assertEqual(merged["data"]["batch_size"], 4)

        flat = Trainer._kwargs_to_config(
            {
                "model": "trainer-toy",
                "batch_size": 2,
                "resize": (16, 24),
                "train_input_dir": "train/input",
                "train_target_dir": "train/target",
                "loss_name": "l1",
                "lr": 0.01,
                "epochs": 4,
                "progress_bar": False,
            }
        )
        self.assertEqual(flat["model"]["name"], "trainer-toy")
        self.assertEqual(flat["data"]["batch_size"], 2)
        self.assertEqual(flat["data"]["resize"], (16, 24))
        self.assertEqual(flat["data"]["train_input_dir"], "train/input")
        self.assertEqual(flat["data"]["train_target_dir"], "train/target")
        self.assertEqual(flat["loss"]["name"], "l1")
        self.assertEqual(flat["train"]["epochs"], 4)

    def test_yaml_loader_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.yaml"
            path.write_text("train:\n  epochs: 2\n", encoding="utf-8")
            self.assertEqual(Trainer._load_config(path)["train"]["epochs"], 2)

            invalid = Path(temp_dir) / "invalid.yaml"
            invalid.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "YAML mapping"):
                Trainer._load_config(invalid)

        with self.assertRaises(FileNotFoundError):
            Trainer._load_config("missing-trainer-config.yaml")
        with self.assertRaises(TypeError):
            Trainer._load_config(object())
        with self.assertRaisesRegex(TypeError, "Unsupported Trainer argument"):
            Trainer._kwargs_to_config({"unknown": 1})


class TrainerConstructionTests(unittest.TestCase):
    def test_registered_name_class_and_instance_construct_descendant_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            instance = TrainerGrandchildModel()
            cases = (
                ("trainer-grandchild", TrainerGrandchildModel, None),
                (TrainerGrandchildModel, TrainerGrandchildModel, None),
                (instance, TrainerGrandchildModel, instance),
            )
            for index, (
                model_input,
                expected_class,
                expected_instance,
            ) in enumerate(cases):
                with self.subTest(model=model_input):
                    config = trainer_config(root / str(index), model=model_input)
                    trainer = Trainer(config)
                    self.assertIsInstance(trainer.model, expected_class)
                    if expected_instance is not None:
                        self.assertIs(trainer.model, expected_instance)
                    self.assertEqual(
                        next(trainer.model.parameters()).device.type,
                        "cpu",
                    )
                    self.assertNotIn("device", trainer.model.config)
                    self.assertEqual(trainer.model.config["mode"], "train")

    def test_model_checkpoint_path_reconstructs_registered_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = TrainerToyModel()
            with torch.no_grad():
                source.projection.weight.fill_(0.25)
            checkpoint = source.save_model(root / "source")

            trainer = Trainer(
                trainer_config(root / "run", model=str(checkpoint))
            )
            self.assertIsInstance(trainer.model, TrainerToyModel)
            torch.testing.assert_close(
                trainer.model.projection.weight,
                source.projection.weight,
            )

    def test_default_loss_matches_model_then_falls_back_to_charbonnier(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            toy_config = trainer_config(root / "toy")
            toy_config["loss"]["name"] = None
            toy = Trainer(toy_config)
            self.assertEqual(toy.criterion.__class__.__name__, "CharbonnierLoss")

            zero_config = trainer_config(root / "zero", model="ZeroDCE")
            zero_config["model"]["params"] = {"number_f": 4}
            zero_config["loss"]["name"] = None
            zero = Trainer(zero_config)
            self.assertIsInstance(zero.model, ZeroDCE)
            self.assertEqual(zero.criterion.__class__.__name__, "ZeroDCE_Loss")

    def test_package_exports_trainer(self):
        self.assertIs(deep_learning_package.Trainer, Trainer)
        self.assertIn("Trainer", deep_learning_package.__all__)


class TrainerExecutionTests(unittest.TestCase):
    def test_paired_training_validation_scheduler_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "paired"
            config = trainer_config(
                output_dir,
                epochs=2,
                val_split="val",
            )
            config["scheduler"] = {
                "name": "StepLR",
                "params": {"step_size": 1, "gamma": 0.5},
            }
            trainer = Trainer(config)
            initial_weight = trainer.model.projection.weight.detach().clone()

            result = trainer.train()

            self.assertEqual(len(result["history"]), 2)
            self.assertTrue(
                all(
                    record["val_loss"] is not None
                    for record in result["history"]
                )
            )
            self.assertFalse(
                torch.equal(initial_weight, trainer.model.projection.weight)
            )
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.025)
            self.assertTrue((output_dir / "checkpoints" / "last.pt").is_file())
            self.assertTrue((output_dir / "checkpoints" / "best.pt").is_file())
            self.assertTrue((output_dir / "logs" / "history.json").is_file())
            self.assertTrue((output_dir / "TrainerGrandchildModel.yaml").is_file())

    def test_training_checkpoint_resume_restores_state_and_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "resume"
            first = Trainer(trainer_config(output_dir, epochs=1))
            first.train()
            checkpoint = output_dir / "checkpoints" / "last.pt"

            resumed_config = trainer_config(output_dir, epochs=2)
            resumed_config["train"]["resume"] = str(checkpoint)
            resumed = Trainer(resumed_config)
            self.assertEqual(resumed.start_epoch, 2)
            self.assertEqual(len(resumed.history), 1)

            result = resumed.train()
            self.assertEqual([item["epoch"] for item in result["history"]], [1, 2])

    def test_reference_free_training_collates_none_targets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = TrainerTinyUnpairedDataset(
                length=2,
                image_size=8,
            )
            config = trainer_config(
                Path(temp_dir) / "reference-free",
                model=TrainerToyModel,
                dataset=dataset,
            )
            config["data"]["root_dir"] = None
            config["data"]["batch_size"] = 1
            config["loss"] = {
                "name": TrainerReferenceFreeLoss,
                "params": {},
            }

            trainer = Trainer(config)
            result = trainer.train()
            self.assertEqual(len(result["history"]), 1)
            self.assertIsNone(result["history"][0]["val_loss"])

    def test_real_zerodce_descendant_trains_with_default_loss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "data" / "train" / "input"
            input_dir.mkdir(parents=True)
            for index, value in enumerate((24, 48)):
                image = np.full((16, 16, 3), value, dtype=np.uint8)
                Image.fromarray(image, mode="RGB").save(
                    input_dir / f"{index}.png"
                )

            config = trainer_config(
                root / "zerodce",
                model="ZeroDCE",
            )
            config["model"]["params"] = {"number_f": 4}
            del config["data"]["dataset"]
            config["data"]["root_dir"] = str(root / "data")
            config["data"]["batch_size"] = 1
            config["data"]["params"] = {}
            config["loss"] = {"name": None, "params": {}}

            trainer = Trainer(config)
            result = trainer.train()
            self.assertIsInstance(trainer.model, ZeroDCE)
            self.assertEqual(
                trainer.config["data"]["dataset"],
                "CommonDataset",
            )
            self.assertFalse(trainer.criterion.requires_target)
            self.assertEqual(len(result["history"]), 1)
            self.assertTrue(
                (Path(result["checkpoint_dir"]) / "last.pt").is_file()
            )


if __name__ == "__main__":
    unittest.main()
