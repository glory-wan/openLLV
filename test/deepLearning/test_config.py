"""Tests for built-in deep-learning training configurations."""

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openLLV.data.datasets import BaseDataset
from openLLV.deepLearning.config import (
    CONFIG_DIR,
    DEFAULT_TRAIN_CONFIG,
    deep_update,
    get_config_path,
    get_default_train_config,
    list_available_configs,
    load_config,
)
from openLLV.deepLearning.loss import BaseLoss
from openLLV.deepLearning.models import LLVModel
from openLLV.deepLearning.trainer import Trainer


EXPECTED_CONFIGS = {
    "CIDNet",
    "DarkIR",
    "EnlightenGAN",
    "KinD",
    "KinD++",
    "LEDNet",
    "LLFlow",
    "LLFormer",
    "LLNet",
    "PairLIE",
    "RetinexFormer",
    "RUAS",
    "SCI",
    "URetinexNet",
    "ZeroDCE",
    "ZeroDCE++",
    "ZeroIG",
}

REQUIRED_SECTIONS = {
    "model",
    "data",
    "loss",
    "optimizer",
    "scheduler",
    "train",
}


class BuiltInConfigDiscoveryTests(unittest.TestCase):
    def test_config_module_is_a_package_with_all_migrated_yaml_files(self):
        self.assertTrue(CONFIG_DIR.is_dir())
        self.assertEqual(set(list_available_configs()), EXPECTED_CONFIGS)
        self.assertEqual(
            {path.stem for path in CONFIG_DIR.glob("*.yaml")},
            EXPECTED_CONFIGS,
        )

    def test_resolves_filename_stem_model_name_and_plus_aliases(self):
        cases = {
            "CIDNet": "CIDNet.yaml",
            "cidnet.yaml": "CIDNet.yaml",
            "KinD++": "KinD++.yaml",
            "kind_plus_plus": "KinD++.yaml",
            "KinDPlusPlus": "KinD++.yaml",
            "ZeroDCE": "ZeroDCE.yaml",
            "ZeroDCE++": "ZeroDCE++.yaml",
            "zero_dce_plus_plus": "ZeroDCE++.yaml",
            "ZeroDCEPlusPlus": "ZeroDCE++.yaml",
        }

        for config_name, expected_filename in cases.items():
            with self.subTest(config_name=config_name):
                self.assertEqual(
                    get_config_path(config_name).name,
                    expected_filename,
                )

    def test_explicit_paths_are_supported_and_missing_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "custom.yaml"
            config_path.write_text("train:\n  epochs: 2\n", encoding="utf-8")
            self.assertEqual(get_config_path(config_path), config_path.resolve())
            self.assertEqual(load_config(config_path)["train"]["epochs"], 2)

            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                get_config_path(Path(temp_dir) / "ZeroDCE.yaml")

        with self.assertRaisesRegex(FileNotFoundError, "Available configs"):
            get_config_path("unknown-config")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            get_config_path("  ")
        with self.assertRaisesRegex(TypeError, "string or Path"):
            get_config_path(object())

    def test_load_config_returns_an_independent_mapping(self):
        first = load_config("ZeroDCE")
        second = load_config("ZeroDCE")

        first["model"]["params"]["number_f"] = 8

        self.assertNotIn("number_f", second["model"]["params"])


class BuiltInConfigCompatibilityTests(unittest.TestCase):
    def test_every_yaml_uses_registered_openllv_components(self):
        registered_models = set(LLVModel.list_registered_models())
        registered_losses = set(BaseLoss.list_registered_losses())
        registered_datasets = set(BaseDataset.list_registered_datasets())

        for config_name in list_available_configs():
            with self.subTest(config_name=config_name):
                config = load_config(config_name)

                self.assertEqual(set(config), REQUIRED_SECTIONS)
                self.assertIn(
                    config["model"]["name"].strip().lower(),
                    registered_models,
                )
                self.assertIn(
                    config["loss"]["name"].strip().lower(),
                    registered_losses,
                )
                self.assertIn(
                    config["data"]["dataset"].strip().lower(),
                    registered_datasets,
                )
                self.assertEqual(config["data"]["dataset"], "CommonDataset")
                self.assertIsNone(config["train"]["device"])
                self.assertIsNone(config["train"]["output_dir"])
                self.assertIsInstance(config["model"]["params"], dict)
                self.assertIsInstance(config["loss"]["params"], dict)

    def test_every_model_parameter_set_passes_current_model_validation(self):
        for config_name in list_available_configs():
            config = load_config(config_name)
            model_name = config["model"]["name"]
            model_class = LLVModel._model_registry[
                LLVModel._normalize_key(model_name)
            ]

            with self.subTest(config_name=config_name), patch.object(
                model_class,
                "_init_model",
                autospec=True,
                return_value=None,
            ):
                model = model_class(config=config["model"]["params"])
                self.assertEqual(model.task, "llie")

    def test_every_loss_parameter_is_accepted_by_registered_constructor(self):
        for config_name in list_available_configs():
            config = load_config(config_name)
            loss_name = config["loss"]["name"]
            loss_class = BaseLoss._loss_registry[
                BaseLoss._normalize_key(loss_name)
            ]
            parameters = inspect.signature(loss_class.__init__).parameters
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )

            with self.subTest(config_name=config_name):
                for parameter_name in config["loss"]["params"]:
                    self.assertTrue(
                        accepts_kwargs or parameter_name in parameters,
                        f"{loss_class.__name__} does not accept "
                        f"{parameter_name!r}",
                    )

    def test_trainer_loader_accepts_builtin_names_and_merges_defaults(self):
        loaded = Trainer._load_config("ZeroDCEPlusPlus")
        merged = Trainer._with_defaults(loaded)

        self.assertEqual(loaded["model"]["name"], "ZeroDCEPlusPlus")
        self.assertEqual(loaded["loss"]["name"], "zerodce_extension")
        self.assertTrue(merged["data"]["shuffle"])
        self.assertTrue(merged["train"]["strict_resume"])
        self.assertTrue(merged["train"]["progress_bar"])

    def test_original_config_helpers_remain_compatible(self):
        first = get_default_train_config()
        second = get_default_train_config()

        self.assertEqual(first, DEFAULT_TRAIN_CONFIG)
        first["model"]["params"]["width"] = 8
        self.assertNotIn("width", second["model"]["params"])
        self.assertEqual(
            deep_update({"train": {"epochs": 10}}, {"train": {"epochs": 2}}),
            {"train": {"epochs": 2}},
        )


if __name__ == "__main__":
    unittest.main()
