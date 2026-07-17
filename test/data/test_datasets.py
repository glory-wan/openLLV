"""Tests for :mod:`openLLV.data.datasets`."""

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

import openLLV.data as data_package
from openLLV.data.datasets import BaseDataset, CommonDataset


def write_image(path, value=128, image_format=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), (value, value, value)).save(
        path,
        format=image_format,
    )
    return path


def make_pair_layout(root, split="train", low_name="sample.png", high_name="sample.png"):
    low = write_image(Path(root) / split / "input" / low_name, value=32)
    high = write_image(Path(root) / split / "target" / high_name, value=192)
    return low.parent, high.parent


class BaseDatasetRegistryTests(unittest.TestCase):
    def setUp(self):
        self._registry_snapshot = dict(BaseDataset._dataset_registry)

    def tearDown(self):
        BaseDataset._dataset_registry.clear()
        BaseDataset._dataset_registry.update(self._registry_snapshot)

    def test_base_dataset_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseDataset(root_dir="unused")

    def test_registry_contains_common_dataset_name_and_paired_alias(self):
        registered = BaseDataset.list_registered_datasets()

        for name in ("commondataset", "paireddataset"):
            with self.subTest(name=name):
                self.assertIn(name, registered)
                self.assertIs(BaseDataset.get_dataset_class(name.upper()), CommonDataset)

        self.assertEqual(CommonDataset.aliases, ["PairedDataset"])

    def test_registry_key_normalization(self):
        self.assertEqual(
            BaseDataset._normalize_registry_key("  PairedDataset  "),
            "paireddataset",
        )

    def test_subclass_is_registered_automatically_by_name_and_alias(self):
        class DemoDataset(BaseDataset):
            name = "Demo"
            aliases = ["demo_alias"]

            def _resolve_pair_dirs(self, low_dir, high_dir):
                return low_dir or self.root_dir / "low", high_dir

        self.assertIs(BaseDataset.get_dataset_class("DemoDataset"), DemoDataset)
        self.assertIs(BaseDataset.get_dataset_class("demo"), DemoDataset)
        self.assertIs(BaseDataset.get_dataset_class("DEMO_ALIAS"), DemoDataset)

    def test_manual_registration_rejects_non_dataset_class(self):
        with self.assertRaisesRegex(TypeError, "subclass of BaseDataset"):
            BaseDataset.register(str)

    def test_create_dataset_uses_registered_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            make_pair_layout(temp_dir)

            dataset = BaseDataset.create_dataset(
                "PairedDataset",
                root_dir=temp_dir,
                split="train",
            )

        self.assertIsInstance(dataset, CommonDataset)
        self.assertEqual(len(dataset), 1)

    def test_unknown_or_empty_dataset_name_has_clear_error(self):
        for invalid in (None, "", "   "):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "non-empty string"):
                    BaseDataset.get_dataset_class(invalid)

        with self.assertRaises(ValueError) as context:
            BaseDataset.get_dataset_class("commn")

        message = str(context.exception)
        self.assertIn("Available datasets", message)
        self.assertIn("Did you mean", message)
        self.assertIn("commondataset", message)

    def test_similar_name_helper_has_match_and_fallback(self):
        self.assertEqual(
            BaseDataset._get_similar_dataset_name(
                "paireddatset",
                ["paireddataset"],
            ),
            "paireddataset",
        )
        self.assertEqual(
            BaseDataset._get_similar_dataset_name("xyz", ["paireddataset"]),
            "No similar datasets found",
        )


class CommonDatasetResolutionTests(unittest.TestCase):
    def test_resolves_standard_train_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            low_dir, high_dir = make_pair_layout(temp_dir, split="train")

            dataset = CommonDataset(temp_dir, split="train")

        self.assertEqual(dataset.low_dir, low_dir)
        self.assertEqual(dataset.high_dir, high_dir)

    def test_resolves_validation_alias_and_case_variant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            low_dir, high_dir = make_pair_layout(temp_dir, split="Val")

            dataset = CommonDataset(temp_dir, split="validation")

        self.assertEqual(dataset.low_dir, low_dir)
        self.assertEqual(dataset.high_dir, high_dir)

    def test_resolves_root_level_input_target_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            low_dir = write_image(Path(temp_dir) / "input" / "a.png").parent
            high_dir = write_image(Path(temp_dir) / "target" / "a.png").parent

            dataset = CommonDataset(temp_dir)

        self.assertEqual(dataset.low_dir, low_dir)
        self.assertEqual(dataset.high_dir, high_dir)

    def test_explicit_directories_take_precedence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            low_dir = write_image(root / "custom-low" / "a.png").parent
            high_dir = write_image(root / "custom-high" / "a.png").parent

            dataset = CommonDataset(
                root,
                low_dir=low_dir,
                high_dir=high_dir,
            )

        self.assertEqual(dataset.low_dir, low_dir)
        self.assertEqual(dataset.high_dir, high_dir)

    def test_resolves_unpaired_input_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            low_dir = write_image(Path(temp_dir) / "train" / "input" / "a.png").parent

            dataset = CommonDataset(temp_dir)

        self.assertEqual(dataset.low_dir, low_dir)
        self.assertIsNone(dataset.high_dir)
        self.assertEqual(len(dataset), 1)

    def test_missing_root_low_and_high_directories_are_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "root directory"):
            CommonDataset("definitely-missing-root")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "Low-light"):
                CommonDataset(root)

            low_dir = root / "low"
            low_dir.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "Normal-light"):
                CommonDataset(root, low_dir=low_dir, high_dir=root / "missing-high")


class BaseDatasetPairingTests(unittest.TestCase):
    def test_pairs_by_case_insensitive_stem_and_ignores_unsupported_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            low_dir = root / "low"
            high_dir = root / "high"
            write_image(low_dir / "B.PNG")
            write_image(low_dir / "a.jpg")
            write_image(high_dir / "a.png")
            write_image(high_dir / "b.jpg")
            (low_dir / "ignored.txt").write_text("ignored", encoding="utf-8")

            dataset = CommonDataset(root, low_dir=low_dir, high_dir=high_dir)

        self.assertEqual(len(dataset), 2)
        self.assertEqual([pair[0].name for pair in dataset.pairs], ["a.jpg", "B.PNG"])
        self.assertEqual([pair[1].stem.lower() for pair in dataset.pairs], ["a", "b"])

    def test_unmatched_low_image_warns_and_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            low_dir = root / "low"
            high_dir = root / "high"
            write_image(low_dir / "matched.png")
            write_image(low_dir / "unmatched.png")
            write_image(high_dir / "matched.png")

            with self.assertWarnsRegex(UserWarning, "No matching normal-light"):
                dataset = CommonDataset(root, low_dir=low_dir, high_dir=high_dir)

        self.assertEqual(len(dataset), 1)

    def test_duplicate_high_stem_warns_and_keeps_first_sorted_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            low_dir = root / "low"
            high_dir = root / "high"
            write_image(low_dir / "sample.png")
            first = write_image(high_dir / "sample.jpg", value=10)
            write_image(high_dir / "sample.png", value=200)

            with self.assertWarnsRegex(UserWarning, "Duplicate image stem"):
                dataset = CommonDataset(root, low_dir=low_dir, high_dir=high_dir)

        self.assertEqual(dataset.pairs[0][1], first)

    def test_empty_pair_set_strict_mode_raises_and_non_strict_warns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            low_dir = root / "low"
            high_dir = root / "high"
            low_dir.mkdir()
            high_dir.mkdir()

            with self.assertRaisesRegex(RuntimeError, "No image pairs"):
                CommonDataset(root, low_dir=low_dir, high_dir=high_dir)

            with self.assertWarnsRegex(UserWarning, "No image pairs"):
                dataset = CommonDataset(
                    root,
                    low_dir=low_dir,
                    high_dir=high_dir,
                    strict_pairing=False,
                )

        self.assertEqual(len(dataset), 0)

    def test_custom_image_extensions_are_normalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            low_dir = root / "low"
            high_dir = root / "high"
            write_image(low_dir / "sample.foo", image_format="PNG")
            write_image(high_dir / "sample.foo", image_format="PNG")

            dataset = CommonDataset(
                root,
                low_dir=low_dir,
                high_dir=high_dir,
                image_extensions=["FOO"],
            )

        self.assertEqual(dataset.supported_extensions, {".foo"})
        self.assertEqual(len(dataset), 1)

    def test_image_listing_is_non_recursive_and_sorted_case_insensitively(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            write_image(directory / "z.png")
            write_image(directory / "A.png")
            write_image(directory / "nested" / "ignored.png")
            dataset = CommonDataset.__new__(CommonDataset)
            dataset.supported_extensions = {".png"}

            images = dataset._list_images(directory)

        self.assertEqual([path.name for path in images], ["A.png", "z.png"])


class BaseDatasetItemAndTransformTests(unittest.TestCase):
    def test_getitem_returns_tensors_and_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            make_pair_layout(temp_dir)
            dataset = CommonDataset(temp_dir)

            low, high, name = dataset[0]

        self.assertIsInstance(low, torch.Tensor)
        self.assertIsInstance(high, torch.Tensor)
        self.assertEqual(low.shape, (3, 6, 8))
        self.assertEqual(high.shape, (3, 6, 8))
        self.assertEqual(name, "sample.png")
        self.assertLess(low.mean().item(), high.mean().item())

    def test_return_filename_false_and_separate_transforms(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            make_pair_layout(temp_dir)
            dataset = CommonDataset(
                temp_dir,
                return_filename=False,
                transform_low=lambda image: "low-transformed",
                transform_high=lambda image: "high-transformed",
            )

            sample = dataset[0]

        self.assertEqual(sample, ("low-transformed", "high-transformed"))

    def test_unpaired_getitem_returns_none_high_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_image(Path(temp_dir) / "train" / "input" / "sample.png")
            dataset = CommonDataset(temp_dir)

            low, high, name = dataset[0]

        self.assertIsInstance(low, torch.Tensor)
        self.assertIsNone(high)
        self.assertEqual(name, "sample.png")

    def test_common_transform_accepts_two_arguments(self):
        dataset = CommonDataset.__new__(CommonDataset)
        dataset.common_transform = lambda low, high: (low + 1, high + 2)

        result = dataset._apply_common_transform(1, 10)

        self.assertEqual(result, (2, 12))

    def test_common_transform_accepts_pair_tuple(self):
        dataset = CommonDataset.__new__(CommonDataset)

        def tuple_transform(value):
            if not isinstance(value, tuple):
                raise TypeError("tuple required")
            return value[1], value[0]

        dataset.common_transform = tuple_transform

        self.assertEqual(dataset._apply_common_transform("low", "high"), ("high", "low"))

    def test_common_transform_falls_back_to_individual_images(self):
        dataset = CommonDataset.__new__(CommonDataset)

        def individual(value):
            if isinstance(value, tuple):
                raise TypeError("single value required")
            return value + 1

        dataset.common_transform = individual

        self.assertEqual(dataset._apply_common_transform(1, 2), (2, 3))

    def test_common_transform_handles_unpaired_image_and_none_transform(self):
        dataset = CommonDataset.__new__(CommonDataset)
        dataset.common_transform = lambda value: value + 1
        self.assertEqual(dataset._apply_common_transform(1, None), (2, None))

        dataset.common_transform = None
        self.assertEqual(dataset._apply_common_transform(1, 2), (1, 2))

    def test_apply_transform_handles_none_missing_transform_and_callable(self):
        dataset = CommonDataset.__new__(CommonDataset)

        self.assertIsNone(dataset._apply_transform(None, lambda value: value))
        self.assertEqual(dataset._apply_transform(1, None), 1)
        self.assertEqual(dataset._apply_transform(1, lambda value: value + 1), 2)

    def test_stats_include_base_and_common_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            low_dir, high_dir = make_pair_layout(temp_dir, split="Val")
            dataset = CommonDataset(temp_dir, split="validation")

            stats = dataset.get_stats()

        self.assertEqual(stats["dataset"], "CommonDataset")
        self.assertEqual(stats["root_dir"], str(Path(temp_dir)))
        self.assertEqual(stats["split"], "validation")
        self.assertEqual(stats["low_dir"].lower(), str(low_dir).lower())
        self.assertEqual(stats["high_dir"].lower(), str(high_dir).lower())
        self.assertEqual(stats["num_pairs"], 1)
        self.assertIn("val", stats["split_aliases"].lower())


class DatasetPublicAPITests(unittest.TestCase):
    def test_data_packages_export_existing_dataset_classes(self):
        self.assertIs(data_package.BaseDataset, BaseDataset)
        self.assertIs(data_package.CommonDataset, CommonDataset)


if __name__ == "__main__":
    unittest.main()
