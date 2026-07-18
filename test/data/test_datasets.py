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


def make_pair_layout(
    root,
    split="train",
    input_name="sample.png",
    target_name="sample.png",
):
    input_path = write_image(
        Path(root) / split / "input" / input_name,
        value=32,
    )
    target_path = write_image(
        Path(root) / split / "target" / target_name,
        value=192,
    )
    return input_path.parent, target_path.parent


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

            def _resolve_pair_dirs(self, input_dir, target_dir):
                return input_dir or self.root_dir / "input", target_dir

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
            input_dir, target_dir = make_pair_layout(temp_dir, split="train")

            dataset = CommonDataset(temp_dir, split="train")

        self.assertEqual(dataset.input_dir, input_dir)
        self.assertEqual(dataset.target_dir, target_dir)

    def test_resolves_validation_alias_and_case_variant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir, target_dir = make_pair_layout(temp_dir, split="Val")

            dataset = CommonDataset(temp_dir, split="validation")

        self.assertEqual(dataset.input_dir, input_dir)
        self.assertEqual(dataset.target_dir, target_dir)

    def test_resolves_root_level_input_target_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = write_image(Path(temp_dir) / "input" / "a.png").parent
            target_dir = write_image(Path(temp_dir) / "target" / "a.png").parent

            dataset = CommonDataset(temp_dir)

        self.assertEqual(dataset.input_dir, input_dir)
        self.assertEqual(dataset.target_dir, target_dir)

    def test_explicit_directories_take_precedence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = write_image(root / "custom-input" / "a.png").parent
            target_dir = write_image(root / "custom-target" / "a.png").parent

            dataset = CommonDataset(
                root,
                input_dir=input_dir,
                target_dir=target_dir,
            )

        self.assertEqual(dataset.input_dir, input_dir)
        self.assertEqual(dataset.target_dir, target_dir)

    def test_resolves_unpaired_input_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = write_image(
                Path(temp_dir) / "train" / "input" / "a.png"
            ).parent

            dataset = CommonDataset(temp_dir)

        self.assertEqual(dataset.input_dir, input_dir)
        self.assertIsNone(dataset.target_dir)
        self.assertEqual(len(dataset), 1)

    def test_missing_root_input_and_target_directories_are_rejected(self):
        with self.assertRaisesRegex(FileNotFoundError, "root directory"):
            CommonDataset("definitely-missing-root")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "Input image"):
                CommonDataset(root)

            input_dir = root / "input"
            input_dir.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "Target image"):
                CommonDataset(
                    root,
                    input_dir=input_dir,
                    target_dir=root / "missing-target",
                )


class BaseDatasetPairingTests(unittest.TestCase):
    def test_pairs_by_case_insensitive_stem_and_ignores_unsupported_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            target_dir = root / "target"
            write_image(input_dir / "B.PNG")
            write_image(input_dir / "a.jpg")
            write_image(target_dir / "a.png")
            write_image(target_dir / "b.jpg")
            (input_dir / "ignored.txt").write_text(
                "ignored",
                encoding="utf-8",
            )

            dataset = CommonDataset(
                root,
                input_dir=input_dir,
                target_dir=target_dir,
            )

        self.assertEqual(len(dataset), 2)
        self.assertEqual([pair[0].name for pair in dataset.pairs], ["a.jpg", "B.PNG"])
        self.assertEqual([pair[1].stem.lower() for pair in dataset.pairs], ["a", "b"])

    def test_unmatched_input_image_warns_and_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            target_dir = root / "target"
            write_image(input_dir / "matched.png")
            write_image(input_dir / "unmatched.png")
            write_image(target_dir / "matched.png")

            with self.assertWarnsRegex(UserWarning, "No matching target"):
                dataset = CommonDataset(
                    root,
                    input_dir=input_dir,
                    target_dir=target_dir,
                )

        self.assertEqual(len(dataset), 1)

    def test_duplicate_target_stem_warns_and_keeps_first_sorted_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            target_dir = root / "target"
            write_image(input_dir / "sample.png")
            first = write_image(target_dir / "sample.jpg", value=10)
            write_image(target_dir / "sample.png", value=200)

            with self.assertWarnsRegex(UserWarning, "Duplicate image stem"):
                dataset = CommonDataset(
                    root,
                    input_dir=input_dir,
                    target_dir=target_dir,
                )

        self.assertEqual(dataset.pairs[0][1], first)

    def test_empty_pair_set_strict_mode_raises_and_non_strict_warns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            target_dir = root / "target"
            input_dir.mkdir()
            target_dir.mkdir()

            with self.assertRaisesRegex(RuntimeError, "No image pairs"):
                CommonDataset(
                    root,
                    input_dir=input_dir,
                    target_dir=target_dir,
                )

            with self.assertWarnsRegex(UserWarning, "No image pairs"):
                dataset = CommonDataset(
                    root,
                    input_dir=input_dir,
                    target_dir=target_dir,
                    strict_pairing=False,
                )

        self.assertEqual(len(dataset), 0)

    def test_custom_image_extensions_are_normalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            target_dir = root / "target"
            write_image(input_dir / "sample.foo", image_format="PNG")
            write_image(target_dir / "sample.foo", image_format="PNG")

            dataset = CommonDataset(
                root,
                input_dir=input_dir,
                target_dir=target_dir,
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

            input_tensor, target_tensor, name = dataset[0]

        self.assertIsInstance(input_tensor, torch.Tensor)
        self.assertIsInstance(target_tensor, torch.Tensor)
        self.assertEqual(input_tensor.shape, (3, 6, 8))
        self.assertEqual(target_tensor.shape, (3, 6, 8))
        self.assertEqual(name, "sample.png")
        self.assertLess(input_tensor.mean().item(), target_tensor.mean().item())

    def test_return_filename_false_and_separate_transforms(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            make_pair_layout(temp_dir)
            dataset = CommonDataset(
                temp_dir,
                return_filename=False,
                transform_input=lambda image: "input-transformed",
                transform_target=lambda image: "target-transformed",
            )

            sample = dataset[0]

        self.assertEqual(sample, ("input-transformed", "target-transformed"))

    def test_unpaired_getitem_returns_none_target_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_image(Path(temp_dir) / "train" / "input" / "sample.png")
            dataset = CommonDataset(temp_dir)

            input_tensor, target_tensor, name = dataset[0]

        self.assertIsInstance(input_tensor, torch.Tensor)
        self.assertIsNone(target_tensor)
        self.assertEqual(name, "sample.png")

    def test_integer_resize_makes_input_and_target_square_tensors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            make_pair_layout(temp_dir)
            dataset = CommonDataset(temp_dir, resize=512)

            input_tensor, target_tensor, _ = dataset[0]

        self.assertEqual(dataset.resize_size, (512, 512))
        self.assertEqual(input_tensor.shape, (3, 512, 512))
        self.assertEqual(target_tensor.shape, (3, 512, 512))

    def test_pair_resize_uses_height_width_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            make_pair_layout(temp_dir)
            dataset = CommonDataset(temp_dir, resize=(5, 11))

            input_tensor, target_tensor, _ = dataset[0]

        self.assertEqual(dataset.resize_size, (5, 11))
        self.assertEqual(input_tensor.shape, (3, 5, 11))
        self.assertEqual(target_tensor.shape, (3, 5, 11))

    def test_yaml_style_resize_list_is_supported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            make_pair_layout(temp_dir)
            dataset = CommonDataset(temp_dir, resize=[7, 9])

            input_tensor, target_tensor, _ = dataset[0]

        self.assertEqual(input_tensor.shape, (3, 7, 9))
        self.assertEqual(target_tensor.shape, (3, 7, 9))

    def test_resize_is_combined_with_custom_image_transforms(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            make_pair_layout(temp_dir)
            dataset = CommonDataset(
                temp_dir,
                resize=(7, 9),
                transform_input=lambda image: image.size,
                transform_target=lambda image: image.size,
            )

            input_size, target_size, _ = dataset[0]

        self.assertEqual(input_size, (9, 7))
        self.assertEqual(target_size, (9, 7))

    def test_invalid_resize_values_are_rejected(self):
        invalid_values = (
            True,
            0,
            -1,
            (1,),
            (1, 2, 3),
            (1.5, 2),
            (1, 0),
            "512",
        )
        for resize in invalid_values:
            with self.subTest(resize=resize):
                with self.assertRaises((TypeError, ValueError)):
                    BaseDataset.normalize_resize_size(resize)

    def test_common_transform_accepts_two_arguments(self):
        dataset = CommonDataset.__new__(CommonDataset)
        dataset.common_transform = lambda input_value, target_value: (
            input_value + 1,
            target_value + 2,
        )

        result = dataset._apply_common_transform(1, 10)

        self.assertEqual(result, (2, 12))

    def test_common_transform_accepts_pair_tuple(self):
        dataset = CommonDataset.__new__(CommonDataset)

        def tuple_transform(value):
            if not isinstance(value, tuple):
                raise TypeError("tuple required")
            return value[1], value[0]

        dataset.common_transform = tuple_transform

        self.assertEqual(
            dataset._apply_common_transform("input", "target"),
            ("target", "input"),
        )

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
            input_dir, target_dir = make_pair_layout(temp_dir, split="Val")
            dataset = CommonDataset(temp_dir, split="validation")

            stats = dataset.get_stats()

        self.assertEqual(stats["dataset"], "CommonDataset")
        self.assertEqual(stats["root_dir"], str(Path(temp_dir)))
        self.assertEqual(stats["split"], "validation")
        self.assertEqual(stats["input_dir"].lower(), str(input_dir).lower())
        self.assertEqual(stats["target_dir"].lower(), str(target_dir).lower())
        self.assertIsNone(stats["resize"])
        self.assertEqual(stats["num_pairs"], 1)
        self.assertIn("val", stats["split_aliases"].lower())


class DatasetPublicAPITests(unittest.TestCase):
    def test_data_packages_export_existing_dataset_classes(self):
        self.assertIs(data_package.BaseDataset, BaseDataset)
        self.assertIs(data_package.CommonDataset, CommonDataset)


if __name__ == "__main__":
    unittest.main()
