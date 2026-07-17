"""Tests for :mod:`openLLV.data.coreDataset`."""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torchvision import transforms

import openLLV.data as data_package
from openLLV.data.coreDataset import EvaluateDataset, PredictDataSet, single_data_loader


def write_image(path, value=128, size=(8, 6), image_format=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (value, value, value))
    image.save(path, format=image_format)
    return path


@contextmanager
def silent_discovery():
    with patch(
        "openLLV.data.utils.tqdm",
        side_effect=lambda iterable, **kwargs: iterable,
    ):
        yield


class SingleDataLoaderTests(unittest.TestCase):
    def test_default_transform_returns_batched_float_tensor_and_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_image(Path(temp_dir) / "sample.png", value=255)

            tensor, name = single_data_loader(str(path))

        self.assertEqual(tensor.shape, (1, 3, 6, 8))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.max().item(), 1.0)
        self.assertEqual(name, "sample.png")

    def test_none_transform_falls_back_to_prediction_transform(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_image(Path(temp_dir) / "sample.png", value=64)

            tensor, _ = single_data_loader(path, transform=None)

        self.assertEqual(tensor.shape, (1, 3, 6, 8))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_compose_callable_and_list_transforms_are_supported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_image(Path(temp_dir) / "sample.png")
            compose = transforms.Compose([transforms.ToTensor()])
            callable_transform = lambda image: torch.tensor([2.0])
            transform_list = [lambda image: torch.tensor([3.0])]

            composed, _ = single_data_loader(path, transform=compose)
            called, _ = single_data_loader(path, transform=callable_transform)
            listed, _ = single_data_loader(path, transform=transform_list)

        self.assertEqual(composed.shape, (1, 3, 6, 8))
        torch.testing.assert_close(called, torch.tensor([[2.0]]))
        torch.testing.assert_close(listed, torch.tensor([[3.0]]))

    def test_invalid_transform_type_uses_default_transform(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_image(Path(temp_dir) / "sample.png")

            tensor, _ = single_data_loader(path, transform=object())

        self.assertEqual(tensor.shape, (1, 3, 6, 8))
        self.assertEqual(tensor.dtype, torch.float32)


class PredictDatasetTests(unittest.TestCase):
    def test_dataset_discovers_images_recursively_and_returns_sorted_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_image(root / "z.png", value=20)
            write_image(root / "nested" / "a.jpg", value=40)
            (root / "ignored.txt").write_text("not an image", encoding="utf-8")

            with silent_discovery():
                dataset = PredictDataSet(str(root))
                samples = [dataset[index] for index in range(len(dataset))]

        self.assertEqual(len(samples), 2)
        self.assertEqual([sample[1] for sample in samples], ["a.jpg", "z.png"])
        for tensor, _ in samples:
            self.assertEqual(tensor.shape[0], 3)
            self.assertEqual(tensor.dtype, torch.float32)

    def test_none_transform_uses_to_tensor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_image(root / "sample.png", value=255)

            with silent_discovery():
                dataset = PredictDataSet(str(root), transform=None)
                tensor, name = dataset[0]

        self.assertEqual(name, "sample.png")
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(tensor.max().item(), 1.0)

    def test_custom_transform_is_used(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_image(root / "sample.png")

            with silent_discovery():
                dataset = PredictDataSet(
                    str(root),
                    transform=lambda image: torch.tensor([9.0]),
                )
                tensor, _ = dataset[0]

        torch.testing.assert_close(tensor, torch.tensor([9.0]))

    def test_empty_directory_has_zero_length(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with silent_discovery():
                dataset = PredictDataSet(temp_dir)

        self.assertEqual(len(dataset), 0)


class EvaluateDatasetTests(unittest.TestCase):
    def test_no_reference_mode_returns_all_enhanced_images_and_none_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            enhanced = Path(temp_dir) / "enhanced"
            write_image(enhanced / "a.png", value=10)
            write_image(enhanced / "b.png", value=20)

            with silent_discovery():
                dataset = EvaluateDataset(str(enhanced))
                sample = dataset[0]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.ref_dict, {})
        self.assertIsNone(sample[1])
        self.assertEqual(sample[0].shape, (3, 6, 8))
        self.assertIn(sample[2], {"a.png", "b.png"})

    def test_pairs_images_case_insensitively_by_stem_across_extensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            enhanced = root / "enhanced"
            reference = root / "reference"
            write_image(enhanced / "Sample.PNG", value=20)
            write_image(reference / "sample.jpg", value=80)

            with silent_discovery():
                dataset = EvaluateDataset(str(enhanced), str(reference))
                en_tensor, ref_tensor, name = dataset[0]

        self.assertEqual(len(dataset), 1)
        self.assertEqual(name, "Sample.PNG")
        self.assertIsNotNone(ref_tensor)
        self.assertLess(en_tensor.mean().item(), ref_tensor.mean().item())
        self.assertTrue(dataset._find_matching_ref("SAMPLE.bmp").endswith("sample.jpg"))

    def test_unmatched_enhanced_images_are_warned_and_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            enhanced = root / "enhanced"
            reference = root / "reference"
            write_image(enhanced / "matched.png")
            write_image(enhanced / "unmatched.png")
            write_image(reference / "matched.png")

            with silent_discovery():
                with self.assertWarnsRegex(UserWarning, "No matching reference"):
                    dataset = EvaluateDataset(str(enhanced), str(reference))

        self.assertEqual(len(dataset), 1)
        self.assertEqual(Path(dataset.paired_files[0]).name, "matched.png")

    def test_missing_reference_directory_warns_and_enables_no_ref_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            enhanced = Path(temp_dir) / "enhanced"
            write_image(enhanced / "sample.png")

            with silent_discovery():
                with self.assertWarnsRegex(UserWarning, "does not exist"):
                    dataset = EvaluateDataset(
                        str(enhanced),
                        str(Path(temp_dir) / "missing-reference"),
                    )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.ref_dict, {})

    def test_none_directories_create_empty_dataset(self):
        dataset = EvaluateDataset()

        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.en_files, [])
        self.assertEqual(dataset.paired_files, [])

    def test_reference_dictionary_uses_lowercase_stems(self):
        dataset = EvaluateDataset.__new__(EvaluateDataset)
        dataset.ref_files = ["folder/One.PNG", "folder/TWO.jpg"]

        mapping = dataset._create_ref_dict()

        self.assertEqual(mapping["one"], "folder/One.PNG")
        self.assertEqual(mapping["two"], "folder/TWO.jpg")

    def test_corrupt_reference_warns_and_returns_none(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            enhanced = root / "enhanced"
            reference = root / "reference"
            write_image(enhanced / "sample.png")
            reference.mkdir()
            (reference / "sample.png").write_bytes(b"corrupt image")

            with silent_discovery():
                dataset = EvaluateDataset(str(enhanced), str(reference))
                with self.assertWarnsRegex(UserWarning, "Failed to read reference"):
                    _, ref_tensor, name = dataset[0]

        self.assertEqual(name, "sample.png")
        self.assertIsNone(ref_tensor)


class CoreDatasetPublicAPITests(unittest.TestCase):
    def test_data_package_exports_core_dataset_api(self):
        for name in ("single_data_loader", "PredictDataSet", "EvaluateDataset"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(data_package, name))


if __name__ == "__main__":
    unittest.main()
