"""Tests for :mod:`openLLV.deepLearning.predictor`."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import openLLV.deepLearning as deep_learning
from openLLV.deepLearning.models import LLVModel, ZeroDCE
from openLLV.deepLearning.predictor import Predictor


class PredictorIdentityModel(LLVModel):
    """Small registered model used to test the generic LLVModel contract."""

    task = "predictor-test"
    aliases = ["predictor-identity"]

    def _get_default_config(self):
        config = super()._get_default_config()
        config.update({"style": "dict", "marker": "default"})
        return config

    def _init_model(self):
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.last_kwargs = None

    def forward(self, x, offset=None, nested=None, **kwargs):
        self.last_kwargs = {"offset": offset, "nested": nested, **kwargs}
        prediction = x * self.scale
        if offset is not None:
            prediction = prediction + offset

        style = self.config["style"]
        if style == "tensor":
            return prediction
        if style == "dict":
            return self._format_output(prediction, meta={"style": style})
        if style == "nested":
            return {"aux": {"prediction": prediction}}
        if style == "tuple":
            return ("metadata", [prediction])
        if style == "empty_dict":
            return {"meta": "no tensor"}
        if style == "invalid":
            return "not a tensor"
        if style == "two_channels":
            return prediction[:, :2]
        if style == "two_batches":
            return torch.cat([prediction, prediction], dim=0)
        return prediction


class SilentProgress:
    def __init__(self, iterable, **kwargs):
        self.iterable = iterable
        self.postfixes = []

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, value):
        self.postfixes.append(value)


def sample_rgb(value=128, size=(8, 6)):
    return np.full((size[1], size[0], 3), value, dtype=np.uint8)


def write_image(path, value=128):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sample_rgb(value=value)).save(path)
    return path


class PredictorInitializationTests(unittest.TestCase):
    def test_accepts_any_llvmodel_instance_and_owns_device(self):
        model = PredictorIdentityModel()

        predictor = Predictor(model, device="cpu")

        self.assertIs(predictor.model, model)
        self.assertEqual(predictor.device, torch.device("cpu"))
        self.assertFalse(model.training)
        self.assertEqual(model.config["mode"], "inference")
        self.assertNotIn("device", model.config)
        self.assertEqual(model.scale.device.type, "cpu")

    def test_instance_config_overrides_are_applied_and_validated(self):
        model = PredictorIdentityModel()

        predictor = Predictor(
            model,
            config={"marker": "overridden", "input_channels": 1},
            device="cpu",
        )

        self.assertEqual(predictor.model.config["marker"], "overridden")
        self.assertEqual(predictor.model.config["input_channels"], 1)

    def test_registered_name_and_alias_are_case_insensitive(self):
        by_name = Predictor("PREDICTORIDENTITYMODEL", device="cpu")
        by_alias = Predictor("Predictor-Identity", device="cpu")

        self.assertIsInstance(by_name.model, PredictorIdentityModel)
        self.assertIsInstance(by_alias.model, PredictorIdentityModel)

    def test_default_and_explicit_output_directories(self):
        default = Predictor(PredictorIdentityModel(), device="cpu")
        explicit = Predictor(
            PredictorIdentityModel(),
            output_dir="custom-output",
            device="cpu",
        )

        self.assertEqual(
            default.output_dir,
            Path("results") / "PredictorIdentityModel",
        )
        self.assertEqual(explicit.output_dir, Path("custom-output"))

    def test_invalid_model_config_and_runtime_metadata_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "model must be"):
            Predictor(object(), device="cpu")
        with self.assertRaisesRegex(TypeError, "config must be"):
            Predictor(PredictorIdentityModel(), config="bad", device="cpu")
        with self.assertRaisesRegex(ValueError, "batch_size"):
            Predictor(PredictorIdentityModel(), batch_size=0, device="cpu")
        with self.assertRaisesRegex(ValueError, "num_workers"):
            Predictor(PredictorIdentityModel(), num_workers=-1, device="cpu")

    def test_unknown_model_and_missing_checkpoint_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not registered"):
            Predictor("missing-model", device="cpu")
        with self.assertRaisesRegex(FileNotFoundError, "Checkpoint"):
            Predictor("missing-model.pt", device="cpu")

    def test_checkpoint_created_by_llvmodel_can_be_loaded(self):
        model = PredictorIdentityModel(config={"marker": "saved"})
        model.scale.data.fill_(0.5)

        with tempfile.TemporaryDirectory() as temp_dir:
            with redirect_stdout(io.StringIO()):
                checkpoint = model.save_model(temp_dir)
                predictor = Predictor(
                    checkpoint,
                    config={"marker": "loaded"},
                    device="cpu",
                )

        self.assertIsInstance(predictor.model, PredictorIdentityModel)
        self.assertAlmostEqual(predictor.model.scale.item(), 0.5)
        self.assertEqual(predictor.model.config["marker"], "loaded")
        self.assertFalse(predictor.model.training)

    def test_get_params_reports_model_task_runtime_and_config(self):
        predictor = Predictor(
            PredictorIdentityModel(config={"marker": "value"}),
            output_dir="outputs",
            device="cpu",
            batch_size=2,
            num_workers=3,
        )

        params = predictor.get_params()

        self.assertEqual(params["model"], "PredictorIdentityModel")
        self.assertEqual(params["task"], "predictor-test")
        self.assertEqual(params["device"], "cpu")
        self.assertEqual(params["output_dir"], "outputs")
        self.assertEqual(params["batch_size"], 2)
        self.assertEqual(params["num_workers"], 3)
        self.assertEqual(params["config"]["marker"], "value")

    def test_global_registry_and_deep_learning_package_exports(self):
        available = Predictor.list_available_models()

        self.assertIn("predictoridentitymodel", available)
        self.assertIn("predictor-identity", available)
        self.assertIn("zerodce", available)
        self.assertIs(deep_learning.Predictor, Predictor)
        self.assertIs(deep_learning.LLVModel, LLVModel)


class PredictorInputAndInferenceTests(unittest.TestCase):
    def test_predict_single_accepts_numpy_pil_path_and_tensor_without_saving(self):
        predictor = Predictor(PredictorIdentityModel(), device="cpu")
        pil = Image.fromarray(sample_rgb(64))
        tensor = torch.full((3, 6, 8), 64 / 255)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = write_image(Path(temp_dir) / "sample.png", 64)
            inputs = [sample_rgb(64), pil, path, tensor]

            for image_input in inputs:
                with self.subTest(input_type=type(image_input).__name__):
                    output, saved = predictor.predict_single(image_input, save=False)
                    self.assertIsInstance(output, Image.Image)
                    self.assertEqual(output.mode, "RGB")
                    self.assertEqual(output.size, (8, 6))
                    self.assertIsNone(saved)

    def test_predict_and_call_aliases_route_single_image(self):
        predictor = Predictor(PredictorIdentityModel(), device="cpu")

        via_predict, _ = predictor.predict(sample_rgb(100), save=False)
        via_call, _ = predictor(sample_rgb(100), save=False)

        np.testing.assert_array_equal(np.asarray(via_predict), np.asarray(via_call))

    def test_standard_dict_tensor_nested_and_tuple_outputs_are_supported(self):
        for style in ("dict", "tensor", "nested", "tuple"):
            with self.subTest(style=style):
                predictor = Predictor(
                    PredictorIdentityModel(config={"style": style}),
                    device="cpu",
                )
                output, _ = predictor.predict_single(sample_rgb(), save=False)
                self.assertEqual(output.size, (8, 6))

    def test_model_kwargs_are_forwarded_and_nested_tensors_move_to_device(self):
        model = PredictorIdentityModel()
        predictor = Predictor(model, device="cpu")
        offset = torch.full((1, 3, 1, 1), 0.1)
        nested = {"items": [torch.tensor(2.0), (torch.tensor(3.0),)]}

        output, _ = predictor.predict_single(
            sample_rgb(0),
            save=False,
            model_kwargs={"offset": offset, "nested": nested, "flag": True},
        )

        self.assertGreater(np.asarray(output).mean(), 0)
        self.assertEqual(model.last_kwargs["offset"].device.type, "cpu")
        self.assertEqual(
            model.last_kwargs["nested"]["items"][0].device.type,
            "cpu",
        )
        self.assertTrue(model.last_kwargs["flag"])

    def test_model_kwargs_must_be_mapping(self):
        predictor = Predictor(PredictorIdentityModel(), device="cpu")

        with self.assertRaisesRegex(TypeError, "model_kwargs"):
            predictor.predict_single(
                sample_rgb(),
                save=False,
                model_kwargs=[("offset", 1)],
            )

    def test_transform_override_and_transform_list_are_supported(self):
        predictor = Predictor(PredictorIdentityModel(), device="cpu")
        transform = lambda image: torch.zeros(3, image.height, image.width)

        direct, _ = predictor.predict_single(
            sample_rgb(255),
            save=False,
            transform=transform,
        )
        listed = Predictor(
            PredictorIdentityModel(),
            device="cpu",
            transform=[transform],
        )
        from_list, _ = listed.predict_single(sample_rgb(255), save=False)

        self.assertEqual(np.asarray(direct).max(), 0)
        self.assertEqual(np.asarray(from_list).max(), 0)

    def test_invalid_transforms_and_shapes_have_clear_errors(self):
        with self.assertRaisesRegex(TypeError, "transform must"):
            Predictor(PredictorIdentityModel(), transform=object(), device="cpu")
        with self.assertRaisesRegex(TypeError, "Every item"):
            Predictor(PredictorIdentityModel(), transform=[object()], device="cpu")

        predictor = Predictor(PredictorIdentityModel(), device="cpu")
        with self.assertRaisesRegex(TypeError, "return a torch.Tensor"):
            predictor.predict_single(
                sample_rgb(),
                save=False,
                transform=lambda image: image,
            )
        with self.assertRaisesRegex(ValueError, "Expected transformed image"):
            predictor.predict_single(
                sample_rgb(),
                save=False,
                transform=lambda image: torch.zeros(3, 4),
            )
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            predictor.predict_single(
                sample_rgb(),
                save=False,
                transform=lambda image: torch.zeros(2, 3, 4, 4),
            )

    def test_invalid_model_outputs_have_clear_errors(self):
        for style, error_type, pattern in (
            ("empty_dict", KeyError, "does not contain"),
            ("invalid", TypeError, "Cannot extract"),
            ("two_channels", ValueError, "1, 3, or 4"),
            ("two_batches", ValueError, "output batch size 1"),
        ):
            with self.subTest(style=style):
                predictor = Predictor(
                    PredictorIdentityModel(config={"style": style}),
                    device="cpu",
                )
                with self.assertRaisesRegex(error_type, pattern):
                    predictor.predict_single(sample_rgb(), save=False)

    def test_real_llie_descendant_is_supported(self):
        predictor = Predictor(ZeroDCE(), device="cpu")

        output, saved = predictor.predict_single(sample_rgb(32), save=False)

        self.assertEqual(output.mode, "RGB")
        self.assertEqual(output.size, (8, 6))
        self.assertIsNone(saved)


class PredictorSavingAndBatchTests(unittest.TestCase):
    def test_single_prediction_saves_to_explicit_file(self):
        predictor = Predictor(PredictorIdentityModel(), device="cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "nested" / "prediction.png"
            output, saved = predictor.predict_single(
                sample_rgb(80),
                save_path=target,
            )

            self.assertEqual(saved, target)
            self.assertTrue(target.exists())
            with Image.open(target) as loaded:
                self.assertEqual(loaded.size, output.size)

    def test_single_prediction_resolves_directory_name_and_extension(self):
        predictor = Predictor(PredictorIdentityModel(), device="cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = write_image(root / "source.jpg", 90)
            output_dir = root / "outputs"
            _, saved = predictor.predict_single(
                source,
                save_path=output_dir,
                output_name="renamed.jpg",
                output_ext="png",
            )

        self.assertEqual(saved, output_dir / "renamed.png")

    def test_output_extension_overrides_explicit_file_suffix(self):
        predictor = Predictor(PredictorIdentityModel(), device="cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            _, saved = predictor.predict_single(
                sample_rgb(),
                save_path=Path(temp_dir) / "output.jpg",
                output_ext="png",
            )

        self.assertEqual(saved.suffix, ".png")

    def test_default_output_path_and_non_path_source_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            predictor = Predictor(
                PredictorIdentityModel(),
                output_dir=temp_dir,
                device="cpu",
            )
            _, saved = predictor.predict_single(sample_rgb(), output_ext="png")

            self.assertEqual(saved, Path(temp_dir) / "image.png")
            self.assertTrue(saved.exists())

    def test_batch_prediction_is_recursive_and_preserves_relative_paths(self):
        predictor = Predictor(PredictorIdentityModel(), device="cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            write_image(input_dir / "a.png", 10)
            write_image(input_dir / "nested" / "b.jpg", 20)
            (input_dir / "ignored.txt").write_text("ignored", encoding="utf-8")

            saved = predictor.predict_batch(
                input_dir,
                output_dir,
                progress_bar=False,
            )

            self.assertEqual(
                saved,
                [output_dir / "a.png", output_dir / "nested" / "b.jpg"],
            )
            self.assertTrue(all(path.exists() for path in saved))

    def test_call_routes_directory_and_progress_bar_is_supported(self):
        predictor = Predictor(PredictorIdentityModel(), device="cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            write_image(input_dir / "a.png")

            with patch("openLLV.deepLearning.predictor.tqdm", SilentProgress):
                saved = predictor(input_dir, output=output_dir)

        self.assertEqual(saved, [output_dir / "a.png"])

    def test_batch_prediction_handles_empty_and_invalid_directory(self):
        predictor = Predictor(PredictorIdentityModel(), device="cpu")

        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                predictor.predict_batch(temp_dir, progress_bar=False),
                [],
            )
            with self.assertRaisesRegex(NotADirectoryError, "must be a directory"):
                predictor.predict_batch(
                    Path(temp_dir) / "missing",
                    progress_bar=False,
                )


class PredictorHelperTests(unittest.TestCase):
    def test_prediction_extraction_and_tensor_flattening(self):
        first = torch.tensor(1.0)
        second = torch.tensor(2.0)

        self.assertIs(Predictor._extract_prediction(first), first)
        self.assertIs(Predictor._extract_prediction({"pred": first}), first)
        self.assertIs(
            Predictor._extract_prediction({"aux": [first, {"x": second}]}),
            second,
        )
        self.assertEqual(
            Predictor._flatten_tensors({"a": [first], "b": (second,)}),
            [first, second],
        )

        with self.assertRaises(KeyError):
            Predictor._extract_prediction({"meta": "empty"})
        with self.assertRaises(TypeError):
            Predictor._extract_prediction("invalid")

    def test_image_listing_directory_detection_and_name_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_image(root / "z.PNG")
            write_image(root / "nested" / "a.jpg")
            (root / "ignored.txt").write_text("ignored", encoding="utf-8")

            files = Predictor._list_images(root)

            self.assertEqual(
                files,
                [root / "nested" / "a.jpg", root / "z.PNG"],
            )
            self.assertTrue(Predictor._is_directory_source(root))
            self.assertTrue(Predictor._is_directory_source(str(root)))
            self.assertFalse(Predictor._is_directory_source(sample_rgb()))

        self.assertEqual(
            Predictor._infer_source_name("https://example.com/a%20b.png"),
            "a b.png",
        )
        self.assertEqual(Predictor._infer_source_name(sample_rgb()), "image.jpg")

    def test_suffix_and_file_path_helpers(self):
        self.assertEqual(Predictor._normalize_suffix("png"), ".png")
        self.assertEqual(Predictor._normalize_suffix(".jpg"), ".jpg")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            Predictor._normalize_suffix("  ")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = root / "directory.with.dot"
            directory.mkdir()
            file_path = root / "existing"
            file_path.write_bytes(b"")

            self.assertFalse(Predictor._looks_like_file_path(directory))
            self.assertTrue(Predictor._looks_like_file_path(file_path))
            self.assertTrue(Predictor._looks_like_file_path(root / "new.png"))
            self.assertFalse(Predictor._looks_like_file_path(root / "new-dir"))

    def test_tensor_to_pil_supports_gray_rgb_rgba_and_clamping(self):
        gray = Predictor._tensor_to_pil(torch.tensor([[[[-1.0, 2.0]]]]))
        rgb = Predictor._tensor_to_pil(torch.ones(3, 2, 2))
        rgba = Predictor._tensor_to_pil(torch.ones(1, 4, 2, 2))

        self.assertEqual(gray.mode, "L")
        np.testing.assert_array_equal(np.asarray(gray), np.array([[0, 255]], dtype=np.uint8))
        self.assertEqual(rgb.mode, "RGB")
        self.assertEqual(rgba.mode, "RGBA")

        with self.assertRaisesRegex(ValueError, "1, 3, or 4"):
            Predictor._tensor_to_pil(torch.zeros(2, 2, 2))
        with self.assertRaisesRegex(ValueError, "output batch size 1"):
            Predictor._tensor_to_pil(torch.zeros(2, 3, 2, 2))
        with self.assertRaisesRegex(ValueError, "Expected prediction tensor"):
            Predictor._tensor_to_pil(torch.zeros(2, 2))

    def test_rgba_output_can_be_saved_as_jpeg(self):
        image = Image.new("RGBA", (2, 2), (255, 0, 0, 128))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "output.jpg"
            Predictor._save_pil_image(image, path)

            with Image.open(path) as saved:
                self.assertEqual(saved.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
