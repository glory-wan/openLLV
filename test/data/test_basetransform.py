"""Tests for :mod:`openLLV.data.basetransform`."""

import unittest

import numpy as np
import torch
from PIL import Image

import openLLV.data as data_package
from openLLV.data.basetransform import ToFloat, ToImage, predict_Trans


class PredictTransformTests(unittest.TestCase):
    def test_predict_transform_converts_pil_to_scaled_float_tensor(self):
        image = Image.fromarray(
            np.array([[[0, 127, 255], [255, 64, 0]]], dtype=np.uint8),
            mode="RGB",
        )

        tensor = predict_Trans(image)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 1, 2))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)
        self.assertAlmostEqual(tensor[2, 0, 0].item(), 1.0, places=6)

    def test_predict_transform_accepts_numpy_image(self):
        image = np.full((3, 4, 3), 128, dtype=np.uint8)

        tensor = predict_Trans(image)

        self.assertEqual(tensor.shape, (3, 3, 4))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertAlmostEqual(tensor.mean().item(), 128 / 255, places=6)

    def test_predict_transform_accepts_uint8_tensor(self):
        image = torch.full((3, 2, 2), 255, dtype=torch.uint8)

        tensor = predict_Trans(image)

        self.assertEqual(tensor.dtype, torch.float32)
        torch.testing.assert_close(tensor, torch.ones_like(tensor))

    def test_compatibility_transform_components_are_callable(self):
        self.assertTrue(callable(ToImage))
        self.assertTrue(callable(ToFloat))

    def test_predict_transform_is_exported_from_data_package(self):
        self.assertIs(data_package.predict_Trans, predict_Trans)


if __name__ == "__main__":
    unittest.main()
