"""Regression tests for SCI initialization and its staged training objective."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from openLLV.deepLearning import Trainer
from openLLV.deepLearning.config import load_config
from openLLV.deepLearning.loss.LLIELoss.Sci_Loss import Sci_Loss
from openLLV.deepLearning.models.LLIE.SCI import SCI


class SCITests(unittest.TestCase):
    def test_shared_weights_are_initialized_once(self):
        calls = Counter()
        original_normal = torch.Tensor.normal_

        def record_normal(tensor, *args, **kwargs):
            calls[tensor.data_ptr()] += 1
            return original_normal(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "normal_", record_normal):
            model = SCI(enhance_layers=2, calibrate_layers=3)

        weights = {
            module.weight.data_ptr()
            for module in model.modules()
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d))
        }
        self.assertEqual(calls, Counter({pointer: 1 for pointer in weights}))
        self.assertIs(model.enhance.blocks[0], model.enhance.blocks[1])
        self.assertIs(model.calibrate.blocks[0], model.calibrate.blocks[2])

    def test_loss_uses_every_stage_input_and_illumination(self):
        # Constant images give zero smoothness, making the loss and gradients
        # independently calculable. Distractor outputs must not affect either.
        stage_inputs = [
            torch.full((1, 3, 8, 8), value, requires_grad=True)
            for value in (0.1, 0.45)
        ]
        illuminations = [
            torch.full((1, 3, 8, 8), value, requires_grad=True)
            for value in (0.3, 0.8)
        ]
        original = torch.full((1, 3, 8, 8), 0.05, requires_grad=True)
        prediction = torch.full_like(original, 0.95, requires_grad=True)
        criterion = Sci_Loss()
        for field in ("aux", "loss_inputs"):
            with self.subTest(field=field), patch.object(
                criterion.smooth_loss, "forward", wraps=criterion.smooth_loss.forward
            ) as smooth:
                loss = criterion(original, {
                    "pred": prediction,
                    field: {
                        "enhanced": prediction,
                        "ilist": illuminations,
                        "inlist": stage_inputs,
                    },
                })
                self.assertAlmostEqual(loss.item(), 1.5 * (0.2 ** 2 + 0.35 ** 2))
                self.assertEqual(smooth.call_count, 2)
                for index, call in enumerate(smooth.call_args_list):
                    self.assertIs(call.args[0], stage_inputs[index])
                    self.assertIs(call.args[1], illuminations[index])

        loss.backward()
        for illumination, stage_input, difference in zip(
            illuminations, stage_inputs, (0.2, 0.35)
        ):
            expected = torch.full_like(illumination, 3 * difference / illumination.numel())
            torch.testing.assert_close(illumination.grad, expected)
            torch.testing.assert_close(stage_input.grad, -expected)
        self.assertIsNone(prediction.grad)
        self.assertIsNone(original.grad)

    def test_incomplete_stage_lists_are_rejected(self):
        tensor = torch.ones(1, 3, 8, 8)
        for stages in (
            {"ilist": [tensor]},
            {"inlist": [tensor]},
            {"ilist": [], "inlist": []},
            {"ilist": [tensor], "inlist": [tensor, tensor]},
        ):
            with self.subTest(stages=list(stages)), self.assertRaises(ValueError):
                Sci_Loss()(tensor, {"pred": tensor, "aux": stages})

    def test_model_outputs_and_calibration_gradients(self):
        torch.manual_seed(2)
        model = SCI().train_mode()
        image = torch.rand(1, 3, 8, 8) * 0.2
        output = model(image)
        self.assertEqual(set(output), {"pred", "aux", "meta"})
        self.assertEqual(set(output["aux"]), {
            "enhanced", "ilist", "rlist", "inlist", "attlist",
        })
        self.assertEqual(len(output["aux"]["ilist"]), 3)
        self.assertIs(output["pred"], output["aux"]["rlist"][-1])
        Sci_Loss()(image, output).backward()
        for network in (model.enhance, model.calibrate):
            gradients = [p.grad for p in network.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
            self.assertGreater(sum(grad.abs().sum().item() for grad in gradients), 0)

        model.eval_mode()
        with torch.no_grad(), patch.object(
            model.calibrate, "forward", side_effect=AssertionError("unused in inference")
        ):
            prediction = model(image)
            expected = torch.clamp(image / model.enhance(image), 0, 1)
        self.assertIsInstance(prediction, torch.Tensor)
        torch.testing.assert_close(prediction, expected)

    def test_yaml_trains_without_a_scheduler(self):
        config = load_config("SCI")
        self.assertNotIn("scheduler", config)
        self.assertEqual(config["train"]["epochs"], 100)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val"):
                folder = root / split / "input"
                folder.mkdir(parents=True)
                Image.new("RGB", (8, 8), (20, 30, 40)).save(folder / "image.png")
            trainer = Trainer(
                "SCI", root_dir=root, output_dir=root / "output",
                device="cpu", epochs=1, workers=0, progress_bar=False,
            )
            self.assertIsNone(trainer.scheduler)
            self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 3e-4)
            self.assertEqual(trainer.optimizer.param_groups[0]["weight_decay"], 3e-4)
            self.assertEqual(trainer.config["train"]["grad_clip"], 5)
            result = trainer.train()
            self.assertEqual(len(result["history"]), 1)
            self.assertEqual(result["history"][0]["lr"], 3e-4)
            self.assertTrue((trainer.checkpoint_dir / "last.pt").is_file())


if __name__ == "__main__":
    unittest.main()
