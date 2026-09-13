"""Numerical and integration checks for the standard LLFlow adapter."""

import math
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from openLLV.data.datasets import LLFlowDataset
from openLLV.data.datasets.LLFlowDataset import prepare_llflow_input, reflect_pad16
from openLLV.deepLearning.config import load_config
from openLLV.deepLearning.loss.LLIELoss.LLFlow_Loss import LLFlow_Loss
from openLLV.deepLearning.models.LLIE.LLFlow import (
    ActNorm2d, InvertibleConv1x1, LLFlow, squeeze2d,
)
from openLLV.deepLearning.trainer import Trainer
from openLLV.predictor import Predictor


@pytest.fixture(autouse=True, scope="module")
def limited_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def model():
    # Same operations, reduced depth for routine CI. Standard defaults are
    # separately checked below and in the upstream parity check.
    torch.manual_seed(23)
    np.random.seed(23)
    return LLFlow(nb=8, K=1, inference_padding="none", train_gt_ratio=0)


@pytest.mark.parametrize("gt_prior", [False, True])
def test_nll_and_inverse(model, gt_prior):
    low, target = torch.rand(2, 3, 16, 24), torch.rand(2, 3, 16, 24)
    condition = model.rrdbPreprocessing(prepare_llflow_input(low))
    with patch("random.random", return_value=0.0 if gt_prior else 0.5):
        latent, nll, logdet = model.encode(target, condition)
    mean_image = target / (target.sum(1, keepdim=True) + 1e-4) if gt_prior else condition["color_map"]
    gaussian = -0.5 * ((latent - squeeze2d(mean_image, 8)).square() + math.log(2 * math.pi))
    expected = -(gaussian.sum((1, 2, 3)) + logdet) / (math.log(2) * 16 * 24)
    torch.testing.assert_close(nll, expected)
    restored, reverse_logdet = model.flowUpsamplerNet(
        rrdbResults=condition, z=latent, logdet=logdet, reverse=True
    )
    torch.testing.assert_close(restored, target, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(reverse_logdet, torch.zeros_like(logdet), atol=0.005, rtol=0)
    loss = LLFlow_Loss()(model._format_output(restored, {"nll": nll}), target)
    loss.backward()
    for parameter in (model.RRDB.conv_first.weight, model.flowUpsamplerNet.layers[1].invconv.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    # Zero-initialized coupling heads block encoder gradients on the first
    # GT-prior batch; the predicted-prior branch trains the encoder immediately.
    if not gt_prior:
        assert model.RRDB.conv_first.weight.grad.abs().sum() > 0
    assert model.flowUpsamplerNet.layers[1].invconv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("shape", [(1, 1), (16, 32), (17, 23)])
def test_padding_and_preprocessing_match_opencv(shape):
    array = np.random.default_rng(3).integers(0, 256, (*shape, 3), dtype=np.uint8)
    x = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).float() / 255
    padded, (top, bottom, left, right) = reflect_pad16(x)
    expected = cv2.copyMakeBorder(array, top, bottom, left, right, cv2.BORDER_REFLECT)
    torch.testing.assert_close(padded[0], torch.from_numpy(expected.transpose(2, 0, 1)).float() / 255)
    prepared = prepare_llflow_input(padded)
    equalized = cv2.merge([cv2.equalizeHist(c) for c in cv2.split(expected)])
    torch.testing.assert_close(prepared[:, :3], torch.log(padded + 1e-3))
    torch.testing.assert_close(prepared[0, 3:], torch.from_numpy(equalized.transpose(2, 0, 1)).float() / 255)


def test_dataset_equalizes_before_crop(tmp_path):
    low = np.random.default_rng(4).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    for folder in ("low", "high"):
        directory = tmp_path / "our485" / folder
        directory.mkdir(parents=True)
        Image.fromarray(low).save(directory / "a.png")
    dataset = LLFlowDataset(tmp_path, crop_size=16, use_flip=True)
    with patch("numpy.random.randint", side_effect=[2, 5]), patch("numpy.random.choice", return_value=False):
        prepared, target, name = dataset[0]
    expected = cv2.merge([cv2.equalizeHist(c) for c in cv2.split(low)])[2:18, 5:21, :][:, ::-1, :].copy()
    torch.testing.assert_close(prepared[3:], torch.from_numpy(expected.transpose(2, 0, 1)).float() / 255)
    torch.testing.assert_close(prepared[:3], torch.log(target + 1e-3))
    assert name == "a.png"


def test_checkpoint_roundtrip_and_predictor(model, tmp_path):
    x = torch.rand(2, 3, 16, 24)
    model(x, paired_image=x)
    model.eval()
    with torch.no_grad():
        expected = model(x)
    official = tmp_path / "official_G.pth"
    torch.save({"module." + k: v for k, v in model.state_dict().items()}, official)
    restored = LLFlow(nb=8, K=1, inference_padding="none").load_official_weights(official).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x), expected)
    restored.config["inference_padding"] = "reflect16"
    checkpoint = restored.save_model(tmp_path / "converted")
    predictor = Predictor(checkpoint, device="cpu")
    image = Image.fromarray(np.full((19, 27, 3), 40, dtype=np.uint8))
    first, path = predictor(image, save=False)
    second, _ = predictor(image, save=False)
    assert first.size == (27, 19) and path is None
    np.testing.assert_array_equal(first, second)


def test_actnorm_loaded_zero_bias_is_not_reinitialized():
    layer = ActNorm2d(3)
    layer.load_state_dict({"bias": torch.zeros_like(layer.bias), "logs": torch.ones_like(layer.logs)})
    layer(torch.rand(2, 3, 8, 8))
    assert torch.equal(layer.bias, torch.zeros_like(layer.bias))
    assert torch.equal(layer.logs, torch.ones_like(layer.logs))


def test_singular_convolution_fails_without_retry():
    layer = InvertibleConv1x1(3)
    with torch.no_grad():
        layer.weight.zero_()
    with pytest.raises(RuntimeError, match="singular"):
        layer(torch.rand(1, 3, 8, 8))


def test_rejects_legacy_parameters_and_missing_targets(model):
    with pytest.raises(ValueError, match="Removed simplified"):
        LLFlow(flow_layers=8)
    model.config["mode"] = "train"
    with pytest.raises(ValueError, match="paired_image"):
        model(torch.rand(1, 3, 16, 16))
    with pytest.raises(ValueError, match="aux.nll"):
        LLFlow_Loss()({"pred": torch.rand(1, 3, 16, 16)}, torch.rand(1, 3, 16, 16))


def test_standard_configuration():
    config = load_config("LLFlow")
    params = config["model"]["params"]
    assert (params["nf"], params["nb"], params["K"], params["L"]) == (64, 24, 12, 3)
    assert config["data"]["dataset"] == "LLFlowDataset"
    assert config["loss"]["params"] == {}
    assert config["data"]["train_params"] == {"crop_size": 160, "use_flip": True}


def test_unchanged_trainer_train_validate_resume(tmp_path):
    for split in ("our485", "eval15"):
        for kind in ("low", "high"):
            directory = tmp_path / "data" / split / kind
            directory.mkdir(parents=True)
            for i in range(2):
                array = np.random.default_rng(i).integers(1, 255, (16, 16, 3), dtype=np.uint8)
                Image.fromarray(array).save(directory / f"{i}.png")
    config = load_config("LLFlow")
    config["model"]["params"].update(nb=8, K=1, train_gt_ratio=0)
    config["data"].update(root_dir=str(tmp_path / "data"), workers=0, batch_size=2,
                         train_params={"crop_size": 16})
    config["scheduler"] = {"name": None, "params": {}}
    config["train"].update(epochs=1, device="cpu", amp=False, progress_bar=False,
                           output_dir=str(tmp_path / "run"))
    trainer = Trainer(config)
    before = trainer.model.RRDB.conv_first.weight.detach().clone()
    trainer.train()
    assert not torch.equal(before, trainer.model.RRDB.conv_first.weight)
    assert trainer.model.config["mode"] == "inference"
    checkpoint = tmp_path / "run" / "checkpoints" / "last.pt"
    assert checkpoint.exists()
    config["train"].update(resume=str(checkpoint), epochs=2)
    resumed = Trainer(config)
    resumed.train()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_amp_backward(model):
    model = model.cuda()
    x, target = torch.rand(2, 3, 16, 16, device="cuda"), torch.rand(2, 3, 16, 16, device="cuda")
    with torch.autocast("cuda"):
        output = model(x, paired_image=target)
        loss = LLFlow_Loss()(output, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.RRDB.conv_first.weight.grad).all()
