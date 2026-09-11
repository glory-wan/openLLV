"""Official standard LLFlow (LOL-pc) adapted to the LLVModel interface.

Upstream: https://github.com/wyf0912/LLFlow
Commit: 115da161a96de868d67494a32db848e31f85bbc1
Copyright (c) 2021 Yufei Wang. CC BY-NC-SA 4.0; see _llflow/licenses/.
The adapter preserves official parameter names and conditional Gaussian NLL.
"""

import math
import random
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

from ..BaseModel import LLVModel
from ._llflow.ConditionEncoder import ConEncoder1
from ._llflow.FlowUpsamplerNet import FlowUpsamplerNet
from ._llflow.flow import GaussianDiag, squeeze2d
from ._llflow.options import standard_options
from openLLV.data.llflow_preprocessing import prepare_llflow_input, reflect_pad16


class LLFlow(LLVModel):
    """Standard 64-channel, 24-RRDB, three-level conditional flow.

    Input is RGB in [0, 1], or the six-channel prepared LLFlowDataset input.
    Trainer supplies paired_image through its existing paired-forward hook.
    """

    task = "llie"
    aliases = []
    requires_paired_forward = True

    def _get_default_config(self):
        config = super()._get_default_config()
        config.update(nf=64, nb=24, K=12, L=3, train_gt_ratio=0.2,
                      quant=32, inference_padding="reflect16", mode="inference")
        return config

    def _validate_config(self):
        super()._validate_config()
        removed = {"condition_channels", "condition_blocks", "flow_layers",
                   "flow_hidden_channels", "scale_clamp", "sample_temperature"}
        legacy = removed.intersection(self.config)
        if legacy:
            raise ValueError("Removed simplified LLFlow configuration: " + ", ".join(sorted(legacy)))
        if self.config["input_channels"] != 3 or self.config["nf"] != 64 or self.config["L"] != 3:
            raise ValueError("Standard LLFlow requires input_channels=3, nf=64 and L=3.")
        for key, minimum in (("nb", 8), ("K", 1)):
            value = self.config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}.")
        ratio = float(self.config["train_gt_ratio"])
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("train_gt_ratio must be in [0, 1].")
        if not math.isfinite(float(self.config["quant"])) or float(self.config["quant"]) <= 0:
            raise ValueError("quant must be positive and finite.")
        if self.config["inference_padding"] not in {"reflect16", "none"}:
            raise ValueError("inference_padding must be 'reflect16' or 'none'.")
        if self.config["mode"] not in {"train", "inference"}:
            raise ValueError("mode must be 'train' or 'inference'.")

    def _init_model(self):
        self.opt = standard_options(self.config)
        self.RRDB = ConEncoder1(3, 3, 64, self.config["nb"], 32, 1, self.opt)
        self.flowUpsamplerNet = FlowUpsamplerNet(
            (160, 160, 3), 64, self.config["K"],
            flow_coupling="CondAffineSeparatedAndCond", opt=self.opt,
        )

    def rrdbPreprocessing(self, low):
        """Extract and concatenate the official multi-scale RRDB features."""
        features = self.RRDB(low, get_steps=True)
        concat = torch.cat([features[f"block_{i}"] for i in (1, 3, 5, 7)], dim=1)
        for key in ("last_lr_fea", "fea_up1", "fea_up2", "fea_up4", "fea_up0"):
            features[key] = torch.cat([
                features[key], F.interpolate(concat, features[key].shape[-2:])
            ], dim=1)
        return features

    def encode(self, target, condition, *, add_gt_noise=False):
        """Return (latent, per-sample NLL, logdet) with the official prior.

        condition is the dictionary returned by rrdbPreprocessing. The GT-prior
        choice is one Python random draw per batch, as in upstream normal_flow.
        """
        pixels = target.shape[-2] * target.shape[-1]
        logdet = torch.zeros_like(target[:, 0, 0, 0])
        value = target
        if add_gt_noise:
            quant = float(self.config["quant"])
            value = value + (torch.rand_like(value) - 0.5) / quant
            logdet = logdet - math.log(quant) * pixels
        latent, logdet = self.flowUpsamplerNet(
            rrdbResults=condition, gt=value, logdet=logdet, reverse=False,
        )
        if random.random() > float(self.config["train_gt_ratio"]):
            mean = squeeze2d(condition["color_map"], 8)
        else:
            mean = squeeze2d(target / (target.sum(dim=1, keepdim=True) + 1e-4), 8)
        objective = logdet.clone() + GaussianDiag.logp(mean, latent.new_tensor(0.0), latent)
        return latent, -objective / (math.log(2.0) * pixels), logdet

    def decode(self, condition):
        """Decode the predicted color map, not a zero/random latent tensor."""
        latent = squeeze2d(condition["color_map"], 8)
        image, _ = self.flowUpsamplerNet(
            rrdbResults=condition, z=latent, logdet=latent.new_zeros(latent.shape[0]),
            reverse=True, eps_std=0,
        )
        return image

    def forward(self, x, paired_image=None, *, add_gt_noise=False):
        """Enhance RGB, or encode paired GT for Trainer's NLL loss.

        Paired training/validation requires matching spatial dimensions divisible
        by eight. Inference reflect16 padding follows official test_unpaired.py;
        set inference_padding='none' for the unpadded core computation.
        """
        if x.ndim != 4 or x.shape[1] not in (3, 6) or not x.is_floating_point():
            raise ValueError("LLFlow expects floating BCHW RGB or prepared six-channel input.")
        if min(x.shape[-2:]) < 1:
            raise ValueError("Input spatial dimensions must be positive.")
        original_h, original_w = x.shape[-2:]
        top = left = 0
        if paired_image is not None:
            if paired_image.shape != (x.shape[0], 3, original_h, original_w):
                raise ValueError("Paired GT must be RGB and match the input batch and spatial size.")
        elif self.config["mode"] == "train":
            raise ValueError("LLFlow training requires paired_image (normal-light GT).")
        elif self.config["inference_padding"] == "reflect16":
            x, (top, _, left, _) = reflect_pad16(x)
        if x.shape[-2] % 8 or x.shape[-1] % 8:
            raise ValueError("LLFlow core requires height and width divisible by 8.")
        low = prepare_llflow_input(x) if x.shape[1] == 3 else x
        condition = self.rrdbPreprocessing(low)
        if paired_image is not None:
            latent, nll, logdet = self.encode(paired_image, condition, add_gt_noise=add_gt_noise)
            # Encode first so ActNorm initializes from GT, never generated images.
            # No reverse loss is added; this only preserves the image-output contract.
            with torch.no_grad():
                prediction = self.decode(condition)
            return self._format_output(
                prediction, aux={"nll": nll, "latent": latent, "logdet": logdet},
                meta={"mode": self.config["mode"]},
            )
        prediction = self.decode(condition)
        return prediction[:, :, top:top + original_h, left:left + original_w]

    def load_official_weights(self, path):
        """Strictly load a local official *_G.pth state dictionary; return self.

        A leading DataParallel 'module.' is removed. Use save_model afterwards
        to create a checkpoint accepted by the unchanged Predictor.
        """
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError("Expected a pure official LLFlow state dictionary.")
        state = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
        self.load_state_dict(state, strict=True)
        return self
