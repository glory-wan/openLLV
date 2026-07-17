"""Run memory-safe forward passes through several LLVModel descendants."""

from _utils import ROOT  # noqa: F401

import torch

from openLLV.deepLearning.models import LLVModel


MODEL_CONFIGS = {
    "ZeroDCE": {"number_f": 8},
    "ZeroDCEPlusPlus": {"number_f": 8, "scale_factor": 1},
    "SCI": {},
    "PairLIE": {"feature_channels": 8},
}


def _flatten_tensors(value):
    """Collect tensors recursively from a structured model output."""
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_flatten_tensors(item))
        return tensors
    if isinstance(value, (tuple, list)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


def main() -> None:
    """Create selected models, place them externally, and run inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.rand(1, 3, 64, 64, device=device)

    print(f"Device: {device}, input shape: {image.shape}")
    for model_name, config in MODEL_CONFIGS.items():
        model = LLVModel.create_model(model_name, config=config)
        model.to(device).eval_mode()

        with torch.no_grad():
            output = model(image)

        shapes = [tuple(tensor.shape) for tensor in _flatten_tensors(output)]
        print(f"{model_name:<20}: output tensors={shapes}")


if __name__ == "__main__":
    main()
