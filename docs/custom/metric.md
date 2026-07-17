# Custom Evaluation Metrics

Evaluation metrics inherit from `BaseMetric` and register automatically by class name. A class named `MyMetric` is selected through `"My"` or `"MyMetric"`.

## 1. Full-Reference Metric

```python
import torch

from openLLV.evaluation import BaseMetric


class RelativeErrorMetric(BaseMetric):
    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = float(eps)

    def _compute_impl(self, enImg, Refer):
        if Refer is None:
            raise ValueError("RelativeError requires a reference image.")
        value = torch.mean(
            torch.abs(enImg - Refer) / (torch.abs(Refer) + self.eps)
        )
        return float(value.item())

    @property
    def higher_is_better(self):
        return False
```

The public `compute()` method accepts `[C, H, W]` or `[B, C, H, W]` tensors, moves them to the metric device, and spatially aligns the enhanced image to the reference.

## 2. No-Reference Metric

```python
class BrightnessMetric(BaseMetric):
    def _compute_impl(self, enImg, Refer=None):
        return float(enImg.mean().item())

    @property
    def requires_reference(self):
        return False

    @property
    def higher_is_better(self):
        return True
```

## 3. Export and Evaluate

Import the class from `openLLV/evaluation/__init__.py` or another module imported there so registration occurs.

```python
import openLLV as llv

print(llv.list_metrics())

results = llv.evaluate(
    en_img_dir="results/my_model",
    metrics=["Brightness"],
    device="cpu",
)
```

Metric-specific keyword arguments passed to `evaluate()` are forwarded to every requested metric, so use parameter names that do not accidentally conflict across a multi-metric run.

