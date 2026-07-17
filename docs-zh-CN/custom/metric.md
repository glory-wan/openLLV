# 自定义评估指标

评估指标继承 `BaseMetric`，并按类名自动注册。名为 `MyMetric` 的类可以通过 `"My"` 或 `"MyMetric"` 选择。

## 1. 全参考指标

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

公共 `compute()` 方法接受 `[C, H, W]` 或 `[B, C, H, W]` 张量，将其移动到指标设备，并把增强图像的空间尺寸与参考图像对齐。

## 2. 无参考指标

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

## 3. 导出与评估

从 `openLLV/evaluation/__init__.py` 或在其中导入的其他模块中导入该类，以触发注册。

```python
import openLLV as llv

print(llv.list_metrics())

results = llv.evaluate(
    en_img_dir="results/my_model",
    metrics=["Brightness"],
    device="cpu",
)
```

传给 `evaluate()` 的指标专属关键字参数会转交给每个被请求的指标，因此在多指标评估中，请避免使用可能相互冲突的参数名。

