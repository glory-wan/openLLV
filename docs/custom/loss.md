# Custom Loss Functions

Training losses inherit from `BaseLoss` in `openLLV/deepLearning/loss/BaseLoss.py`. The base class registers subclasses by class name, `name`, and `aliases` and exposes a trainer-facing `compute()` method.

## 1. Supervised Loss

Supervised losses keep the normal PyTorch `forward(prediction, target)` signature:

```python
import torch

from openLLV.deepLearning.loss import BaseLoss


class MyLoss(BaseLoss):
    name = "my_loss"
    aliases = ["myloss"]
    requires_target = True

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.weight * torch.mean(torch.abs(prediction - target))
```

The trainer extracts and aligns the prediction before calling the loss.

## 2. Reference-Free or Structured Loss

Set `requires_target = False` and implement `forward(input_tensor, model_output)`:

```python
class MyReferenceFreeLoss(BaseLoss):
    name = "my_reference_free"
    requires_target = False

    def forward(self, input_tensor, model_output):
        pred = (
            model_output["pred"]
            if isinstance(model_output, dict)
            else model_output
        )
        return torch.mean(torch.abs(pred - input_tensor))
```

Model-specific losses may read intermediate values from `model_output["aux"]`.

## 3. Export and Configure

Place task-specific losses in the matching package, for example `openLLV/deepLearning/loss/LLIELoss/MyLoss.py`, and import them from the package `__init__.py`.

```yaml
loss:
  name: my_loss
  params:
    weight: 0.5
  output_index: null
  output_key: null
```

Or use flat training overrides:

```python
result = llv.train(
    "MyModel",
    root_dir="datasets/my_dataset",
    loss="my_loss",
    loss_params={"weight": 0.5},
)
```

Check registration with `llv.list_losses()`. Avoid broad exception handling in `forward()`; shape and contract errors should remain visible during training.

