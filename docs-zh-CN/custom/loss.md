# 自定义损失函数

训练损失函数继承 `openLLV/deepLearning/loss/BaseLoss.py` 中的 `BaseLoss`。基类会根据类名、`name` 和 `aliases` 注册子类，并提供面向训练器的 `compute()` 方法。

## 1. 有监督损失

有监督损失保留常规 PyTorch `forward(prediction, target)` 签名：

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

训练器会在调用损失函数之前提取并对齐预测结果。

## 2. 无参考或结构化损失

设置 `requires_target = False`，并实现 `forward(input_tensor, model_output)`：

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

模型专属损失可以从 `model_output["aux"]` 中读取中间值。

## 3. 导出与配置

将任务专属损失放在匹配的包中，例如 `openLLV/deepLearning/loss/LLIELoss/MyLoss.py`，并从包的 `__init__.py` 中导入。

```yaml
loss:
  name: my_loss
  params:
    weight: 0.5
  output_index: null
  output_key: null
```

也可以使用扁平训练覆盖参数：

```python
result = llv.train(
    "MyModel",
    root_dir="datasets/my_dataset",
    loss="my_loss",
    loss_params={"weight": 0.5},
)
```

使用 `llv.list_losses()` 检查注册结果。不要在 `forward()` 中使用过于宽泛的异常处理；训练期间应让形状错误和接口约定错误保持可见。

