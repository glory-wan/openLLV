# 自定义深度学习模型

所有深度学习低层视觉模型都直接继承 `LLVModel`。任务分组由类级别的 `task` 值声明，并体现在包目录中；不存在任务专属的中间基类。

## 1. 基类约定

基类定义在 `openLLV/deepLearning/models/BaseModel.py` 中，并提供：

- 按类名和 `aliases` 自动进行不区分大小写的注册；
- 默认配置、显式配置和关键字参数的合并；
- 通过注册名称构建模型；
- 检查点和 YAML 序列化；
- 标准的 `{"pred", "aux", "meta"}` 输出辅助函数。

具体模型必须实现 `_init_model()` 和 `forward()`。

## 2. 最小模型示例

在对应任务包中创建模型，例如 `openLLV/deepLearning/models/LLIE/MyModel.py`：

```python
from typing import Any

import torch
from torch import nn

from openLLV.deepLearning.models import LLVModel


class MyModel(LLVModel):
    task = "llie"
    aliases = ["my_model"]

    def _get_default_config(self):
        config = super()._get_default_config()
        config.update({
            "hidden_channels": 32,
            "mode": "inference",
        })
        return config

    def _validate_config(self):
        super()._validate_config()
        if self.config["hidden_channels"] <= 0:
            raise ValueError("hidden_channels must be positive.")

    def _init_model(self):
        channels = self.config["hidden_channels"]
        self.network = nn.Sequential(
            nn.Conv2d(self.config["input_channels"], channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, self.config["input_channels"], 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, **kwargs: Any):
        pred = self.network(x)
        if self.training:
            return self._format_output(pred, aux={"input": x})
        return pred
```

在任务包的 `__init__.py` 中导入该类，然后再从 `openLLV/deepLearning/models/__init__.py` 导入。模块被导入时会触发注册。

## 3. 配置

`LLVModel.__init__` 按以下顺序合并配置：

1. `_get_default_config()`；
2. `config` 字典；
3. 直接传入的关键字参数。

```python
from openLLV.deepLearning.models import LLVModel

model = LLVModel.create_model(
    "my_model",
    config={"hidden_channels": 64},
)
```

不要在模型配置中保存运行时设备状态。预测器和训练器会调用 `model.to(device)` 并负责设备放置。

## 4. 输出约定

推理时可以直接返回预测张量。结构化训练输出应使用：

```python
return self._format_output(
    pred,
    aux={"feature": feature},
    meta={"scale": 1},
)
```

训练器会为有监督损失提取 `pred`，并为模型专属或无参考损失保留 `aux`。

## 5. 预测和训练

模块导入后即可使用：

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "my_model",
    "input.jpg",
    device="cuda",
)
```

通过扁平覆盖参数进行训练：

```python
result = llv.train(
    model="my_model",
    model_params={"hidden_channels": 64},
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="l1",
    epochs=10,
)
```

使用 `llv.list_models()` 检查注册结果。如果名称与其他模型类或别名冲突，导入时会抛出错误。

