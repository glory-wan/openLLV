# Zero-DCE++

> 任务：低光图像增强（`llie`）

Zero-DCE++ 是 openLLV 已实现的轻量级零参考低光增强模型。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://ieeexplore.ieee.org/document/9369102/ |
| 官方源代码 | https://github.com/Li-Chongyi/Zero-DCE_extension |
| 官方项目页面 | https://li-chongyi.github.io/Proj_Zero-DCE++.html |

## 模型介绍

Zero-DCE++ 是 Zero-DCE 的轻量扩展版本。它的目标是在保留零参考训练优势的同时进一步降低模型复杂度。该模型通过更紧凑的网络结构估计增强曲线，适用于对速度和参数量有严格要求的低光增强应用。

在 openLLV 中，Zero-DCE++ 保持曲线估计模型的接口风格。训练模式下返回增强结果和曲线相关的中间变量；推理模式下返回增强图像。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/ZeroDCEPlusPlus.py` |
| 模型类名 | `ZeroDCEPlusPlus` |
| 默认配置 | `openLLV/deepLearning/config/ZeroDCE++.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/ZeroDCE_Loss.py` |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ZeroDCEPlusPlus",
    "input.jpg",
    output="results/ZeroDCEPlusPlus/output.png",
    device="cuda",
)
```

训练示例：

```python
llv.train(
    model="ZeroDCEPlusPlus",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="zerodce_extension",
    epochs=10,
    batch_size=4,
)
```

