# Gamma

> 文档分组：低光图像增强（LLIE）

Gamma 是 openLLV 已实现的伽马校正增强算法。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文链接 | 无 |
| 官方源代码 | 无 |
| 官方项目页面 | 无 |

## 算法介绍

伽马校正是一种经典的幂律灰度变换方法。它按以下形式调整图像亮度：

```text
output = input ^ gamma
```

图像归一化到 `[0, 1]` 后：

- `gamma < 1` 通常会提亮暗部区域。
- `gamma > 1` 通常会使图像变暗。

当前 openLLV 实现的默认值为 `gamma=0.6`。用户可以根据低光增强需求传入更合适的伽马值。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `openLLV/tradition/algorithms/LLIE/Gamma.py` |
| 算法类名 | `Gamma` |
| 注册名称 | `Gamma` |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## 实现说明

openLLV 的 Gamma 首先把图像转换到浮点数值范围，然后应用幂律变换，最后恢复原始数据类型。

主要参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `gamma` | `float` | `0.6` | 幂律指数，必须为正数 |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "Gamma",
    "input.jpg",
    output="results/gamma/output.jpg",
    gamma=0.6,
)
```

如需通过统一 Predictor 显式使用传统算法后端：

```python
from openLLV import Predictor

predictor = Predictor("Gamma", backend="traditional")
predictor("input.jpg", output="results/gamma/output.jpg", gamma=0.6)
```

