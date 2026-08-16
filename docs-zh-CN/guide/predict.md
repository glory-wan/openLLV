# 预测 API

`openLLV.predict()` 会把请求路由到深度学习后端或传统算法后端。它支持不区分大小写的注册名称、模型检查点，以及现有的 `LLVModel` 或 `LLVEnhancer` 实例。

## 函数形式

```python
openLLV.predict(target, source, output=None, **kwargs)
```

| 参数 | 含义 |
| --- | --- |
| `target` | 模型名称、检查点路径、算法名称或后端实例 |
| `source` | `ImageReader` 支持的图像输入，或图像目录 |
| `output` | 单张图像的可选输出文件，或目录输入对应的输出目录 |
| `backend` | `"auto"`、`"deep"` 或 `"traditional"` |

使用 `backend="auto"` 时，注册名称和后端实例会自动选择对应后端。以 `.pt` 或 `.pth` 结尾的文件会选择深度学习后端。如果将来某个名称同时存在于两个注册表中，请显式指定后端。

## 返回约定

预测单张图像时返回一个二元组：

```python
enhanced_image, saved_path = openLLV.predict(...)
```

深度学习后端返回 RGB PIL 图像，传统算法后端返回 RGB NumPy 数组。`saved_path` 为 `Path`；当 `save=False` 时为 `None`。

对于目录输入，预测器会按确定的源路径顺序递归处理支持的图像，并保留相对目录结构。`save=True` 时返回已保存的 `Path` 列表；`save=False` 时不创建输出文件或目录，并返回增强后的 PIL 图像列表（深度后端）或 RGB NumPy 数组列表（传统后端）。

## 传统算法

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "Gamma",
    "input.jpg",
    output="results/gamma/output.png",
    gamma=0.6,
)
```

算法构造参数可以随顶层调用传入。使用 `Predictor.predict_single()` 时，也可以提供方法专属的单图像覆盖参数。

传统算法接受 `value_range="auto"`（默认）、`"unit"`、`"byte"` 或自定义 `(min, max)` 值域。自动模式会分别保持通常的浮点 `[0,1]` 与浮点 `[0,255]` 约定。若字节值域浮点图的最大值 `<= 1`，其数值内容与单位值域图无法区分，请显式传 `value_range="byte"`。

## 深度学习模型

```python
enhanced, saved_path = llv.predict(
    "ZeroDCE",
    "input.jpg",
    output="results/zero_dce/output.png",
    device="cuda",
)
```

`device` 由预测器管理，不由 `LLVModel` 保存或管理。未指定设备时，优先使用可用的 CUDA，否则使用 CPU。

模型构造参数可以直接通过配置覆盖传入：

```python
enhanced, saved_path = llv.predict(
    "PairLIE",
    "input.jpg",
    config={"enhancement_gamma": 0.14},
    save=False,
)
```

传给模型 forward 调用的参数应放在 `model_kwargs` 中：

```python
enhanced, _ = llv.predict(
    "MyModel",
    "input.jpg",
    save=False,
    model_kwargs={"strength": 0.8},
)
```

## 使用检查点预测

openLLV 训练检查点包含模型类、配置和状态字典：

```python
enhanced, saved_path = llv.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    "input.jpg",
    output="results/from_checkpoint.png",
    device="cpu",
)
```

上游项目的原始状态字典不包含 openLLV 模型元数据。使用这类权重时，请手动创建匹配的模型类并加载权重。

## 目录预测

```python
saved_paths = llv.predict(
    "ZeroDCE",
    "images/",
    output="results/zero_dce",
    output_ext=".PNG",
    progress_bar=True,
)
```

未指定 `output_ext` 时，每个源文件名和扩展名都会逐字符保留，包括字母大小写。显式 `output_ext` 会替换全部后缀并保留参数中的大小写。`output_name` 仅支持单图，目录输入使用它会抛 `ValueError`。

```python
images = llv.predict(
    "Gamma",
    "images/",
    save=False,
    progress_bar=False,
)
```

深度目录预测会按源图尺寸分组。每个包含 `batch_size` 个兼容张量的完整组只执行一次模型调用；不足完整批次的图像逐张运行。流水线不会对图像做 padding。`num_workers` 控制并行图像读取和 CPU 预处理，不控制模型推理。

默认 `resize=None`，不会进行任何缩放，并保持每张源图的原始尺寸。需要主动缩放时，可以传正整数作为正方形尺寸，或显式传入 `(height, width)`：

```python
images = llv.predict(
    "ZeroDCE",
    "images/",
    save=False,
    resize=(384, 512),
    batch_size=4,
    num_workers=2,
)
```

缩放会在自定义 `transform` 之前执行。在使用 spawn 的平台上，`num_workers > 0` 时该变换必须可序列化。

## 统一 Predictor 对象

```python
from openLLV import Predictor

predictor = Predictor(
    "ZeroDCE",
    backend="deep",
    device="cuda",
    output_dir="results/zero_dce",
)

enhanced, saved_path = predictor("input.jpg")
print(predictor.get_params())
```

需要显式选择传统算法后端时，请使用 `backend="traditional"`。可通过 `Predictor.list_available_models()` 和 `Predictor.list_available_methods()` 查看可用的查找名称。
