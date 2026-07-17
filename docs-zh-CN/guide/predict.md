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

深度学习后端返回 PIL 图像，传统算法后端返回 NumPy 数组。`saved_path` 为 `Path`；当 `save=False` 时为 `None`。

对于目录输入，预测器会递归处理支持的图像，并按确定的顺序返回已保存的 `Path` 列表。相对目录结构和源文件扩展名都会保留。

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
    progress_bar=True,
)
```

目录推理会逐张处理图像，以安全支持不同尺寸的输入。深度学习预测器的 `batch_size` 和 `num_workers` 当前是为未来批处理流水线保留的元数据。

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
