# openLLV 顶层 API 概览

openLLV 为深度学习低层视觉模型、传统增强算法、图像输入输出、训练和图像质量评估提供统一的公共接口。当前内置的深度学习任务为低光图像增强（`llie`）；目录结构已经为增加更多低层视觉任务做好准备，而无须再引入任务专属的中间模型基类。

## 主要 API

| API | 用途 |
| --- | --- |
| `openLLV.predict()` | 运行已注册的模型、检查点或传统算法 |
| `openLLV.enhance()` | `predict()` 的别名 |
| `openLLV.train()` | 使用内置配置、YAML 文件或字典训练 `LLVModel` |
| `openLLV.evaluate()` / `openLLV.eval()` | 评估输出图像目录 |
| `openLLV.imread()` / `openLLV.read_image()` | 读取并转换图像 |
| `openLLV.imwrite()` / `openLLV.write_image()` | 保存图像 |
| `openLLV.list_available()` | 列出已注册的模型、算法、指标、损失函数和数据集 |

## 包结构

| 包 | 职责 |
| --- | --- |
| `openLLV.deepLearning.models` | `LLVModel`、模型注册表以及各任务的具体实现 |
| `openLLV.deepLearning.loss` | `BaseLoss` 和已注册的训练损失函数 |
| `openLLV.deepLearning.config` | 内置 YAML 配置和默认训练参数 |
| `openLLV.tradition.algorithms` | `LLVEnhancer` 和传统算法 |
| `openLLV.data` | 图像输入输出、变换和数据集 |
| `openLLV.evaluation` | 评估指标和目录评估器 |

具体实现的文档采用相同的任务分组方式：

```text
docs/
  models/
    llie/
  algorithms/
    base_methods/
    dehazing/
    llie/
```

## 查看可用组件

```python
import openLLV as llv

components = llv.list_available()
print(components["models"])
print(components["algorithms"])
```

以下扁平化辅助函数会返回所有可接受的注册表键，包括别名：

```python
print(llv.list_models())
print(llv.list_algorithms())
print(llv.list_metrics())
print(llv.list_losses())
print(llv.list_datasets())
```

## 基本工作流程

读取并保存图像：

```python
import openLLV as llv

image = llv.imread("input.jpg", output_format="pil")
saved_path = llv.imwrite(image, "results/copy.png")
```

运行传统算法：

```python
enhanced, saved_path = llv.predict(
    "Gamma",
    "input.jpg",
    output="results/gamma.png",
    gamma=0.6,
)
```

运行深度学习模型：

```python
enhanced, saved_path = llv.predict(
    "ZeroDCE",
    "input.jpg",
    output="results/zero_dce.png",
    device="cuda",
)
```

使用内置配置进行训练：

```python
result = llv.train(
    "ZeroDCE",
    root_dir="datasets/my_dataset",
    epochs=10,
)
```

评估结果目录：

```python
results = llv.evaluate(
    en_img_dir="results/zero_dce",
    ref_img_dir="datasets/my_dataset/test/target",
    metrics=["PSNR", "SSIM"],
)
```

## 后续阅读

完整的公共工作流程请参阅 `predict.md`、`train.md`、`evaluate.md` 和 `image_io.md`。扩展接口说明位于 `docs/custom/`。

