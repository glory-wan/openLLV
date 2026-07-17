# 训练 API

`openLLV.train()` 会创建由配置驱动的 `Trainer` 并运行完整训练循环。训练器负责运行时设备管理，并能实例化 `LLVModel` 派生的所有具体模型。

## 函数形式

```python
openLLV.train(config=None, **kwargs)
```

`config` 可以是：

- 内置配置名称，例如 `"ZeroDCE"` 或 `"ZeroDCE.yaml"`；
- YAML 文件路径；
- 嵌套的配置字典；
- `None`，此时通过关键字参数提供全部必需值。

关键字参数会覆盖所选配置中加载的值。

## 数据集目录结构

当前所有内置配置都使用 `CommonDataset`。推荐的成对数据目录结构为：

```text
dataset_root/
  train/
    input/
      image_001.png
    target/
      image_001.png
  val/
    input/
      image_001.png
    target/
      image_001.png
```

文件会按不区分大小写的主干名称配对。默认验证划分为 `_test`，它能够解析常见的 `test`、`val` 和 `validation` 目录名称。也支持显式指定 `train_low_dir`、`train_high_dir`、`val_low_dir` 和 `val_high_dir`。

## 使用内置配置训练

```python
import openLLV as llv

result = llv.train(
    "ZeroDCE",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=4,
    device="cuda",
)
```

内置名称匹配不区分大小写和标点。例如，`"ZeroDCEPlusPlus"` 会解析为 `ZeroDCE++.yaml`。

## 使用 YAML 文件训练

```python
result = llv.train(
    "configs/experiment.yaml",
    lr=5e-5,
    amp=True,
)
```

## 使用字典训练

```python
config = {
    "model": {"name": "ZeroDCE", "params": {}},
    "data": {
        "dataset": "CommonDataset",
        "root_dir": "datasets/my_dataset",
        "batch_size": 4,
    },
    "loss": {"name": "zerodce", "params": {}},
    "optimizer": {"name": "adam", "lr": 1e-4},
    "train": {"epochs": 10, "device": "cuda"},
}

result = llv.train(config)
```

## 直接使用 Trainer

```python
from openLLV.deepLearning import Trainer

trainer = Trainer("ZeroDCE", root_dir="datasets/my_dataset")
result = trainer.train()
```

返回的字典包含训练历史、最佳验证损失以及检查点目录路径。训练时间戳保存在检查点和已保存的训练配置中。

## 常用覆盖参数

| 关键字 | 对应配置项 |
| --- | --- |
| `model`、`model_name` | `model.name` |
| `model_params` | `model.params` |
| `dataset`、`root_dir`、`batch_size`、`num_workers` | `data.*` |
| `loss`、`loss_params` | `loss.*` |
| `optimizer`、`lr`、`optimizer_params` | `optimizer.*` |
| `scheduler`、`scheduler_params` | `scheduler.*` |
| `epochs`、`device`、`amp`、`grad_clip` | `train.*` |
| `output_dir`、`resume`、`save_every` | `train.*` |

未知的扁平关键字会抛出 `TypeError`，不会被静默忽略。

## 输出与恢复训练

默认情况下，训练会写入 `checkpoints/<Model>_<Dataset>/`：

```text
checkpoints/<Model>_<Dataset>/
  checkpoints/
    best.pt
    last.pt
  logs/
    history.json
  <Model>.yaml
```

使用 openLLV 检查点恢复训练：

```python
result = llv.train(
    "ZeroDCE",
    root_dir="datasets/my_dataset",
    resume="checkpoints/ZeroDCE_CommonDataset/checkpoints/last.pt",
)
```

只有在有意加载部分兼容的状态字典时，才应使用 `strict_resume=False`。

