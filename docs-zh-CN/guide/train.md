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

文件会按不区分大小写的主干名称配对。Trainer 的通用默认验证划分为 `"val"`；内置配置可以覆盖该值，常用的 `_test` 能够解析常见的 `test`、`val` 和 `validation` 目录名称。也支持显式指定 `train_input_dir`、`train_target_dir`、`val_input_dir` 和 `val_target_dir`。

## 使用内置配置训练

```python
import openLLV as llv

result = llv.train(
    "ZeroDCE",
    root_dir="datasets/my_dataset",
    epochs=10,
    batch_size=4,
    resize=512,
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
        "resize": [384, 512],
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

## Trainer 参数

下表默认值是内置 YAML、自定义 YAML 或配置字典覆盖之前的 Trainer 通用默认值。`model_params`、`loss_params`、`optimizer_params`、`scheduler_params` 及数据集参数字典中的组件专属参数不会在此展开。

| 参数名 | 默认值 | 参数作用解释 |
| --- | --- | --- |
| `config` | `None` | 基础配置，可以是内置配置名、YAML 路径、嵌套字典或 `None`。 |
| `model` | `None` | 设置 `model.name`；接受注册名、检查点路径、`LLVModel` 类/实例或完整 `model` 节字典。模型为必填项。 |
| `model_name` | `None` | 使用 `model.name` 选择模型时 `model` 的别名。 |
| `model_params` | `{}` | 作为 `model.params` 转发给所选模型的字典；模型专属参数不在此列出。 |
| `model_config` | `{}` | 直接合并到完整 `model` 节的字典。 |
| `dataset` | `"CommonDataset"` | 选择已注册数据集、数据集类或已有数据集实例。 |
| `dataset_name` | `"CommonDataset"` | `dataset` 的别名。 |
| `root_dir` | `None` | 数据集根目录；除非 `dataset` 是已有数据集实例，否则必填。 |
| `batch_size` | `4` | 每个训练或验证 batch 的样本数；必须为正整数。 |
| `num_workers` | `0` | DataLoader 工作进程数；必须为非负整数。 |
| `pin_memory` | `True` | 在 CUDA 训练时启用 DataLoader 固定内存。 |
| `shuffle` | `True` | 是否打乱训练 DataLoader；验证数据不会打乱。 |
| `drop_last` | `False` | 是否丢弃训练集最后一个不完整 batch。 |
| `train_split` | `"train"` | 训练使用的数据集划分名称。 |
| `val_split` | `"val"` | 验证使用的数据集划分名称；假值会关闭验证数据集的自动构建。 |
| `return_filename` | `True` | 要求支持该选项的数据集返回文件名。 |
| `resize` | `None` | 数据集输出尺寸；`None` 保持原尺寸，整数表示正方形，二项序列表示 `(height, width)`。 |
| `train_input_dir` | `None` | 显式训练输入目录，覆盖数据集目录结构自动发现。 |
| `train_target_dir` | `None` | 显式训练目标目录；无参考训练可以保持 `None`。 |
| `val_input_dir` | `None` | 显式验证输入目录，覆盖数据集目录结构自动发现。 |
| `val_target_dir` | `None` | 显式验证目标目录。 |
| `data_params` | `{}` | 传给训练和验证数据集构造器的共享参数字典；数据集专属参数不在此列出。 |
| `train_params` | `{}` | 合并在 `data_params` 之上的训练划分数据集构造参数。 |
| `val_params` | `{}` | 合并在 `data_params` 之上的验证划分数据集构造参数。 |
| `data` | 默认 `data` 节 | 合并在通用数据默认值之上的完整嵌套 data 节字典。 |
| `loss` | `None` | 设置 `loss.name`；接受已注册/可导入损失、损失类/实例或完整 `loss` 节字典。省略时 Trainer 推断匹配损失，否则使用 Charbonnier 损失。 |
| `loss_name` | `None` | 使用 `loss.name` 选择损失时 `loss` 的别名。 |
| `loss_params` | `{}` | 传给所选损失构造器的字典；损失函数专属参数不在此列出。 |
| `output_index` | `None` | 从 tuple/list 模型输出中选择一个元素参与损失计算。 |
| `output_key` | `None` | 从字典模型输出中选择一个值参与损失计算。 |
| `optimizer` | `"adam"` | 选择优化器名称/类/实例，或传入完整 `optimizer` 节字典；内置名称为 Adam、AdamW、SGD 和 RMSprop。 |
| `optimizer_name` | `"adam"` | 使用 `optimizer.name` 选择优化器时 `optimizer` 的别名。 |
| `lr` | `1e-4` | 优化器学习率；必须大于零。 |
| `optimizer_params` | `{}` | 额外优化器构造参数；优化器专属参数不在此列出。 |
| `scheduler` | `None` | 选择调度器名称/实例，或传入完整 `scheduler` 节字典；`None` 关闭学习率调度。 |
| `scheduler_name` | `None` | 使用 `scheduler.name` 选择调度器时 `scheduler` 的别名。 |
| `scheduler_params` | `{}` | 调度器构造参数；调度器专属参数不在此列出。 |
| `epochs` | `100` | 总训练轮数；必须为正整数。 |
| `output_dir` | `None` | 训练输出目录；`None` 解析为 `checkpoints/<Model>_<Dataset>`。 |
| `save_every` | `1` | 每隔多少个 epoch 保存一次 `last.pt`；必须为正整数。 |
| `validate_every` | `1` | 存在验证 loader 时每隔多少个 epoch 验证一次；必须为正整数。 |
| `log_every` | `10` | 每隔多少个 batch 更新一次训练进度显示；必须为正整数。 |
| `grad_clip` | `None` | 最大梯度范数；`None` 关闭裁剪，否则必须大于零。 |
| `amp` | `False` | 解析出的设备为 CUDA 时启用自动混合精度。 |
| `resume` | `None` | 用于恢复模型及训练状态的 openLLV 训练检查点路径。 |
| `resume_path` | `None` | `resume` 的别名。 |
| `strict_resume` | `True` | 恢复训练时是否严格加载模型状态字典。 |
| `seed` | `42` | Python、NumPy 和 PyTorch 随机种子；`None` 表示不设置种子。 |
| `device` | 依次选择 CUDA、MPS、CPU | 未显式指定时从可用后端中选择最佳训练设备。 |
| `progress_bar` | `True` | 是否显示训练和验证 tqdm 进度条。 |
| `train` | 默认 `train` 节 | 合并在通用训练默认值之上的完整嵌套 train 节字典。 |

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
