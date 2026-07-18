# 训练配置

openLLV 将内置 YAML 训练配置保存在 `openLLV/deepLearning/config/` 中。该包同时定义了默认值，以及用于解析、加载和递归合并配置的辅助函数。

## 内置 YAML 文件

| 配置名称 | 模型 | 数据集 | 损失函数 |
| --- | --- | --- | --- |
| `CIDNet` | `CIDNet` | `CommonDataset` | `cidnet` |
| `DarkIR` | `DarkIR` | `CommonDataset` | `darkir` |
| `EnlightenGAN` | `EnlightenGAN` | `CommonDataset` | `enlightengan` |
| `KinD` | `KinD` | `CommonDataset` | `kind` |
| `KinD++` | `KinDPlusPlus` | `CommonDataset` | `kindplusplus` |
| `LEDNet` | `LEDNet` | `CommonDataset` | `lednet` |
| `LLFlow` | `LLFlow` | `CommonDataset` | `llflow` |
| `LLFormer` | `LLFormer` | `CommonDataset` | `llformer` |
| `LLNet` | `LLNet` | `CommonDataset` | `llnet` |
| `PairLIE` | `PairLIE` | `CommonDataset` | `pairlie` |
| `RetinexFormer` | `RetinexFormer` | `CommonDataset` | `retinexformer` |
| `RUAS` | `RUAS` | `CommonDataset` | `ruas` |
| `SCI` | `SCI` | `CommonDataset` | `sci` |
| `URetinexNet` | `URetinexNet` | `CommonDataset` | `uretinex` |
| `ZeroDCE` | `ZeroDCE` | `CommonDataset` | `zerodce` |
| `ZeroDCE++` | `ZeroDCEPlusPlus` | `CommonDataset` | `zerodce_extension` |
| `ZeroIG` | `ZeroIG` | `CommonDataset` | `zeroig` |

通过程序列出内置配置名称：

```python
from openLLV.deepLearning.config import list_available_configs

print(list_available_configs())
```

名称可以是文件名、文件名主干或 YAML 中的 `model.name`。匹配时不区分大小写和标点。

## YAML 结构

```yaml
model:
  name: ZeroDCE
  params: {}

data:
  dataset: CommonDataset
  root_dir: path/to/your/dataset
  train_split: train
  val_split: _test
  batch_size: 4
  num_workers: 4
  pin_memory: true
  return_filename: true
  resize: null

loss:
  name: zerodce
  params: {}
  output_index: null
  output_key: null

optimizer:
  name: adam
  lr: 0.0001
  params:
    betas: [0.9, 0.999]
    weight_decay: 0.0

scheduler:
  name: cosineannealinglr
  params:
    T_max: 100
    eta_min: 0.000001

train:
  epochs: 100
  device: null
  output_dir: null
  save_every: 1
  validate_every: 1
  log_every: 10
  grad_clip: 1.0
  amp: false
  seed: 42
  resume: null
```

## 配置分区

### `model`

`name` 选择一个已注册的 `LLVModel`。`params` 会传给该模型的构造函数。

### `data`

`dataset` 选择已注册的数据集。内置的 `CommonDataset` 配置要求提供 `root_dir`。共享的构造参数放在 `params` 中，特定数据划分的参数分别放在 `train_params` 和 `val_params` 中。同时支持显式指定 `train_input_dir`、`train_target_dir`、`val_input_dir` 和 `val_target_dir`。`resize` 为整数时输出正方形，为 `[height, width]` 时输出指定高宽。

### `loss`

`name` 选择已注册的 `BaseLoss`。`params` 用于配置损失函数。`output_key` 或 `output_index` 可以从非标准的结构化模型输出中选择张量；常规的 `pred` 字典不需要这两个选项。

### `optimizer`

`name` 可以选择 `adam`、`adamw`、`sgd` 或 `rmsprop`。`lr` 是学习率，`params` 会转交给优化器。

### `scheduler`

设置 `name: null` 可禁用调度器。否则，`params` 会转交给所选的 PyTorch 学习率调度器。

### `train`

该分区控制训练轮数、运行设备、输出位置、验证与检查点频率、混合精度、梯度裁剪、可复现性和恢复训练行为。设备为 null 时，依次选择可用的 CUDA、MPS，最后回退到 CPU。

## 默认值与合并辅助函数

`openLLV/deepLearning/config/__init__.py` 导出：

- `DEFAULT_TRAIN_CONFIG`；
- 返回独立配置副本的 `get_default_train_config()`；
- `get_default_device()`；
- 用于递归合并字典的 `deep_update()`；
- `get_config_path()` 和 `load_config()`；
- `list_available_configs()`。

内置 YAML 的值会递归覆盖 `DEFAULT_TRAIN_CONFIG`，然后再应用显式的训练器关键字覆盖参数。

## 覆盖参数示例

```python
import openLLV as llv

result = llv.train(
    "ZeroDCE",
    root_dir="datasets/my_dataset",
    batch_size=2,
    lr=5e-5,
    epochs=20,
    device="cpu",
)
```

也可以直接提供嵌套分区：

```python
result = llv.train(
    {
        "model": {"name": "ZeroDCE"},
        "data": {
            "dataset": "CommonDataset",
            "root_dir": "datasets/my_dataset",
        },
        "loss": {"name": "zerodce"},
        "train": {"epochs": 20},
    }
)
```

不要在 `model.params` 中添加 `device`。设备放置由训练器或预测器负责。
