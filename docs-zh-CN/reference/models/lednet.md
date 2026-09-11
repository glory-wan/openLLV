# LEDNet

> Task: 低光图像增强与去模糊（LLIE）

LEDNet 是 openLLV 中结合金字塔池化、曲线注意力与动态滤波的编码器—解码器。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://arxiv.org/pdf/2202.03373 |
| 官方源代码 | https://github.com/sczhou/LEDNet |
| 官方项目页 | None |
| 默认配置 | `openLLV/deepLearning/config/LEDNet.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/LEDNet.py` |
| 类名 | `LEDNet` |
| 注册名 | `LEDNet`（无别名；查询忽略大小写及首尾空白） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/LEDNet_Loss.py` |

## Implementation Notes

三级下采样后接金字塔池化与曲线注意力模块；解码器应用生成的动态核与可选跳跃相加。推理返回张量；训练返回标准输出，仅激活辅助监督时含 `aux.side_output`。前向 `side_loss=None` 仅在配置的 `use_side_loss` 和训练 `mode` 同时为真时启用辅助输出；显式布尔值覆盖这一决定。YAML 将直接构造时的 `use_side_loss=False` 覆盖为 `True`，并省略保留源码默认值的架构键。

内置 YAML 对训练与验证的输入和 GT 均在 `ToTensor` 后使用 mean/std `[0.5,0.5,0.5]` 归一化，得到 `[-1,1]` 张量。训练先进行成对随机 `256×256` 裁剪，再依次以各自独立的 0.5 概率执行水平翻转、垂直翻转和宽高转置。输入与 GT 共用裁剪坐标及增强决定。验证保留原尺寸，不做随机空间变换。裁剪遇到尺寸不足或输入/GT 尺寸不一致时会报错，不会自动 resize。

`lednet` 损失及 YAML 默认 `range_norm=True`，与[官方发布的 BasicSR 设置](https://github.com/sczhou/LEDNet/blob/master/options/train_LEDNet.yml)一致。主输出和侧输出的感知项先执行 `(x + 1) / 2`，再进行 ImageNet mean/std 归一化，不截断预测或目标；像素 L1 使用原始 `[-1,1]` 张量。

YAML 同时设置模型 `image_range="minus_one_one"`，并将其随 checkpoint 保存。Predictor 在 forward 前将预处理后的 `[0,1]` 张量转换到 `[-1,1]`，预测后通过 `(output+1)/2` 转回 `[0,1]`，再进行 8-bit 转换。因此自定义 Predictor transform 仍应返回 `[0,1]` 张量，避免重复归一化。原始 `LEDNet.forward` 不转换值域：直接调用者应提供训练所用值域的输入，并按相应值域解释原始输出。直接构造默认 `image_range="zero_one"`，适配论文版权重及旧的 `[0,1]` checkpoint。

YAML 设置 `train.seed=10`、`train.cudnn_deterministic=False`、`train.cudnn_benchmark=True`，与官方训练脚本的默认 cuDNN 行为一致。Trainer 设置 Python、NumPy、PyTorch、当前 CUDA 设备及全部 CUDA 设备的随机种子。启用 benchmark 不保证多次运行结果完全相同。使用 `llv.train(config="LEDNet", root_dir="datasets/my_dataset")` 加载这些设置。

共享深度学习 Predictor 将输出转为 8-bit 图像时，依次截断至 `[0,1]`、乘 255、舍入到最近整数（中点取偶数）、转为 `uint8`。灰度、RGB、RGBA 输出均遵循此规则；模型原始张量不变。

## Parameters

| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 配置映射；非字典抛 `TypeError`；`**kwargs` 优先。 |
| `model_name` | `str` | `"LEDNet"` | 基类元数据。 |
| `input_channels` | `int` | `3` | 输入/输出通道；须为正整数。 |
| `save_dir` | `str` | `"./checkpoints/llie/LEDNet"` | 默认检查点目录。 |
| `channels` | `List[int]` | `[32, 64, 128, 128]` | 四级宽度；须含四个正值。 |
| `connection` | `bool` | `False` | 启用解码器与编码器跳跃相加；要求级间形状/通道兼容，但无显式校验。 |
| `use_side_loss` | `bool` | `False` | 默认辅助监督开关；YAML 覆盖为 `True`。 |
| `mode` | `str` | `"inference"` | 须为 `"train"` 或 `"inference"`。 |
| `image_range` | `str` | `"zero_one"` | Predictor 使用的原始网络输入/输出约定；仅接受 `"zero_one"`、`"minus_one_one"`，其他值抛 `ValueError`。YAML 覆盖为 `"minus_one_one"`；不改变原始 forward 计算。 |
| `kernel_size` | `int` | `5` | 动态卷积核尺寸；须为奇数，源码未显式校验正数。 |
| `curve_n` | `int` | `3` | 曲线注意力迭代次数；须为正。 |
| `ppm_bins` | `Tuple[int, ...]` | `(1, 2, 3, 6)` | 金字塔池化的自适应输出尺寸；无显式校验。 |

## 数据集设置

以下为 Trainer 转发给 `CommonDataset` 的可选参数，不改变其他数据集的默认行为。

| 参数 | 默认值 | LEDNet YAML | 含义 |
| --- | --- | --- | --- |
| `mean` | `None` | `data.params.mean: [0.5,0.5,0.5]` | 输入与 GT 的逐通道张量均值，验证集同样生效。 |
| `std` | `None` | `data.params.std: [0.5,0.5,0.5]` | 有限正标准差；须与 mean 同时提供。 |
| `crop_size` | `None` | `data.train_params.crop_size: 256` | 正整数方形裁剪尺寸；转换张量前对图像对同步应用。 |
| `use_flip` | `False` | `data.train_params.use_flip: true` | 以 0.5 概率同步水平翻转。 |
| `use_rot` | `False` | `data.train_params.use_rot: true` | 垂直翻转、转置分别以 0.5 概率同步执行。 |

`resize` 仍为 `None`。数据重复、`drop_last`、按 epoch 训练和学习率调度沿用现有设置。

## 损失参数

以下是 `LEDNet_Loss` 的仅关键字参数，也可通过 `loss.params` 传入：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `pixel_weight` | `float` | `1.0` | 像素 L1 权重。 |
| `perceptual_weight` | `float` | `0.01` | VGG 特征 L1 总权重；非正数禁用感知项。 |
| `side_loss_weight` | `float` | `0.8` | 完整侧输出损失的权重。 |
| `use_side_loss` | `bool` | `True` | 提供侧输出时加入辅助监督。 |
| `use_perceptual` | `bool` | `True` | 启用感知项。 |
| `pretrained_vgg` | `bool` | `True` | 加载 ImageNet VGG19 权重。 |
| `layer_weights` | `Optional[Dict[str, float]]` | `None` | 默认采用 `conv1_2`、`conv2_2`、`conv3_4`、`conv4_4`，各权重为 1；不支持的名称抛 `KeyError`。 |
| `range_norm` | `bool` | `True` | VGG 前执行 `(x+1)/2`，不截断。 |
| `use_input_norm` | `bool` | `True` | VGG 前使用 ImageNet mean/std 归一化。 |

`VGG19PerceptualLoss` 也默认使用 `range_norm=True`。设置 `loss_params={"range_norm": False}` 可仅关闭范围转换。

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("LEDNet", "input.jpg", output="results/lednet/output.png", config={"connection": True, "curve_n": 4})
```

## Checkpoint / Official Weights

将 openLLV 检查点路径作为预测目标。未实现官方权重自动下载。
