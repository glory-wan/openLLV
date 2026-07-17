# HVI-CIDNet

> 任务：低光图像增强（`llie`）

HVI-CIDNet 是论文 **HVI: A New Color Space for Low-light Image Enhancement**（CVPR 2025）提出的颜色与强度解耦网络（Color and Intensity Decoupling Network）。它将 RGB 图像转换到可学习的 HVI 颜色空间，在两个交互分支中分别处理色度 H/V 特征和强度特征，并通过 Lighten Cross-Attention（LCA）块进行耦合。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://arxiv.org/abs/2502.20272 |
| 官方源代码 | https://github.com/Fediory/HVI-CIDNet |
| 默认配置 | `openLLV/deepLearning/config/CIDNet.yaml` |

## openLLV 实现

| 项目 | 位置 |
| --- | --- |
| 模型 | `openLLV/deepLearning/models/LLIE/CIDNet.py` |
| 模型名称 | `CIDNet`（别名：`HVI-CIDNet`） |
| 损失函数 | `openLLV/deepLearning/loss/LLIELoss/CIDNet_Loss.py` |
| 损失名称 | `cidnet` |

集成的损失函数在 RGB 和 HVI 两个域中遵循官方训练目标：

```text
L = L_RGB + hvi_weight * L_HVI
L_domain = pixel_weight * L1
         + ssim_weight * (1 - SSIM)
         + edge_weight * L_edge
         + perceptual_weight * L_VGG19
```

默认权重分别为 `1.0`、`0.5`、`50.0`、`0.01`，并使用 `hvi_weight=1.0`，与上游默认值一致。VGG19 感知特征使用 `conv1_2`、`conv2_2`、`conv3_4` 和 `conv4_4`，距离度量为 MSE。首次训练时可能会通过 torchvision 下载 ImageNet VGG19 权重。离线或轻量冒烟测试可设置 `loss.params.use_perceptual: false`。

CIDNet 会在内部将张量填充到 8 的倍数，并将输出裁剪回原始大小，因此预测支持任意图像尺寸。`input_gamma`、`saturation_scale` 和 `intensity_scale` 暴露了上游推理控制项，默认值均为 `1.0`。

## 训练

编辑 YAML 文件中的数据集路径，然后运行：

```bash
python -m openLLV.cli train openLLV/deepLearning/config/CIDNet.yaml
```

或者使用 Python：

```python
import openLLV as llv

result = llv.train("CIDNet", root_dir="datasets/my_dataset")
print(result["checkpoint_dir"])
```

内置配置使用 `CommonDataset`，推荐的成对目录为 `train/input`、`train/target`、`val/input` 和 `val/target`。可以通过 `data.dataset` 选择其他已注册数据集。

## 预测

使用 openLLV 训练器生成的检查点：

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "checkpoints/CIDNet_CommonDataset/checkpoints/best.pt",
    "input.jpg",
    output="results/CIDNet/output.png",
    device="cuda",
)
```

如需调整可选的上游风格推理控制项，可以通过 `Predictor` 传入检查点配置覆盖参数，或者使用 `LLVModel.create_model("CIDNet", config={...})` 创建模型。

## 官方原始权重

该架构保留了官方参数名称，因此可以把上游原始 `.pth` 状态字典加载到新创建的模型中：

```python
import torch
from openLLV.deepLearning.models import LLVModel

model = LLVModel.create_model("CIDNet")
state = torch.load("LOLv1.pth", map_location="cpu")
model.load_state_dict(state, strict=True)
model.to("cuda").eval_mode()
```

openLLV 训练检查点还包含配置和优化器元数据；这些检查点可以直接交给 `llv.predict` 使用。

