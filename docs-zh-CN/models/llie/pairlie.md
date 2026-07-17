# PairLIE

> 任务：低光图像增强（`llie`）

PairLIE 是论文 **Learning a Simple Low-light Image Enhancer from Paired Low-light Instances**（CVPR 2023）提出的模型。它不要求正常光真值，而是从同一场景的两次低光观测中学习。共享网络估计去噪表示、照明和反射率，并通过反射率一致性关联两次观测。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2023/papers/Fu_Learning_a_Simple_Low-Light_Image_Enhancer_From_Paired_Low-Light_Instances_CVPR_2023_paper.pdf |
| 官方源代码 | https://github.com/zhenqifu/PairLIE |
| 默认配置 | `openLLV/deepLearning/config/PairLIE.yaml` |

## openLLV 实现

| 项目 | 位置 |
| --- | --- |
| 模型 | `openLLV/deepLearning/models/LLIE/PairLIE.py` |
| 模型名称 | `PairLIE`（别名：`Pair-LIE`） |
| 损失函数 | `openLLV/deepLearning/loss/LLIELoss/PairLIE_Loss.py` |
| 损失名称 | `pairlie` |
| 数据集 | `openLLV/data/datasets/CommonDataset.py` |

推理期间，PairLIE 生成：

```text
enhanced = illumination ** enhancement_gamma * reflectance
```

`enhancement_gamma` 的默认值为 `0.2`，与官方通用设置一致。官方 LOL 示例使用 `0.14`；预测时可将其作为检查点配置覆盖参数。

集成的训练目标遵循官方实现：

```text
L = consistency_weight * MSE(R1, R2)
  + reconstruction_weight * L_reconstruction
  + preservation_weight * MSE(input1, denoised1)
```

`L_reconstruction` 包括 Retinex 重建、反射率估计、照明与最大 RGB 的一致性，以及照明总变差。默认保持权重为 `500`。

## 训练数据

PairLIE 需要同一场景的两个不同低光实例，而不是低光/正常光图像对。配置的 `CommonDataset` 使用成对目录结构：

```text
PairLIE-training-dataset/
  train/
    input/
      scene_1.png
      scene_2.png
    target/
      scene_1.png
      scene_2.png
```

将第一次曝光放入 `input`，第二次曝光放入 `target`，并使用匹配的文件名。对于 PairLIE，`target` 中的图像是另一次低光观测，而不是正常光真值。

## 训练

在 YAML 文件中设置 `data.root_dir`，然后运行：

```bash
python -m openLLV.cli train openLLV/deepLearning/config/PairLIE.yaml
```

或者使用 Python：

```python
import openLLV as llv

result = llv.train("PairLIE", root_dir="datasets/pairlie")
print(result["checkpoint_dir"])
```

统一 Trainer 会检测 PairLIE 的成对前向传播约定，并将两个低光实例送入同一个模型。其他已注册模型仍使用标准单输入路径。

## 预测

使用 openLLV 写入的检查点：

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "checkpoints/PairLIE_CommonDataset/checkpoints/best.pt",
    "input.jpg",
    output="results/PairLIE/output.png",
    device="cuda",
)
```

使用 LOL 风格推理：

```python
enhanced, saved_path = llv.predict(
    "checkpoints/PairLIE_CommonDataset/checkpoints/best.pt",
    "input.jpg",
    output="results/PairLIE/lol.png",
    device="cuda",
    config={"enhancement_gamma": 0.14},
)
```

该实现保留官方网络参数名称，因此发布的原始 `PairLIE.pth` 状态字典也可以通过 `load_state_dict(..., strict=True)` 加载到默认 `PairLIE` 模型中。openLLV 检查点包含 `llv.predict` 所需的额外模型配置。

