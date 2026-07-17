# LLFormer

> 任务：低光图像增强（`llie`）

LLFormer 是论文 **Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method**（AAAI 2023 Oral）提出的 Transformer 方法。它结合行/列轴向自注意力、双门控前馈网络、跨层注意力融合、四级编码器—解码器以及可学习的加权跳跃连接。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://arxiv.org/abs/2212.11548 |
| 官方源代码 | https://github.com/TaoWangzj/LLFormer |
| 默认配置 | `openLLV/deepLearning/config/LLFormer.yaml` |

## 许可证说明

集成架构改编自上游实现，该实现以 CC BY-NC-SA 4.0 许可用于学术、非商业用途。在重新分发或商业使用前，请查看官方源代码仓库中的许可证；其条款可能比 openLLV 仓库级许可证更严格。

## openLLV 实现

| 项目 | 位置 |
| --- | --- |
| 模型 | `openLLV/deepLearning/models/LLIE/LLFormer.py` |
| 模型名称 | `LLFormer`（别名：`LL-Former`） |
| 损失函数 | `openLLV/deepLearning/loss/LLIELoss/LLFormer_Loss.py` |
| 损失名称 | `llformer` |

默认架构使用 `dim=16`、块数量 `[2, 4, 8, 16]`、注意力头数 `[1, 2, 4, 8]`、两个细化块、WithBias 层归一化，并且不使用全局残差跳跃。已注册损失函数遵循官方 Smooth L1 训练目标。

LLFormer 会将输入填充到尺寸可被 16 整除，并将预测裁剪回原始大小。

## 训练

内置配置使用 `CommonDataset`，要求 `train/input` 和 `train/target` 下存在匹配文件，并提供对应的验证目录。

```bash
python -m openLLV.cli train LLFormer --kwargs root_dir=datasets/my_dataset
```

或者使用 Python：

```python
import openLLV as llv

result = llv.train("LLFormer", root_dir="datasets/my_dataset")
print(result["checkpoint_dir"])
```

## 预测

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "checkpoints/LLFormer_CommonDataset/checkpoints/best.pt",
    "input.png",
    output="results/LLFormer/output.png",
    device="cuda",
)
```

为了节省 UHD 推理内存，可以启用重叠分块并取平均：

```python
enhanced, saved_path = llv.predict(
    "checkpoints/LLFormer_CommonDataset/checkpoints/best.pt",
    "uhd_input.png",
    output="results/LLFormer/uhd_output.png",
    device="cuda",
    config={
        "tile_size": [720, 1280],
        "tile_overlap": [360, 640],
    },
)
```

分块尺寸必须可被 16 整除。包括边界分块在内的重叠预测会进行平均。

## 官方检查点

官方检查点可能把权重存储在 `state_dict` 下，并在使用 `DataParallel` 训练后为键添加 `module.` 前缀。将原始状态字典加载到 `LLVModel.create_model("LLFormer")` 之前，请移除该前缀。openLLV 生成的检查点已经包含模型元数据，可以直接交给 `llv.predict`。

