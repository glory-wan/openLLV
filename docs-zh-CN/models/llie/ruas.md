# RUAS

> 任务：低光图像增强（`llie`）

RUAS 是 openLLV 已实现的 Retinex 启发式低光增强模型。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Retinex-Inspired_Unrolling_With_Cooperative_Prior_Architecture_Search_for_Low-Light_Image_CVPR_2021_paper.pdf |
| 官方源代码 | https://github.com/KarelZhang/RUAS |

## 模型介绍

RUAS 将 Retinex 思想、展开优化和架构搜索结合起来，用于低光图像增强。模型包含与增强和去噪相关的结构，通过展开模拟迭代优化，并引入协作先验架构搜索以获得有效的网络结构。

在 openLLV 中，RUAS 包含照明增强和去噪模块。默认配置可以控制 IEM 迭代次数、NRM 层数、增强网络通道数和去噪网络通道数。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 模型实现 | `openLLV/deepLearning/models/LLIE/RUAS.py` |
| 模型类名 | `RUAS` |
| 默认配置 | `openLLV/deepLearning/config/RUAS.yaml` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/RUAS_Loss.py` |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "RUAS",
    "input.jpg",
    output="results/RUAS/output.png",
    device="cuda",
)
```

训练示例：

```python
llv.train(
    model="RUAS",
    dataset="CommonDataset",
    root_dir="datasets/my_dataset",
    loss="ruas",
    epochs=10,
    batch_size=4,
)
```

