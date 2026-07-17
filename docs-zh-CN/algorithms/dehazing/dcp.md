# DCP / Dark Channel

> 文档分组：去雾

DCP 是 openLLV 已实现的基于暗通道先验的传统增强算法。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://ieeexplore.ieee.org/document/5206515 |
| 扩展期刊版本 | https://pubmed.ncbi.nlm.nih.gov/20820075/ |
| 官方源代码 | 无 |
| 官方项目页面 | 无 |

## 算法介绍

DCP（Dark Channel Prior，暗通道先验）最初用于单幅图像去雾。其核心观察是：在非天空自然图像的大多数局部区域中，至少有一个颜色通道的像素值非常低。该先验可用于估计透射率和大气光。

在低光增强中，一种常见做法是先反转低光图像，将其转换为类似有雾图像的形式；然后使用暗通道先验进行恢复；最后再次反转，得到增强图像。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `openLLV/tradition/algorithms/Dehazing/DCP.py` |
| 算法类名 | `DarkChannel` |
| 注册名称 | `DarkChannel` |
| 别名 | `dcp` |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## 实现说明

openLLV 的 DCP 流水线包括：

1. 归一化输入图像。
2. 反转图像。
3. 计算暗通道。
4. 估计大气光。
5. 估计并细化透射图。
6. 恢复图像并再次反转。

主要参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `size` | `int` | `15` | 暗通道腐蚀核大小 |
| `omega` | `float` | `0.95` | 透射率估计权重 |
| `t_min` | `float` | `0.1` | 最小透射率 |
| `guided_radius` | `int` | `60` | 引导滤波半径 |
| `guided_eps` | `float` | `1e-4` | 引导滤波正则项 |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "dcp",
    "input.jpg",
    output="results/dcp/output.jpg",
    size=15,
    omega=0.95,
    t_min=0.1,
)
```

文件夹批处理：

```python
saved_paths = llv.predict(
    "dcp",
    "images/",
    output="results/dcp",
    progress_bar=True,
)
```

