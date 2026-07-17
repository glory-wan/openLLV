# BIMEF

> 文档分组：低光图像增强（LLIE）

BIMEF 是一种用于低光图像增强的仿生多曝光融合框架。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.48550/arXiv.1711.00591 |
| 论文名称 | A Bio-Inspired Multi-Exposure Fusion Framework for Low-light Image Enhancement |
| 官方源代码 | 无 |
| 官方项目页面 | 无 |

## 算法介绍

BIMEF 通过创建输入图像的曝光调整版本，并将其与原始图像融合来增强低光图像。融合权重根据对比度、饱和度和适度曝光程度计算，从而在细节与自然观感之间取得平衡。

在 openLLV 中，曝光比可以手动指定，也可以根据图像亮度均值自动估计。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `openLLV/tradition/algorithms/LLIE/BIMEF.py` |
| 算法类名 | `BIMEF` |
| 注册名称 | `BIMEF` |
| 别名 | `bimef`、`bio_inspired_multi_exposure_fusion` |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `exposure_ratio` | `Optional[float]` | `None` | 手动曝光比；为 `None` 时自动估计 |
| `target_mean` | `float` | `0.55` | 自动曝光的目标亮度均值 |
| `max_ratio` | `float` | `5.0` | 自动曝光比的最大值 |
| `well_exposed_sigma` | `float` | `0.2` | 适度曝光权重的 Sigma |
| `contrast_weight` | `float` | `1.0` | 对比度权重指数 |
| `saturation_weight` | `float` | `1.0` | 饱和度权重指数 |
| `well_exposed_weight` | `float` | `1.0` | 适度曝光权重指数 |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "bimef",
    "input.jpg",
    output="results/bimef/output.jpg",
)
```

手动设置曝光：

```python
enhanced, saved_path = llv.predict(
    "bimef",
    "input.jpg",
    output="results/bimef/manual.jpg",
    exposure_ratio=3.0,
)
```

