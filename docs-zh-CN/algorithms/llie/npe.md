# NPE

> 文档分组：低光图像增强（LLIE）

NPE 是一种用于非均匀照明图像的自然性保持增强算法。

## 相关链接

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1109/TIP.2013.2261309 |
| 论文名称 | Naturalness Preserved Enhancement Algorithm for Non-Uniform Illumination Images |
| 官方源代码 | 无 |
| 官方项目页面 | 无 |

## 算法介绍

NPE 在增强非均匀照明图像的同时保持自然的视觉观感。该方法估计照明、分离类似反射率的细节、使用保持自然性的变换映射照明，并将增强细节与原始图像混合。

在 openLLV 中，NPE 使用亮通道照明估计、高斯平滑、双对数风格的照明变换以及自然性混合权重。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `openLLV/tradition/algorithms/LLIE/NPE.py` |
| 算法类名 | `NPE` |
| 注册名称 | `NPE` |
| 别名 | `npe`、`naturalness_preserved_enhancement` |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## 主要参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `sigma` | `float` | `15.0` | 亮通道照明滤波的高斯尺度 |
| `illumination_floor` | `float` | `0.05` | 照明下界 |
| `enhancement_strength` | `float` | `4.0` | 双对数照明映射强度 |
| `naturalness` | `float` | `0.35` | 保留原始自然性的混合权重 |
| `detail_weight` | `float` | `1.0` | 应用于反射率细节恢复的权重 |

## 使用示例

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "npe",
    "input.jpg",
    output="results/npe/output.jpg",
)
```

调整自然性：

```python
enhanced, saved_path = llv.predict(
    "npe",
    "input.jpg",
    output="results/npe/natural.jpg",
    naturalness=0.5,
    enhancement_strength=3.0,
)
```

