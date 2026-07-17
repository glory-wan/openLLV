# Retinex

> 文档分组：低光图像增强（LLIE）

Retinex 是 openLLV 已实现的一组传统低光增强算法。当前实现包括 SSR、MSR 和 MSRCR。

## 相关链接

| 类型 | URL |
| --- | --- |
| SSR 相关论文 | https://doi.org/10.1109/83.557356 |
| MSR / MSRCR 相关论文 | https://doi.org/10.1109/83.597272 |
| 官方源代码 | 无 |
| 官方项目页面 | 无 |

## 算法介绍

Retinex 方法基于分离照明和反射率的思想。在低光增强中，Retinex 抑制缓慢变化的照明分量，并突出与反射率相关的图像细节。

openLLV 当前实现了三个 Retinex 变体：

| 算法 | 含义 |
| --- | --- |
| `SSR` | 单尺度 Retinex（Single Scale Retinex） |
| `MSR` | 多尺度 Retinex（Multi Scale Retinex） |
| `MSRCR` | 带颜色恢复的多尺度 Retinex（Multi Scale Retinex with Color Restoration） |

## SSR

SSR 使用一个高斯环绕尺度估计照明：

```text
R(x, y) = log(I(x, y)) - log(G_sigma(x, y) * I(x, y))
```

SSR 简单且高效，但单一尺度可能无法同时良好适应局部细节和全局照明。

## MSR

MSR 对多个高斯环绕尺度下的 Retinex 响应取平均：

```text
MSR = mean(SSR_sigma_1, SSR_sigma_2, ..., SSR_sigma_n)
```

与 SSR 相比，MSR 能够更有效地平衡细节增强和全局色调一致性。

## MSRCR

MSRCR 在 MSR 的基础上增加颜色恢复项，以减少逐通道 Retinex 处理造成的颜色失真。它常用于彩色低光图像。

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 算法实现 | `openLLV/tradition/algorithms/LLIE/Retinex.py` |
| SSR 类名 | `SSR` |
| MSR 类名 | `MSR` |
| MSRCR 类名 | `MSRCR` |
| 注册名称 | `SSR`、`MSR`、`MSRCR` |
| 别名 | `ssr`、`single_scale_retinex`、`msr`、`multi_scale_retinex`、`msrcr`、`multi_scale_retinex_color_restoration` |
| 基类 | `openLLV/tradition/algorithms/BaseModel.py` 中的 `LLVEnhancer` |

## 主要参数

通用参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `low_clip` | `float` | `1.0` | 显示归一化使用的低百分位 |
| `high_clip` | `float` | `99.0` | 显示归一化使用的高百分位 |
| `eps` | `float` | `1e-6` | 避免对数和除法不稳定的小数值 |

SSR 参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `sigma` | `float` | `80.0` | 高斯环绕尺度 |

MSR 参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `scales` | `Sequence[float]` | `(15.0, 80.0, 250.0)` | 高斯环绕尺度 |

MSRCR 参数：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `scales` | `Sequence[float]` | `(15.0, 80.0, 250.0)` | 高斯环绕尺度 |
| `alpha` | `float` | `125.0` | 颜色恢复强度增益 |
| `beta` | `float` | `46.0` | 颜色恢复对数增益 |
| `gain` | `float` | `1.0` | 应用于恢复后 Retinex 响应的全局增益 |
| `offset` | `float` | `0.0` | 显示归一化之前应用的全局偏移 |

## 使用示例

SSR：

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "ssr",
    "input.jpg",
    output="results/ssr/output.jpg",
    sigma=80.0,
)
```

MSR：

```python
enhanced, saved_path = llv.predict(
    "msr",
    "input.jpg",
    output="results/msr/output.jpg",
    scales=(15.0, 80.0, 250.0),
)
```

MSRCR：

```python
enhanced, saved_path = llv.predict(
    "msrcr",
    "input.jpg",
    output="results/msrcr/output.jpg",
    alpha=125.0,
    beta=46.0,
)
```

文件夹批处理：

```python
saved_paths = llv.predict(
    "msrcr",
    "images/",
    output="results/msrcr",
    progress_bar=True,
)
```

