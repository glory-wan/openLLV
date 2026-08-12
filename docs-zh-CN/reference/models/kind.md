# KinD

> Task: 低光图像增强（LLIE）

KinD 是 openLLV 中采用分解、反射率恢复与照明调整流程的 Retinex 风格模型。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1145/3343031.3350926 |
| 官方源代码 | https://github.com/zhangyhuaee/KinD |
| 官方项目页 | None |
| 默认配置 | `openLLV/deepLearning/config/KinD.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/KinD.py` |
| 类名 | `KinD` |
| 注册名 | `KinD`（无别名；查询忽略大小写及首尾空白） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/KinD_Loss.py` |

## Implementation Notes

KinD 将输入分解成反射率和照明，恢复反射率，用常量 `illumination_ratio` 比例图调整照明，二者相乘后裁剪至 `[0, 1]`。推理返回张量；训练返回标准字典及全部中间张量和 `decompose_fn`。默认 YAML 与模型默认值一致，但省略 `mode`、`model_name`、`save_dir`。

## Parameters

| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 配置映射；非字典抛 `TypeError`；`**kwargs` 优先。 |
| `model_name` | `str` | `"KinD"` | 基类元数据。 |
| `input_channels` | `int` | `3` | 图像通道；须为正整数。 |
| `save_dir` | `str` | `"./checkpoints/llie/KinD"` | 默认检查点目录。 |
| `decomposition_channels` | `int` | `64` | 分解网络宽度；转为 `int` 且须为正。 |
| `decomposition_layers` | `int` | `5` | 分解网络层数；转为 `int` 且须为正。 |
| `restoration_channels` | `int` | `32` | 恢复网络宽度；转为 `int` 且须为正。 |
| `adjustment_channels` | `int` | `32` | 照明调整网络宽度；转为 `int` 且须为正。 |
| `adjustment_layers` | `int` | `3` | 照明调整网络层数；转为 `int` 且须为正。 |
| `illumination_ratio` | `float` | `5.0` | 常量曝光比例图的值；须为正。 |
| `mode` | `str` | `"inference"` | 须为 `"train"` 或 `"inference"`。 |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("KinD", "input.jpg", output="results/kind/output.png", config={"illumination_ratio": 4.0})
```

## Checkpoint / Official Weights

将 openLLV 检查点路径作为预测目标。未实现官方权重自动下载。
