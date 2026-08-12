# KinD++

> Task: 低光图像增强（LLIE）

KinD++ 是 openLLV 中增强版的 Retinex 分解、恢复与照明调整模型。

## Links

| 类型 | URL |
| --- | --- |
| 论文 | https://doi.org/10.1007/s11263-020-01407-x |
| 官方源代码 | https://github.com/zhangyhuaee/KinD_plus |
| 官方项目页 | None |
| 默认配置 | `openLLV/deepLearning/config/KinD++.yaml` |

## Location in openLLV

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/KinDPlusPlus.py` |
| 类名 | `KinDPlusPlus` |
| 注册名 | `KinDPlusPlus`（别名：`Kind++`；查询忽略大小写及首尾空白） |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 `LLVModel` |
| 相关损失 | `openLLV/deepLearning/loss/LLIELoss/KinDPlusPlus_Loss.py` |

## Implementation Notes

KinD++ 分解输入，以多尺度照明感知网络恢复反射率，用常量比例图调整照明，二者相乘后裁剪至 `[0, 1]`。推理返回张量；训练返回标准输出以及全部中间张量和 `decompose_fn`。默认 YAML 与模型默认值一致，但省略 `mode`、`model_name`、`save_dir`。

## Parameters

| 参数 | 类型 | 默认值 | 含义 / 约束 |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | 配置映射；非字典抛 `TypeError`；`**kwargs` 优先。 |
| `model_name` | `str` | `"KinDPlusPlus"` | 基类元数据。 |
| `input_channels` | `int` | `3` | 图像通道；须为正整数。 |
| `save_dir` | `str` | `"./checkpoints/llie/KinDPlusPlus"` | 默认检查点目录。 |
| `decomposition_channels` | `int` | `32` | 分解网络基础宽度；转为 `int` 且须为正。 |
| `restoration_channels` | `int` | `32` | 恢复网络基础宽度；转为 `int` 且须为正。 |
| `adjustment_channels` | `int` | `32` | 照明调整网络宽度；转为 `int` 且须为正。 |
| `illumination_ratio` | `float` | `5.0` | 常量曝光比例图的值；须为正。 |
| `mode` | `str` | `"inference"` | 须为 `"train"` 或 `"inference"`。 |

## Usage Example

```python
import openLLV as llv
enhanced, saved_path = llv.predict("Kind++", "input.jpg", output="results/kind++/output.png", config={"illumination_ratio": 4.0})
```

## Checkpoint / Official Weights

将 openLLV 检查点路径作为预测目标。未实现官方权重自动下载。
