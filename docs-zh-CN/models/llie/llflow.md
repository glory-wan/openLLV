# LLFlow

标准 LLFlow（AAAI 2022），依据 [官方 LOL-pc 实现](https://github.com/wyf0912/LLFlow) 移植。

使用 24 个 RRDB 的条件编码器和三级 normalizing flow。训练最小化条件高斯 NLL，推理从预测颜色图先验均值解码，保留官方直方图均衡、对数预处理和对称补边。

模型层与辅助函数合并在 `LLFlow.py`，预处理函数放在 `LLFlowDataset.py`，
训练与推理共用相同逻辑。

完整参数、官方权重转换、训练示例、许可证与集成限制见 [LLFlow 功能文档](../../reference/models/llflow.md)。旧简化模型参数和权重不兼容。
