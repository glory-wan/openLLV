# LLFlow

> 任务：低光图像增强（`llie`）

依据官方 LOL-pc 实现的标准 LLFlow，替换原来的简化 flow。
Trainer、统一 Predictor 和深度学习 Predictor 均未修改。

## 链接

| 类型 | 地址 |
| --- | --- |
| 论文 | https://doi.org/10.1609/aaai.v36i3.20162 |
| 官方代码 | https://github.com/wyf0912/LLFlow |
| 对照版本 | `115da161a96de868d67494a32db848e31f85bbc1` |
| 官方配置 | https://github.com/wyf0912/LLFlow/blob/115da161a96de868d67494a32db848e31f85bbc1/code/confs/LOL-pc.yml |
| openLLV 配置 | `openLLV/deepLearning/config/LLFlow.yaml` |

## 在 openLLV 中的位置

| 项目 | 位置 |
| --- | --- |
| 实现 | `openLLV/deepLearning/models/LLIE/LLFlow.py` |
| 官方模块与许可证 | `openLLV/deepLearning/models/LLIE/_llflow/` |
| 类名 / 注册名 | LLFlow，大小写不敏感，无别名 |
| 基类 | `openLLV/deepLearning/models/BaseModel.py` 中的 LLVModel |
| 损失 | `openLLV/deepLearning/loss/LLIELoss/LLFlow_Loss.py` |
| 数据适配 | `openLLV/data/datasets/LLFlowDataset.py` |

## 实现说明

ConEncoder1 使用 64 通道、24 个 RRDB，将 block 1/3/5/7 的特征与各尺度特征
拼接。共三级 flow，每级依次 squeeze、两个无耦合的 ActNorm/可逆 1×1
卷积步骤、十二个 CondAffineSeparatedAndCond 步骤。关闭 split；
最终 latent 为 192 通道，高宽均为输入的 1/8。

预处理拼接 log(low + 0.001) 与 OpenCV 逐通道直方图均衡结果，编码器再
计算颜色和梯度/噪声图。LLFlowDataset 先对整张图均衡，再同步裁剪/翻转
low、GT 和均衡图。支持 LOL 的 our485/low、our485/high、eval15/low、
eval15/high 和 CommonDataset 的 input/target 布局。其他布局需要显式
指定 input_dir/target_dir，不自动发现 LOL-v2 目录。

训练仅使用条件高斯 NLL 均值：
`-(logdet + log N(z; mean, I)) / (log(2) * H * W)`。
每批一次 Python 随机抽样，以 0.8 概率用预测颜色图、0.2 概率用
GT / (GT 通道和 + 0.0001) 作为先验均值。
不添加重建、L1、颜色或 TV 惩罚。GT 去量化默认关闭，与官方实际训练调用一致。

推理确定性地解码 squeeze 后的预测颜色图，不使用零或随机 latent。
默认补边匹配官方 test_unpaired.py：OpenCV BORDER_REFLECT，每轴增加
16 - size % 16 像素，已经整除时仍增加完整的 16 像素。先补边再均衡，
最终裁回原尺寸。模型原始输出不裁剪数值范围，Predictor 负责图像转换。

成对前向先编码 GT 以初始化 ActNorm，再在 no_grad 下解码图像满足输出
契约，NLL 保留梯度。加载权重后，即使 ActNorm bias 为零也保留已加载统计量。
可逆卷积遇到奇异权重抛出 RuntimeError，避免无限重试。

## 参数

构造器：`LLFlow(config=None, **kwargs)`。合并顺序：默认值、config、kwargs。
未知键保留但不使用；已移除的旧参数明确抛出 ValueError。

| 参数 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| config | dict 或 None | None | 其他类型抛出 TypeError。 |
| model_name | str | "LLFlow" | 基类名称元数据。 |
| input_channels | int | 3 | 必须为 3；forward 另可接收六通道预处理输入。 |
| save_dir | str | "./checkpoints/llie/LLFlow" | 基类保存目录。 |
| nf | int | 64 | 编码器宽度，必须为 64。 |
| nb | int | 24 | RRDB 数量，整数 >= 8，不接受 bool。 |
| K | int | 12 | 每级耦合步骤数，整数 >= 1，不接受 bool。 |
| L | int | 3 | flow 级数，必须为 3。 |
| train_gt_ratio | float | 0.2 | 转成 float 后有限且在 [0,1] 内。 |
| quant | float | 32 | 正有限量化数，仅开启 GT 噪声时使用。 |
| inference_padding | str | "reflect16" | "reflect16" 或 "none"；成对调用从不补边。 |
| mode | str | "inference" | "train" 必须传 paired_image；"inference" 允许普通增强。 |

已移除：condition_channels、condition_blocks、flow_layers、
flow_hidden_channels、scale_clamp、sample_temperature。旧简化模型权重不兼容。
修改 nb/K 会改变架构；nb=8/K=1 仅适合轻量测试，既不是官方 smallNet 也不是标准版。

### 前向与权重加载

`forward(x, paired_image=None, *, add_gt_noise=False)`：

| 参数 | 类型 | 默认值 | 契约 |
| --- | --- | --- | --- |
| x | Tensor | 必填 | 浮点 BCHW，RGB 范围 [0,1]，或六通道预处理输入；高宽为正。 |
| paired_image | Tensor 或 None | None | RGB GT，批次/高宽匹配；成对高宽整除 8。 |
| add_gt_noise | bool | False | 加入 (-0.5,0.5)/quant 均匀 GT 噪声及量化 logdet 校正。 |

无 GT 返回原尺寸 BCHW RGB；有 GT 返回
`{"pred": Tensor, "aux": {"nll": B维向量, "latent": Tensor, "logdet": B维向量}, "meta": {"mode": str}}`。
inference_padding="none" 时，非成对高宽也必须整除 8。
直接推理应调用 eval()，Predictor 会自动设置。

- `rrdbPreprocessing(low)`：六通道预处理输入转换为特征字典。
- `encode(target, condition, *, add_gt_noise=False)`：RGB GT 和特征字典
  转换为 (latent, 每样本 NLL, logdet)。
- `decode(condition)`：特征字典转换为从预测先验均值解码的 RGB。
- `load_official_weights(path)`：读取本地生成器纯 state_dict，移除可选
  module. 前缀，严格加载并返回 self，不自动下载。

损失注册名 llflow，别名 llflow_loss、low_light_flow、normalizing_flow_loss。
构造器无模型专用参数。
`LLFlow_Loss()(model_output, target)` 与 Trainer compute 接口均归约 aux.nll；
缺少 GT 或 NLL 格式错误时抛出 ValueError。

## 使用示例

```python
import openLLV as llv
from openLLV.deepLearning.models.LLIE.LLFlow import LLFlow

model = LLFlow().load_official_weights("pretrained/39000_G.pth").eval()
checkpoint = model.save_model("checkpoints/llflow")
enhanced, saved_path = llv.predict(
    checkpoint, "input.jpg", output="results/llflow/output.png",
)
```

```python
import openLLV as llv

llv.train("LLFlow", root_dir="datasets/LOL", output_dir="runs/llflow")
```

YAML 使用 LLFlowDataset、160 像素裁剪、水平翻转、batch 16、
Adam lr=0.0005 和 AMP。显存不足时降低 batch_size，并重新计算 epoch 调度。

## 保持现有集成接口不变时的限制

| 官方行为 | openLLV 集成 |
| --- | --- |
| 40,000 iteration；首次迭代跳过优化 | 完整 epoch、每批优化。YAML 为 1,334 epoch；485 张图、batch 16、drop_last=True 时共 40,020 次更新。 |
| 3,000 iteration warmup；20k/30k/36k/38k 衰减 | 无逐 update 调度钩子。YAML 近似为 epoch 667/1000/1200/1267，gamma=0.5，无 warmup。改变数据量/batch 后须重算。 |
| 每 200 iteration 验证、1,000 iteration 保存 | 只能按 epoch 配置；YAML 每 epoch 验证，采用 Trainer 的 best/last 保存。 |
| 训练 batch 16，验证 batch 1 | Trainer 共用一个 batch_size；全分辨率验证可能明显增加显存占用，必要时降低共用 batch。 |
| 报告 PSNR 时用 GT 调整亮度、自定义 SSIM、最佳 PSNR 权重 | Trainer 验证 NLL 并按最佳 loss 保存；Predictor 无 GT 亮度校正，通用指标不能直接与官方结果比较。 |
| 分开的生成器与优化器/调度器文件 | 支持生成器转换；官方 optimizer/scheduler resume 文件不能直接交给 Trainer。 |
| DataParallel 特有参数分组 | YAML 对齐官方实际单 GPU Adam：betas=(0.9,0.999)、无 decay。官方 DataParallel 会将 RRDB 分到单独 1e-5 decay 组；Trainer 的 YAML 优化器构建不支持按参数名分组。 |

官方 beta1/beta2 是未被使用的参数组键，实际 Adam 使用 (0.9,0.999)。
train_RRDB=false、train_RRDB_delay 在官方标准执行路径中也未冻结编码器，
本实现从开始就训练编码器。标准架构与目标函数已对齐，完整训练/评估驱动
没有完全复刻；这些差异使得本实现不能承诺复现论文指标。

## 权重与许可证

保留 RRDB.*、flowUpsamplerNet.* 参数名和形状，包括官方未使用参数，
以支持严格加载生成器权重。应使用 LOL-pc 权重，而非 LOL_smallNet。
随机初始化可用于训练，有意义的推理需要训练好的权重。
官方代码保留 CC BY-NC-SA 4.0 及其组件许可证，见 _llflow/NOTICE.md 和 licenses/。
