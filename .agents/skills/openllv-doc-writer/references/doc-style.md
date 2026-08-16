# 文档风格细则（docs/reference 专用）

本文件定义 `docs/reference/` 下功能文档的写作规范。风格参考现有 `docs/guide/*.md`、`docs/models/**/*.md`、`docs/algorithms/**/*.md`，但**内容一律以源码为准**，不复制旧文档内容。

## 总体要求

- 文档给 AI 消费：信息密度高、无歧义、无宣传性语言（"强大""先进"这类词不要出现在行为描述里，只可出现在简短背景介绍中）。
- 每个功能文档**独立可读**：不依赖读者先读其他文档；确需引用时用相对链接（如 `../guide/overview.md`）。
- 中英文两份内容逐节对应，翻译要准确（术语表：enhanced image=增强图、checkpoint=检查点、registered name=注册名、alias=别名、forward pass=前向、predictor=预测器、evaluator=评估器）。

## 标题与元信息

- 一级标题用功能名（方法文档用签名式标题，组件文档用组件名）。
- 方法文档：`# openLLV.predict()` 或 `# Predictor.predict_single()`。
- 组件文档：`# CLAHE` / `# Zero-DCE`，标题下用一行 `> Task: <task>` 或 `> Documentation group: <group>` 标注归属。

## 方法文档结构（对应 templates/method.md）

1. **一句话概述**：该方法做什么、返回什么。
2. **Function Form**：完整签名代码块，含所有显式参数与默认值。
3. **参数表**（核心，见下节）。
4. **返回契约**：`Returns` 描述，明确各分支输出。
5. **行为细节**：路由规则、别名、推断规则、异常（用 `Raises` 分条）。
6. **示例**：2–4 个可运行代码块，覆盖典型用法与非默认参数。
7. **相关链接**：指向顶层 API 文档、组件文档。

## 组件文档结构（对应 templates/component.md）

1. **概述**：一句话 + 该组件在 openLLV 中的角色。
2. **Links** 表：Paper / 官方源码 / 官方项目页（无则 `None`）。
3. **Location in openLLV** 表：实现文件、类名、注册名、基类（模型另加默认配置 YAML、关联损失）。
4. **实现说明**：关键行为（如 HE 的色空间处理、CIDNet 的 8 对齐 padding）、训练/推理差异。
5. **参数表**：构造参数 + 基类参数 + 模型 config 键，逐个列出。
6. **Usage Example**：`llv.predict(...)` 预测示例；必要时加训练示例。
7. **（可选）官方权重 / checkpoint 说明**。

## 参数表写法（最重要）

| 列 | 要求 |
| --- | --- |
| Parameter | 参数名，用反引号包裹 |
| Type | 源码类型注解或实测类型（`str`、`int`、`float`、`bool`、`Tuple[int, int]`、`Optional[str]`、`Dict[str, Any]` 等） |
| Default | 字面量默认值（`"yuv"`、`(8, 8)`、`2.0`、`None`、`False`） |
| Meaning | 一句话含义 |
| Constraints（如有） | 校验规则：取值集合、范围、抛出的异常类型与消息要点 |

规则：

- **显式参数**与 **kwargs 转发键**统一放进同一张表；kwargs 转发键在 Meaning 里注明"经 `**kwargs` 转发至 `<目标>`"。
- 每个默认值、类型必须能对应到源码中的字面量。拿不准就用 `python3` introspection 或直接读源码确认，不要写"见源码"。
- 有校验逻辑的参数必须写明约束，例如：
  - `clip_limit`：必须为大于 0 的有限数，否则抛 `ValueError`。
  - `tile_grid_size`：二元组/列表，两个正整数，否则抛 `ValueError`。
  - `batch_size`：正整数，否则抛 `ValueError`。
  - `train(**kwargs)` 未知键：抛 `TypeError: Unsupported Trainer argument: <key>`。
- 别名单独成小节或用表注说明（如 `en`/`ref`、`read_image`/`write_image`、注册名大小写不敏感）。

## 返回契约写法

- 单图：`(image, path)`，其中 `image` 深度后端为 `PIL.Image.Image`、传统后端为 `numpy.ndarray`；`save=False` 时 `path` 为 `None`。
- 目录：`save=True` 返回按源路径排序的 `Path` 列表；`save=False` 返回按相同顺序排列的增强图列表且不创建输出文件/目录。递归处理并保留相对子目录；未指定 `output_ext` 时逐字符保留源文件名与后缀（含大小写），指定时只替换后缀；目录传 `output_name` 抛 `ValueError`。
- 异常：`TypeError` / `ValueError` / `FileNotFoundError` / `NotADirectoryError` 各自触发条件。

## 示例代码要求

- 使用 `import openLLV as llv` 风格。
- 示例必须真实可运行（参数真实存在、路径写法合理）。写完后建议用 `python3` 至少验证一个最小示例不抛 `ImportError`/`AttributeError`。
- 展示非默认参数优先（如 `color_space="hsv"`、`model_kwargs={"strength": 0.8}`、`backend="traditional"`）。

## 语言对照要点

- 中文版与英文版节标题一致；中英参数表同列同值。
- 中文版可在句末保留英文术语原文（如"注册名（registered name）"），便于 AI 检索。
