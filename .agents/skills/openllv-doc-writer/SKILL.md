---
name: openllv-doc-writer
description: 为 openLLV 项目编写"给 AI 看的"单功能文档（原子级：一份文档 = 一个公开功能），输出到 docs/reference/（英文）与 docs-zh-CN/reference/（中文）。当用户要求编写某个 openLLV 公开方法、模型或算法的功能文档时使用。强制先在源码中验证每个参数名、默认值、类型与行为，再对照模板输出文档；所有参数（含经 **kwargs 转发的键）必须逐个列出。
---

# openLLV 单功能文档编写

为 openLLV 编写"给 AI 看的"功能文档。核心原则：

1. **原子性**：一份文档只描述**一个**公开功能（一个公开方法，或一个模型/算法组件）。禁止一份文档混写多个功能，禁止顺手修改无关文件。
2. **源码优先**：文档中出现的每个参数名、默认值、类型、行为都必须从源码中读取并验证，**禁止凭记忆或凭旧文档杜撰**。旧文档只作风格参考，不作内容依据。
3. **机器可读**：文档是给 AI（以及追求精确的人）看的，不是营销文案。参数表必须完整、精确、无歧义：列出**所有**可接受参数（包括通过 `**kwargs` 转发的），注明类型、默认值、取值范围/约束、别名与错误行为。

## 输出位置

所有本 skill 生成的文档写入 **`docs/reference/`**（英文）与 **`docs-zh-CN/reference/`**（中文），两者一一对应：

| 功能类别                 | 输出路径                                                 | 示例                                                   |
| ------------------------ | -------------------------------------------------------- | ------------------------------------------------------ |
| 顶层 API 方法            | `docs/reference/<方法名>.md`                             | `docs/reference/predict.md`、`docs/reference/train.md` |
| 统一/后端 Predictor 方法 | `docs/reference/predict.md`（与 `predict` 同文件分节）   | —                                                      |
| 图像 I/O 方法            | `docs/reference/imread.md` / `docs/reference/imwrite.md` | —                                                      |
| 模型组件                 | `docs/reference/models/<name>.md`                        | `docs/reference/models/zero-dce.md`                    |
| 算法组件                 | `docs/reference/algorithms/<name>.md`                    | `docs/reference/algorithms/clahe.md`                   |

- 方法名小写、保持源码中的拼写；组件名用小写连字符（`ZeroDCE++` → `zero-dce++.md` 或 `zero-dce.md`，与 README/现有 docs 命名习惯一致）。
- 用户未指定语言时，默认英文 + 中文两份都写；只写一份时在最终回复中说明。
- `docs/reference/` 目录已存在但为空；`docs-zh-CN/reference/` 若不存在则创建。

## 功能范围（feature 的定义）

本 skill 把 openLLV 的"一个公开功能"定义为以下任一单元，每单元对应一份文档：

| 类别                | 示例                                                                                                                                                  |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 顶层 API 方法       | `openLLV.predict()`、`openLLV.train()`、`openLLV.evaluate()`、`openLLV.imread()`、`openLLV.imwrite()`、`openLLV.list_*()`、`openLLV.list_available()` |
| 统一 Predictor 方法 | `Predictor.__call__` / `predict` / `predict_single` / `predict_batch` / `get_params`                                                                  |
| 训练 / 评估         | `Trainer` 构造与 `train()`、`Evaluator` 构造与 `eval()`                                                                                               |
| 深度学习模型组件    | `ZeroDCE`、`CIDNet` 等 `LLVModel` 子类（含构造 config 键）                                                                                            |
| 传统算法组件        | `HE`、`CLAHE`、`DCP` 等 `LLVEnhancer` 子类（含构造参数）                                                                                              |

**不要**单独建文档：私有方法（下划线开头）、基类内部实现细节（可在相关文档"实现说明"中简述）、CLI 子命令（已有 `docs/usage/cli.md` 单独维护）。

## 执行流程（必须按顺序执行）

### 第 1 步：定位功能与其源码

- 先读 `references/api-map.md`，里面有已核实的公开 API 签名、源码位置和 kwargs 路由表；功能不在表中时用 `grep` 在 `openLLV/` 下按名字搜索（类名 / 注册名 / 方法名）。
- 记录该功能的完整调用链。例：`openLLV.predict` → `openLLV/api.py` → `Predictor`（`openLLV/predictor.py`）→ 后端 Predictor（`openLLV/deepLearning/predictor.py` 或 `openLLV/tradition/predictor.py`）→ 组件构造器（`LLVModel` / `LLVEnhancer` 子类）。
- 找到组件实现文件与注册名：模型在 `openLLV/deepLearning/models/<task>/`，算法在 `openLLV/tradition/algorithms/<group>/`；注册名/别名在类属性 `name` / `aliases` 中。

### 第 2 步：枚举全部可接受参数（核心，强制）

对一个公开方法，参数 = **显式签名参数** + **`**kwargs` 实际消费的键**。逐个展开：

1. **显式参数**：从源码签名读取每个参数的名字、默认值、类型注解。注意 `*`（keyword-only）、`*args` 的存在。
2. **`**kwargs` 路由追踪**：找到 kwargs 被转交的下一层，逐层展开，直到消费点（构造器、`forward`、`_enhance`、`create_metric` 等），记录每一层支持的键。
   - 例：`openLLV.predict(**kwargs)` 在 `api.py` 用 `_PREDICT_CALL_KWARGS` 拆分：`progress_bar`、`output_name`、`output_ext`、`save`、`model_kwargs`、`ext`、`timeout`、`headers`、`verify_ssl` 归调用参数；其余进 `Predictor` 构造器（`backend`、`output_dir`、`config`、`device`、`transform`、`batch_size`、`num_workers` 等），再继续转交组件构造器。
   - 例：`openLLV.train(**kwargs)` 由 `Trainer._kwargs_to_config` 的 `flat_map` 决定支持哪些平铺键，未知键抛 `TypeError`。
3. **组件参数**：模型/算法组件的参数 = 其构造器的显式参数 + 基类构造器参数（`LLVEnhancer` 的 `output_type`、`keep_dtype`、`clip_output`、`value_range`；`LLVModel` 的 `config`）+ `_init_model()` 实际读取的 config 键。
4. **校验规则提取**：阅读构造器内验证代码，记录约束。例：CLAHE 的 `clip_limit` 必须是大于 0 的有限数；`tile_grid_size` 必须含两个正整数。约束必须写进参数表。
5. **运行时确认（可选但推荐）**：openLLV 可在本仓库直接导入，用 `python3` 核对签名与默认值：
   ```bash
   python3 -c "import inspect, openLLV; print(inspect.signature(openLLV.predict))"
   ```
   静态读码是权威来源；运行时 introspection 用于二次确认，冲突时以源码为准并说明。

### 第 3 步：按模板撰写文档到 `docs/reference/`

- 公开方法文档用 `templates/method.md`；模型/算法组件文档用 `templates/component.md`。
- 可快速浏览 `references/doc-style.md` 了解风格细则，以及现有 `docs/guide/*.md`、`docs/models/**/*.md` 作为风格范例（仅参考风格，**不复制其内容**）。
- 中文版与英文版内容一一对应。

### 第 4 步：自查清单（交付前必过）

- [ ] 参数表覆盖了签名中每个显式参数 + 每个经 kwargs 消费的键？无遗漏、无杜撰？
- [ ] 每个默认值、类型、约束都来自源码（能指出具体文件与行号）？
- [ ] 所有示例代码的调用方式与参数真实存在？（可用 `python3` 实际跑一遍示例）
- [ ] 返回值契约写清楚了（单图返回 `(增强结果, Path|None)`，目录输入返回路径列表）？
- [ ] 别名与大小写/标点规则写清楚了（注册名大小写不敏感；`en`/`ref` 是 `en_img_dir`/`ref_img_dir` 的别名）？
- [ ] 中文版与英文版内容一致？
- [ ] 没有改动与本功能无关的文件？
- [ ] 文档已放在 `docs/reference/`（或 `docs-zh-CN/reference/`）而非别处？

## 写作风格要求（给 AI 看的关键）

- **完整优先**：宁可表格长，不可漏参数。`**kwargs` 转发的键必须显式列出，不能只写"其他参数见 XXX"。
- **精确表述**：默认值写成字面量（`"yuv"`、`(8, 8)`、`2.0`、`None`），不用"大约""默认即可"这类模糊词。
- **行为契约**：写清"什么输入 → 什么输出"。例如：单图返回 `(image, path)`；`save=False` 时 `path` 为 `None`；目录输入递归处理并返回按路径排序的 `Path` 列表；`backend="auto"` 时 `.pt`/`.pth` 文件走深度后端。
- **约束与异常**：记录会抛出的 `ValueError`/`TypeError` 场景（如 `train` 未知平铺键抛 `TypeError`；`predict` 双重注册名需显式 `backend`）。
- **代码示例**：每个参数表后配 1–2 个真实可运行的示例，优先展示非默认参数。
- **引用源码位置**：组件文档保留"Location in openLLV"表格（实现文件、类名、注册名、基类），方便溯源。

## 参考文件

- `references/api-map.md` — 公开 API 索引：已核实的签名、源码位置、kwargs 路由表、注册名规则。
- `references/doc-style.md` — 文档风格细则与参数表写法。
- `templates/method.md` — 公开方法文档模板。
- `templates/component.md` — 模型/算法组件文档模板。
