<!--
openLLV 模型/算法组件文档模板。
使用前：
  1. 读 SKILL.md 第 1-2 步，在源码中核实：实现文件、类名、注册名/别名、
     构造参数（含基类参数）与校验规则、模型 _init_model 读取的 config 键。
  2. 所有 <...> 占位符替换为真实内容；未涉及的章节删除。
  3. 英文版写入 docs/reference/models/<name>.md 或 docs/reference/algorithms/<name>.md；
     中文版写入 docs-zh-CN/reference/ 对应位置。
-->

# <组件名>

> Task: <task>（或 > Documentation group: <group>）

<!-- 一句话概述：该组件的性质（模型/算法）与在 openLLV 中的角色。 -->

<概述>。

## Links

| Type | URL |
| --- | --- |
| Paper | <URL 或 None> |
| Official source code | <URL 或 None> |
| Official project page | <URL 或 None> |
| Default configuration | <YAML 路径或 None>（仅模型） |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `<openLLV/... 相对路径>` |
| Class name | `<类名>` |
| Registered name | `<注册名>`（别名：<别名列表>） |
| Base class | `<基类名>` 于 `<基类文件>` |
| Related loss | `<损失文件路径>`（仅模型，且存在时） |

## Implementation Notes

<!-- 关键行为，从源码读取：
- 算法：支持的输入/色空间、处理流程、边界情况。
- 模型：训练/推理输出差异、padding 策略、config 读取位置、依赖下载等。 -->

<实现说明>。

## Parameters

<!--
参数表：构造参数 + 基类参数 + 模型 config 键，逐个列出，来源必须是源码。
列：Parameter | Type | Default | Meaning | Constraints
基类参数（如 output_type / keep_dtype / clip_output）也列出，并注明属于基类。
模型参数以 config 键形式给出（如 input_gamma: float = 1.0）。
-->

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `<参数>` | `<类型>` | `<默认值>` | `<含义>`（基类参数：`LLVEnhancer`） |
| `<config 键>` | `<类型>` | `<默认值>` | `<含义>`（存于 `model.config`） |

## Usage Example

<!-- 预测示例（必写）；训练示例（模型可选）。优先展示非默认参数。 -->

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "<注册名>",
    "input.jpg",
    output="results/<名称>/output.png",
    <参数>=<非默认值>,
)
```

<!-- 文件夹批量处理示例（算法）或训练示例（模型），可选。 -->

```python
# <示例说明>
<示例代码>
```

## Checkpoint / Official Weights

<!-- 模型可选：openLLV checkpoint 用法、官方权重加载方式（需读源码确认模型是否保持官方参数名）。 -->

<说明>。
