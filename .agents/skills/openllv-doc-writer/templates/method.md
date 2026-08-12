<!--
openLLV 公开方法文档模板。
使用前：
  1. 读 SKILL.md 第 1-2 步，在源码中核实签名、默认值与 kwargs 路由。
  2. 所有 <...> 占位符替换为真实内容；未涉及的章节删除。
  3. 英文版写入 docs/reference/<方法名>.md，中文版写入 docs-zh-CN/reference/<方法名>.md。
-->

# openLLV.<方法名>()

<!-- 一句话概述：该方法做什么、返回什么。 -->

<概述>。

## Function Form

```python
<!-- 完整签名，含所有显式参数与默认值（从源码签名复制）。 -->
openLLV.<方法名>(<参数1>, <参数2>=<默认值>, **kwargs)
```

<!-- 必要时的说明：参数类别（位置/关键字）、别名、config 形式等。 -->

## Parameters

<!--
参数表：显式参数与 **kwargs 转发键全部列出。
列：Parameter | Type | Default | Meaning | Constraints（Constraints 无内容可省略该列）
每个值必须来自源码；kwargs 转发键在 Meaning 注明"经 **kwargs 转发至 <目标>"。
-->

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `<参数1>` | `<类型>` | `<默认值>` | `<含义>` |
| `<kwargs 键>` | `<类型>` | `<默认值>` | `<含义>（经 **kwargs 转发至 <目标>）` |

<!-- 别名单独小节。 -->

### Aliases

| 别名 | 指向 |
| --- | --- |
| `<别名>` | `<原参数/方法>` |

## Returns

<!-- 明确各分支输出，例如：
单图返回 `(image, path)`；`save=False` 时 `path` 为 `None`。
目录输入返回按路径排序的 `Path` 列表。 -->

<返回契约描述>。

## Behavior Details

<!-- 行为细节：路由规则、推断规则、校验与异常。逐条列出。 -->

- <规则 1>。
- <规则 2>。
- ...

### Raises

| Exception | Condition |
| --- | --- |
| `<TypeError>` | `<触发条件>` |
| `<ValueError>` | `<触发条件>` |

## Examples

<!-- 2-4 个可运行示例，优先展示非默认参数。 -->

```python
import openLLV as llv

# <示例说明>
<示例代码>
```

```python
# <示例说明>
<示例代码>
```

## Related

<!-- 相关文档/组件链接。 -->

- 相关组件：<组件名>（`<文档路径>`）
- 顶层 API：<方法名>（`<文档路径>`）
