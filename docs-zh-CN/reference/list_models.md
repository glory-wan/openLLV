# openLLV.list_models()

`openLLV.list_models()` 返回每个可接受的已注册深度学习模型查找名，包括别名。

## Function Form

```python
openLLV.list_models() -> List[str]
```

## Parameters

该函数没有参数。

## Returns

排序后的 `List[str]`，包含具体 `LLVModel` 类及别名的规范化注册表键。注册时执行 `strip().lower()`，因此返回键为小写。调用前会导入 openLLV 模型包。

## Behavior Details

- 模型查找去除首尾空白后不区分大小写。
- 带别名的实现会贡献多个查找字符串；若需要每个类一条去重记录，使用 `list_available()`。

## Example

```python
import openLLV as llv

if "zerodce" in llv.list_models():
    enhanced, path = llv.predict("ZeroDCE", "input.jpg")
```

## Related

- 去重组件元数据：[`openLLV.list_available()`](list_available.md)
- 传统算法：[`openLLV.list_algorithms()`](list_algorithms.md)
