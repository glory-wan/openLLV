# openLLV.list_losses()

`openLLV.list_losses()` 返回每个已注册深度学习 loss 查找名及别名。

## Function Form

```python
openLLV.list_losses() -> List[str]
```

## Parameters

该函数没有参数。

## Returns

排序后的 `List[str]`，包含 `BaseLoss` 注册表接受的规范化小写键：类名、声明的规范 `name` 及别名。

## Behavior Details

- 注册表查找不区分大小写。
- 结果是查找键列表，而不是按类去重的列表；规范类名及其别名数组请用 `list_available()`。

## Example

```python
import openLLV as llv

print(llv.list_losses())
```

## Related

- 训练：[`openLLV.train()`](train.md)
- 去重组件元数据：[`openLLV.list_available()`](list_available.md)
