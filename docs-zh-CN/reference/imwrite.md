# openLLV.imwrite()

`openLLV.imwrite()` 把图像类值转换为 PIL 后保存，并返回解析后的输出 `Path`。`openLLV.write_image()` 是同一函数。

## Function Form

```python
openLLV.imwrite(
    image,
    output=None,
    *,
    save_format=None,
    output_name=None,
    **kwargs,
)
```

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `image` | `str`、`Path`、`bytes`、`bytearray`、`numpy.ndarray`、`PIL.Image.Image` 或 `torch.Tensor` | 必填 | `ImageWriter`/`ImageReader` 接受的图像数据；三通道 NumPy 和 tensor 值使用 RGB 顺序。 | Tensor 形状为 `[H,W]`、`[C,H,W]` 或 `[1,C,H,W]`；batch size 大于 1 抛 `ValueError`。 |
| `output` | `Optional[Union[str, Path]]` | `None` | 带后缀时为精确文件路径，否则为输出目录。 | `None` 使用 `results/`。 |
| `save_format` | `Optional[str]` | `None` | 仅关键字的后缀/格式覆盖，可带或不带前导点；也会覆盖显式输出文件的后缀。 | 空字符串抛 `ValueError`。 |
| `output_name` | `Optional[str]` | `None` | `output` 为目录或省略时使用的仅关键字文件名。 | 省略时尽量保留来源文件名，否则使用 `image.png`。 |
| `ext` | `Optional[str]` | `None` | 转换期间经 `**kwargs` 转发给 `ImageReader` 的来源编码扩展名。 | 适合 bytes/base64 来源。 |
| `timeout` | `float` | `10` | 经 `**kwargs` 转发给 `ImageReader` 的 URL 超时。 | 用于远程输入。 |
| `headers` | `Dict[str, str]` | 类浏览器 User-Agent 字典 | 经 `**kwargs` 转发的 HTTP 请求头。 | 合并进 reader 默认值。 |
| `verify_ssl` | `bool` | `True` | 经 `**kwargs` 转发的 TLS 校验选项。 | 适用于通过 `requests` 下载 URL。 |

### Aliases

| Alias | Points to |
| --- | --- |
| `openLLV.write_image()` | `openLLV.imwrite()` |

## Returns

最终保存位置的 `pathlib.Path`。父目录会自动创建。

## Behavior Details

- `output` 已存在且是文件，或自身带后缀时，按文件路径处理；否则按目录处理。
- `save_format` 在路径解析后替换后缀。
- JPEG 输出会先把非 `RGB`/`L` 模式图像转为 RGB。
- `[0,1]` tensor 浮点值缩放至 `[0,255]`，所有值裁剪为 `uint8`。
- 三通道 NumPy 输入按 RGB 解释；OpenCV 所需的 BGR 转换仅发生在内部编码阶段。

### Raises

| Exception | Condition |
| --- | --- |
| `ValueError` | `save_format` 为空、tensor 形状/batch 不支持或输入转换失败。 |
| `FileNotFoundError` | 路径类来源图像不存在。 |

## Examples

```python
import openLLV as llv

saved = llv.imwrite("input.png", "results/copy.jpg")
```

```python
saved = llv.imwrite(
    image_array,
    output="results/exports",
    output_name="enhanced.png",
    save_format="webp",
)
```

## Related

- 读图像：[`openLLV.imread()`](imread.md)
