# openLLV.imread()

`openLLV.imread()` 读取本地或远程图像类来源，并转换为五种输出表示之一。`openLLV.read_image()` 是同一函数。

## Function Form

```python
openLLV.imread(source, output_format="pil", **kwargs)
```

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `source` | `str`、`Path`、`bytes`、`bytearray`、`numpy.ndarray`、`PIL.Image.Image` 或 `torch.Tensor` | 必填 | `ImageReader` 消费的图像来源。字符串可为本地路径、`http`/`https`/`ftp`/`file` URL、base64 文本或 data URI。 | 不存在的路径类输入抛 `FileNotFoundError`；无法转换的输入抛 `ValueError`。 |
| `output_format` | `str` | `"pil"` | 输出表示：`"pil"`（RGB）、`"numpy"`（RGB）、`"bytes"`、`"base64"` 或 `"file"`（临时文件路径）。 | 匹配不区分大小写；其它值抛 `ValueError`。 |
| `ext` | `Optional[str]` | `None` | 经 `**kwargs` 转发给 `ImageReader` 的显式编码扩展名，可带或不带前导点，主要用于 bytes/base64 输入及编码输出。 | 字符串会转小写并去掉前导点；省略时自动检测，无法检测则为 `"jpg"`。 |
| `timeout` | `float` | `10` | 经 `**kwargs` 转发的网络超时秒数。 | 用于 URL 输入。 |
| `headers` | `Dict[str, str]` | 类浏览器 User-Agent 字典 | 经 `**kwargs` 转发并合并进默认值的 HTTP 请求头。 | 必须支持字典更新语义。 |
| `verify_ssl` | `bool` | `True` | 经 `**kwargs` 转发；使用 `requests` 时是否校验 TLS 证书。 | 适用于通过 `requests` 下载 URL。 |

### Aliases

| Alias | Points to |
| --- | --- |
| `openLLV.read_image()` | `openLLV.imread()` |

## Returns

- `output_format="pil"`：RGB `PIL.Image.Image`。
- `output_format="numpy"`：RGB 顺序的 `numpy.ndarray`。
- `output_format="bytes"`：编码后的 `bytes`。
- `output_format="base64"`：编码后的 base64 `str`。
- `output_format="file"`：新建临时文件的路径 `str`；调用方负责清理该文件。

## Behavior Details

- 所有公共三通道输入和输出统一使用 RGB 顺序；`ImageReader` 不会向外暴露 OpenCV 风格的 BGR 数组。
- Tensor 输入接受 `[H,W]`、`[C,H,W]` 或 `[1,C,H,W]`；`[0,1]` 浮点值缩放到 `[0,255]`，所有值裁剪为 `uint8`。
- 四维 tensor 的 batch size 必须为 `1`。
- 安装 `requests` 时，URL 下载失败会包装为 `ValueError`。

### Raises

| Exception | Condition |
| --- | --- |
| `FileNotFoundError` | 路径类本地图像来源不存在。 |
| `ValueError` | 输出格式不支持、tensor 形状/batch 不支持、URL 下载失败或输入无法转换。 |

## Examples

```python
import openLLV as llv

rgb = llv.imread("input.png", output_format="numpy")
```

```python
encoded = llv.imread(
    "https://example.com/image.png",
    output_format="bytes",
    timeout=20,
    verify_ssl=True,
)
```

## Related

- 写图像：[`openLLV.imwrite()`](imwrite.md)
- 预测：[`openLLV.predict()`](predict.md)
