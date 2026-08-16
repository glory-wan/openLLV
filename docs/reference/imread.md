# openLLV.imread()

`openLLV.imread()` reads a local or remote image-like source and converts it to one of five output representations. `openLLV.read_image()` is the same function.

## Function Form

```python
openLLV.imread(source, output_format="pil", **kwargs)
```

## Parameters

| Parameter | Type | Default | Meaning | Constraints |
| --- | --- | --- | --- | --- |
| `source` | `str`, `Path`, `bytes`, `bytearray`, `numpy.ndarray`, `PIL.Image.Image`, or `torch.Tensor` | required | Image source consumed by `ImageReader`. Strings may be local paths, `http`/`https`/`ftp`/`file` URLs, base64 text, or data URIs. | Missing path-like inputs raise `FileNotFoundError`; unconvertible inputs raise `ValueError`. |
| `output_format` | `str` | `"pil"` | Output representation: `"pil"` (RGB), `"numpy"` (RGB), `"bytes"`, `"base64"`, or `"file"` (temporary-file path). | Matched case-insensitively; any other value raises `ValueError`. |
| `ext` | `Optional[str]` | `None` | Explicit encoding extension, with or without a leading dot, forwarded through `**kwargs` to `ImageReader`. It is mainly useful for bytes/base64 inputs and encoded outputs. | A string is lowercased and stripped of its leading dot. If omitted, the reader detects the extension or falls back to `"jpg"`. |
| `timeout` | `float` | `10` | Network timeout in seconds, forwarded through `**kwargs`. | Used for URL inputs. |
| `headers` | `Dict[str, str]` | browser-like User-Agent dictionary | HTTP headers merged into the reader defaults, forwarded through `**kwargs`. | Must support dictionary update semantics. |
| `verify_ssl` | `bool` | `True` | Whether `requests` verifies TLS certificates, forwarded through `**kwargs`. | Applies to URL downloads through `requests`. |

### Aliases

| Alias | Points to |
| --- | --- |
| `openLLV.read_image()` | `openLLV.imread()` |

## Returns

- `output_format="pil"`: `PIL.Image.Image` in RGB.
- `output_format="numpy"`: `numpy.ndarray` in RGB order.
- `output_format="bytes"`: encoded `bytes`.
- `output_format="base64"`: encoded base64 `str`.
- `output_format="file"`: path `str` for a newly created temporary file. The caller owns cleanup of this file.

## Behavior Details

- Every public three-channel input and output uses RGB order. `ImageReader` does not expose OpenCV-style BGR arrays.
- Tensor input accepts `[H,W]`, `[C,H,W]`, or `[1,C,H,W]`. Floating values in `[0,1]` are scaled to `[0,255]`; all values are clipped to `uint8`.
- A four-dimensional tensor must have batch size `1`.
- URL failures are wrapped in `ValueError` when `requests` is available.

### Raises

| Exception | Condition |
| --- | --- |
| `FileNotFoundError` | A path-like local image source does not exist. |
| `ValueError` | Unsupported output format, unsupported tensor shape/batch, failed URL download, or input cannot be converted. |

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

- Writing images: [`openLLV.imwrite()`](imwrite.md)
- Prediction: [`openLLV.predict()`](predict.md)
