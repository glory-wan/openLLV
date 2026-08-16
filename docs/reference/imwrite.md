# openLLV.imwrite()

`openLLV.imwrite()` converts an image-like value to PIL and saves it, returning the resolved output `Path`. `openLLV.write_image()` is the same function.

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
| `image` | `str`, `Path`, `bytes`, `bytearray`, `numpy.ndarray`, `PIL.Image.Image`, or `torch.Tensor` | required | Image data accepted by `ImageWriter`/`ImageReader`; three-channel NumPy and tensor values use RGB order. | Tensor shapes are `[H,W]`, `[C,H,W]`, or `[1,C,H,W]`; batch size greater than 1 raises `ValueError`. |
| `output` | `Optional[Union[str, Path]]` | `None` | Exact file path when it has a suffix, otherwise an output directory. | `None` uses `results/`. |
| `save_format` | `Optional[str]` | `None` | Keyword-only suffix/format override, with or without a leading dot. It overrides an explicit output-file suffix too. | Empty strings raise `ValueError`. |
| `output_name` | `Optional[str]` | `None` | Keyword-only filename used when `output` is a directory or omitted. | If omitted, a source filename is preserved where possible; otherwise `image.png` is used. |
| `ext` | `Optional[str]` | `None` | Source encoding extension forwarded via `**kwargs` to `ImageReader` during conversion. | Useful for bytes/base64 sources. |
| `timeout` | `float` | `10` | URL timeout forwarded via `**kwargs` to `ImageReader`. | Applies to remote input. |
| `headers` | `Dict[str, str]` | browser-like User-Agent dictionary | HTTP headers forwarded via `**kwargs`. | Merged into reader defaults. |
| `verify_ssl` | `bool` | `True` | TLS verification forwarded via `**kwargs`. | Applies to URL downloads through `requests`. |

### Aliases

| Alias | Points to |
| --- | --- |
| `openLLV.write_image()` | `openLLV.imwrite()` |

## Returns

The final saved `pathlib.Path`. Parent directories are created automatically.

## Behavior Details

- If `output` exists as a file or has a suffix, it is treated as a file path. Otherwise it is a directory.
- `save_format` replaces the suffix after path resolution.
- JPEG output converts images outside `RGB`/`L` mode to RGB before saving.
- Tensor floats in `[0,1]` are scaled to `[0,255]`; values are clipped to `uint8`.
- Three-channel NumPy input is interpreted as RGB; OpenCV BGR conversion is internal to encoding.

### Raises

| Exception | Condition |
| --- | --- |
| `ValueError` | Empty `save_format`, unsupported tensor shape/batch, or input conversion failure. |
| `FileNotFoundError` | A path-like source image does not exist. |

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

- Reading images: [`openLLV.imread()`](imread.md)
