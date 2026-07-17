# Image I/O API

openLLV exposes `ImageReader` and `ImageWriter` through compact top-level helpers.

## Read Images

```python
import openLLV as llv

pil_image = llv.imread("input.jpg")
bgr_array = llv.imread("input.jpg", output_format="numpy")
```

Accepted input types are:

- local file path or `Path`;
- HTTP/HTTPS URL;
- base64 string;
- `bytes` or `bytearray`;
- PIL image;
- NumPy array;
- single-image PyTorch tensor with `[H, W]`, `[C, H, W]`, or `[1, C, H, W]` shape.

URL options such as `timeout`, `headers`, and `verify_ssl` are forwarded to the reader.

## Output Formats

| `output_format` | Result |
| --- | --- |
| `"pil"` | RGB `PIL.Image.Image` |
| `"numpy"` | OpenCV-style BGR `numpy.ndarray` |
| `"bytes"` | Encoded image bytes |
| `"base64"` | Base64-encoded image string |
| `"file"` | Path string for a temporary image file |

`tensor` is an accepted input type, not an `ImageReader` output format. Convert a PIL or NumPy result to a tensor with the transform used by your model or dataset.

For byte/base64 output when the source has no detectable suffix, provide `ext`:

```python
encoded = llv.imread(image_array, output_format="bytes", ext="png")
```

## Inspect Input Metadata

```python
from openLLV.data import ImageReader

reader = ImageReader()
info = reader.get_info("input.jpg")
print(info)
```

## Save Images

Save to an exact file:

```python
saved_path = llv.imwrite(image, "results/output.png")
```

Save to a directory and choose a name:

```python
saved_path = llv.imwrite(
    image,
    output="results/",
    output_name="enhanced",
    save_format="jpg",
)
```

When `output` is omitted, `ImageWriter` uses `results/`. A source filename is preserved when it can be inferred; otherwise a generated name is used.

## Aliases

`read_image()` is an alias of `imread()`, and `write_image()` is an alias of `imwrite()`.

