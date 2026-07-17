# 图像输入输出 API

openLLV 通过简洁的顶层辅助函数公开 `ImageReader` 和 `ImageWriter`。

## 读取图像

```python
import openLLV as llv

pil_image = llv.imread("input.jpg")
bgr_array = llv.imread("input.jpg", output_format="numpy")
```

支持的输入类型包括：

- 本地文件路径或 `Path`；
- HTTP/HTTPS URL；
- Base64 字符串；
- `bytes` 或 `bytearray`；
- PIL 图像；
- NumPy 数组；
- 形状为 `[H, W]`、`[C, H, W]` 或 `[1, C, H, W]` 的单图像 PyTorch 张量。

`timeout`、`headers` 和 `verify_ssl` 等 URL 选项会转交给读取器。

## 输出格式

| `output_format` | 结果 |
| --- | --- |
| `"pil"` | RGB `PIL.Image.Image` |
| `"numpy"` | OpenCV 风格的 BGR `numpy.ndarray` |
| `"bytes"` | 编码后的图像字节 |
| `"base64"` | Base64 编码的图像字符串 |
| `"file"` | 临时图像文件的路径字符串 |

`tensor` 是一种可接受的输入类型，而不是 `ImageReader` 的输出格式。请使用模型或数据集所采用的变换，将 PIL 或 NumPy 结果转换为张量。

当源数据没有可检测的扩展名时，如需输出字节或 Base64，请提供 `ext`：

```python
encoded = llv.imread(image_array, output_format="bytes", ext="png")
```

## 查看输入元数据

```python
from openLLV.data import ImageReader

reader = ImageReader()
info = reader.get_info("input.jpg")
print(info)
```

## 保存图像

保存到指定文件：

```python
saved_path = llv.imwrite(image, "results/output.png")
```

保存到目录并指定文件名：

```python
saved_path = llv.imwrite(
    image,
    output="results/",
    output_name="enhanced",
    save_format="jpg",
)
```

省略 `output` 时，`ImageWriter` 使用 `results/`。能够推断源文件名时会保留该名称，否则会生成一个名称。

## 别名

`read_image()` 是 `imread()` 的别名，`write_image()` 是 `imwrite()` 的别名。

