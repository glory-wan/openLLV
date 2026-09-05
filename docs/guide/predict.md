# Prediction API

`openLLV.predict()` routes a request to the deep-learning or traditional backend. It accepts registry names case-insensitively, model checkpoints, and existing `LLVModel` or `LLVEnhancer` instances.

## Function Form

```python
openLLV.predict(target, source, output=None, **kwargs)
```

| Argument | Meaning |
| --- | --- |
| `target` | Model name, checkpoint path, algorithm name, or backend instance |
| `source` | Image input accepted by `ImageReader`, or an image directory |
| `output` | Optional output file for one image, or output directory for a directory input |
| `backend` | `"auto"`, `"deep"`, or `"traditional"` |

With `backend="auto"`, registry names and backend instances select their own backend. Files ending in `.pt` or `.pth` select the deep-learning backend. If a name ever exists in both registries, specify the backend explicitly.

## Return Contract

For one image, prediction returns a pair:

```python
enhanced_image, saved_path = openLLV.predict(...)
```

The deep backend returns an RGB PIL image; the traditional backend returns an RGB NumPy array. `saved_path` is a `Path`, or `None` when `save=False`.

For a directory, prediction recursively processes supported images in deterministic source-path order and preserves relative subdirectories. With `save=True`, it returns saved `Path` objects. With `save=False`, it creates no output files or directories and returns enhanced PIL images (deep backend) or RGB NumPy arrays (traditional backend).

## Traditional Algorithm

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "Gamma",
    "input.jpg",
    output="results/gamma/output.png",
    gamma=0.6,
)
```

Algorithm constructor parameters can be passed with the top-level call. Method-specific per-image overrides can also be supplied when using `Predictor.predict_single()`.

Traditional algorithms accept `value_range="auto"` (default), `"unit"`, `"byte"`, or a custom `(min, max)` range. Auto preserves ordinary float `[0,1]` and float `[0,255]` conventions. For a byte-range float image whose maximum is `<= 1`, pass `value_range="byte"` because its numeric content is indistinguishable from a unit-range image.

## Deep-Learning Model

```python
enhanced, saved_path = llv.predict(
    "ZeroDCE",
    "input.jpg",
    output="results/zero_dce/output.png",
    device="cuda",
)
```

`device` is owned by the predictor. It is not stored or managed by `LLVModel`. If no device is provided, CUDA is used when available and CPU otherwise.

Model constructor overrides are passed directly:

```python
enhanced, saved_path = llv.predict(
    "PairLIE",
    "input.jpg",
    config={"enhancement_gamma": 0.14},
    save=False,
)
```

Forward-call arguments belong in `model_kwargs`:

```python
enhanced, _ = llv.predict(
    "MyModel",
    "input.jpg",
    save=False,
    model_kwargs={"strength": 0.8},
)
```

## Checkpoint Prediction

openLLV training checkpoints include the model class, configuration, and state dictionary:

```python
enhanced, saved_path = llv.predict(
    "checkpoints/ZeroDCE_CommonDataset/checkpoints/best.pt",
    "input.jpg",
    output="results/from_checkpoint.png",
    device="cpu",
)
```

Raw upstream state dictionaries do not contain the openLLV model metadata. Create the matching model class and load such weights manually.

## Directory Prediction

```python
saved_paths = llv.predict(
    "ZeroDCE",
    "images/",
    output="results/zero_dce",
    output_ext=".PNG",
    progress_bar=True,
)
```

When `output_ext` is omitted, each source filename and extension is preserved exactly, including letter case. An explicit `output_ext` replaces every suffix and preserves the supplied case. `output_name` is single-image-only and raises `ValueError` for directory input.

```python
images = llv.predict(
    "Gamma",
    "images/",
    save=False,
    progress_bar=False,
)
```

Deep directory inference groups images by source size. Every complete group of `batch_size` compatible tensors uses one model call; incomplete groups run one image at a time. The pipeline never pads images. `num_workers` controls parallel image reading and CPU preprocessing, not model inference.

By default, `resize=None` applies no scaling and preserves every source image's original dimensions. To opt into resizing, pass a positive square size or an explicit `(height, width)` pair:

```python
images = llv.predict(
    "ZeroDCE",
    "images/",
    save=False,
    resize=(384, 512),
    batch_size=4,
    num_workers=2,
)
```

Resizing happens before a custom `transform`. With `num_workers > 0` on a spawn-based platform, that transform must be picklable.

## Unified Predictor Object

```python
from openLLV import Predictor

predictor = Predictor(
    "ZeroDCE",
    backend="deep",
    device="cuda",
    output_dir="results/zero_dce",
)

enhanced, saved_path = predictor("input.jpg")
print(predictor.get_params())
```

Use `backend="traditional"` for an explicit algorithm backend. Available lookup names can be inspected with `Predictor.list_available_models()` and `Predictor.list_available_methods()`.

## External Cancellation

Pass a `CancelSignal` through the `cancel` keyword and call `signal.cancel()` from another thread to stop a directory prediction. On cancellation, `predict` raises `TaskCancelled` at the next image or batch boundary; images already saved remain on disk.
