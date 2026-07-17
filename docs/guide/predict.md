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

The deep backend returns a PIL image; the traditional backend returns a NumPy array. `saved_path` is a `Path`, or `None` when `save=False`.

For a directory, prediction recursively processes supported images and returns a deterministic list of saved `Path` objects. Relative subdirectories and source suffixes are preserved.

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
    progress_bar=True,
)
```

Directory inference runs one image at a time so differently sized inputs remain safe. The deep predictor's `batch_size` and `num_workers` values are currently metadata reserved for future batched pipelines.

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
