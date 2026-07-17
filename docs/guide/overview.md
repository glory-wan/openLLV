# openLLV Top-Level API Overview

openLLV provides one public interface for learned low-level vision models, traditional enhancement algorithms, image I/O, training, and image-quality evaluation. The current built-in deep-learning task is low-light image enhancement (`llie`); the directory layout is ready to add more low-level vision tasks without introducing another intermediate model base class.

## Main APIs

| API | Purpose |
| --- | --- |
| `openLLV.predict()` | Run a registered model, checkpoint, or traditional algorithm |
| `openLLV.enhance()` | Alias of `predict()` |
| `openLLV.train()` | Train an `LLVModel` from a built-in config, YAML file, or dictionary |
| `openLLV.evaluate()` / `openLLV.eval()` | Evaluate a directory of output images |
| `openLLV.imread()` / `openLLV.read_image()` | Read and convert an image |
| `openLLV.imwrite()` / `openLLV.write_image()` | Save an image |
| `openLLV.list_available()` | List registered models, algorithms, metrics, losses, and datasets |

## Package Map

| Package | Responsibility |
| --- | --- |
| `openLLV.deepLearning.models` | `LLVModel`, model registry, and task-specific implementations |
| `openLLV.deepLearning.loss` | `BaseLoss` and registered training losses |
| `openLLV.deepLearning.config` | Built-in YAML configs and default training values |
| `openLLV.tradition.algorithms` | `LLVEnhancer` and traditional algorithms |
| `openLLV.data` | Image I/O, transforms, and datasets |
| `openLLV.evaluation` | Metrics and directory evaluator |

Documentation for concrete implementations follows the same task grouping:

```text
docs/
  models/
    llie/
  algorithms/
    base_methods/
    dehazing/
    llie/
```

## View Available Components

```python
import openLLV as llv

components = llv.list_available()
print(components["models"])
print(components["algorithms"])
```

The flat helpers return every accepted registry key, including aliases:

```python
print(llv.list_models())
print(llv.list_algorithms())
print(llv.list_metrics())
print(llv.list_losses())
print(llv.list_datasets())
```

## Basic Workflow

Read and save an image:

```python
import openLLV as llv

image = llv.imread("input.jpg", output_format="pil")
saved_path = llv.imwrite(image, "results/copy.png")
```

Run a traditional algorithm:

```python
enhanced, saved_path = llv.predict(
    "Gamma",
    "input.jpg",
    output="results/gamma.png",
    gamma=0.6,
)
```

Run a deep-learning model:

```python
enhanced, saved_path = llv.predict(
    "ZeroDCE",
    "input.jpg",
    output="results/zero_dce.png",
    device="cuda",
)
```

Train with a packaged configuration:

```python
result = llv.train(
    "ZeroDCE",
    root_dir="datasets/my_dataset",
    epochs=10,
)
```

Evaluate a result directory:

```python
results = llv.evaluate(
    en_img_dir="results/zero_dce",
    ref_img_dir="datasets/my_dataset/test/target",
    metrics=["PSNR", "SSIM"],
)
```

## Next Pages

Use `predict.md`, `train.md`, `evaluate.md`, and `image_io.md` for the complete public workflows. Extension points are described under `docs/custom/`.

