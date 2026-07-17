# Evaluation API

`openLLV.evaluate()` evaluates every supported image in an output directory. Reference images are optional, but full-reference metrics are skipped when no matching reference is available.

## Function Form

```python
openLLV.evaluate(
    en_img_dir,
    ref_img_dir=None,
    metrics=None,
    save_path=None,
    return_evaluator=False,
    **kwargs,
)
```

`en` and `ref` are accepted as aliases for `en_img_dir` and `ref_img_dir`. Do not pass both forms of the same argument.

## Available Metrics

```python
import openLLV as llv

print(llv.list_metrics())
```

| Metric | Reference required | Better |
| --- | --- | --- |
| PSNR | Yes | Higher |
| SSIM | Yes | Higher |
| MSE | Yes | Lower |
| MAE | Yes | Lower |
| LPIPS | Yes | Lower |
| LOE | Yes; use the original low-light image as reference | Lower |
| NIQE | No | Lower |
| MUSIQ | No | Higher |
| PI | No | Lower |

LPIPS, NIQE, MUSIQ, and PI use the optional `pyiqa` package. If one of these metrics is requested, install that dependency and allow any metric weights required by its backend to be available.

## Full-Reference Evaluation

```python
results = llv.evaluate(
    en_img_dir="results/zero_dce",
    ref_img_dir="datasets/my_dataset/test/target",
    metrics=["PSNR", "SSIM", "LPIPS"],
    save_path="results/zero_dce/evaluation.json",
    device="cuda",
    batch_size=1,
    num_workers=4,
)
```

Enhanced and reference images are paired by case-insensitive filename stem; image suffixes may differ.

## No-Reference Evaluation

```python
results = llv.evaluate(
    en_img_dir="results/zero_dce",
    metrics=["NIQE", "MUSIQ", "PI"],
)
```

The evaluator records `NaN` for a requested full-reference metric when no reference directory is supplied.

## Result Structure

The returned dictionary contains per-image values and aggregate statistics:

```python
print(results["filenames"])
print(results["metrics"]["PSNR"])
print(results["statistics"]["PSNR"]["mean"])
```

Evaluation results are saved to `./results/eval.json` when `save_path` is omitted.

## Return the Evaluator

```python
evaluator = llv.evaluate(
    en="results/zero_dce",
    ref="datasets/my_dataset/test/target",
    metrics=["PSNR", "SSIM"],
    return_evaluator=True,
)

print(evaluator.results)
```

## Prediction and Evaluation

```python
llv.predict(
    "ZeroDCE",
    "datasets/my_dataset/test/input",
    output="results/zero_dce",
)

results = llv.evaluate(
    "results/zero_dce",
    "datasets/my_dataset/test/target",
    metrics=["PSNR", "SSIM"],
)
```
