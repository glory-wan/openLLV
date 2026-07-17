# 评估 API

`openLLV.evaluate()` 会评估输出目录中的所有受支持图像。参考图像是可选的，但没有匹配参考图像时会跳过全参考指标。

## 函数形式

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

`en` 和 `ref` 分别是 `en_img_dir` 和 `ref_img_dir` 的别名。不要同时传入同一参数的两种形式。

## 可用指标

```python
import openLLV as llv

print(llv.list_metrics())
```

| 指标 | 是否需要参考图像 | 趋势 |
| --- | --- | --- |
| PSNR | 是 | 越高越好 |
| SSIM | 是 | 越高越好 |
| MSE | 是 | 越低越好 |
| MAE | 是 | 越低越好 |
| LPIPS | 是 | 越低越好 |
| LOE | 是；使用原始低光图像作为参考 | 越低越好 |
| NIQE | 否 | 越低越好 |
| MUSIQ | 否 | 越高越好 |
| PI | 否 | 越低越好 |

LPIPS、NIQE、MUSIQ 和 PI 使用可选的 `pyiqa` 包。请求这些指标时，请安装该依赖，并确保其后端所需的指标权重可用。

## 全参考评估

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

增强图像和参考图像按不区分大小写的文件名主干进行配对；图像扩展名可以不同。

## 无参考评估

```python
results = llv.evaluate(
    en_img_dir="results/zero_dce",
    metrics=["NIQE", "MUSIQ", "PI"],
)
```

没有提供参考图像目录时，评估器会为请求的全参考指标记录 `NaN`。

## 结果结构

返回的字典包含逐图像数值和汇总统计：

```python
print(results["filenames"])
print(results["metrics"]["PSNR"])
print(results["statistics"]["PSNR"]["mean"])
```

省略 `save_path` 时，评估结果会保存到 `./results/eval.json`。

## 返回评估器

```python
evaluator = llv.evaluate(
    en="results/zero_dce",
    ref="datasets/my_dataset/test/target",
    metrics=["PSNR", "SSIM"],
    return_evaluator=True,
)

print(evaluator.results)
```

## 预测与评估

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

