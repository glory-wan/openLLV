# openLLV Command-Line Interface

The current repository exposes its CLI module with:

```bash
python -m openLLV.cli --help
```

The parser program name is `openllv`, but module invocation does not depend on a separately installed console-script entry point.

## Available Commands

| Command | Purpose |
| --- | --- |
| `predict` | Process one image or a directory |
| `train` | Train from a built-in config, YAML path, or keyword overrides |
| `evaluate` / `eval` | Evaluate an output directory |
| `imwrite` | Write or convert an image |
| `list` | List registered components |

## List Components

```bash
python -m openLLV.cli list
```

The result is grouped into models, algorithms, metrics, losses, and datasets, with one row per implementation class and its aliases.

## Predict

Traditional algorithm:

```bash
python -m openLLV.cli predict Gamma input.jpg -o results/gamma.png --kwargs gamma=0.6
```

Deep-learning model:

```bash
python -m openLLV.cli predict ZeroDCE input.jpg -o results/zero_dce.png --device cuda
```

Directory input:

```bash
python -m openLLV.cli predict ZeroDCE images -o results/zero_dce --no-progress
```

Important options include `--backend`, `--device`, `--output-dir`, `--no-save`, `--output-name`, and `--output-ext`.

## Train

Select a packaged configuration by name:

```bash
python -m openLLV.cli train ZeroDCE --kwargs root_dir=datasets/my_dataset epochs=10 batch_size=4 device=cuda
```

Or provide a YAML path:

```bash
python -m openLLV.cli train configs/experiment.yaml --kwargs lr=5e-5 amp=true
```

## Evaluate

Full-reference:

```bash
python -m openLLV.cli evaluate --en results/zero_dce --ref datasets/my_dataset/test/target --metrics PSNR SSIM --save-path results/zero_dce/eval.json
```

No-reference:

```bash
python -m openLLV.cli eval --en results/zero_dce --metrics NIQE MUSIQ
```

`--en-img-dir` and `--ref-img-dir` are long aliases of `--en` and `--ref`.

## Imwrite

```bash
python -m openLLV.cli imwrite input.jpg -o results/copy.png
```

Override the format or directory filename:

```bash
python -m openLLV.cli imwrite input.jpg -o results --save-format png --output-name converted
```

## `KEY=VALUE` Parsing

`--kwargs` accepts zero or more `KEY=VALUE` tokens. Common Python literals are parsed:

- `true`, `false`, and `none` become Python values;
- numbers, lists, tuples, and dictionaries use literal parsing;
- other values remain strings.

```bash
python -m openLLV.cli train ZeroDCE --kwargs epochs=5 amp=false optimizer_params="{'weight_decay': 0.0001}"
```

Unknown training overrides and malformed tokens fail explicitly.

## Command Help

```bash
python -m openLLV.cli predict --help
python -m openLLV.cli train --help
python -m openLLV.cli evaluate --help
```

