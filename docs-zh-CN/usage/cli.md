# openLLV 命令行接口

当前仓库可以通过以下方式调用 CLI 模块：

```bash
python -m openLLV.cli --help
```

解析器中的程序名称为 `openllv`，但模块调用方式不依赖单独安装的控制台脚本入口。

## 可用命令

| 命令 | 用途 |
| --- | --- |
| `predict` | 处理单张图像或目录 |
| `train` | 使用内置配置、YAML 路径或关键字覆盖参数进行训练 |
| `evaluate` / `eval` | 评估输出目录 |
| `imwrite` | 写入或转换图像 |
| `list` | 列出已注册的组件 |

## 列出组件

```bash
python -m openLLV.cli list
```

结果按模型、算法、指标、损失函数和数据集分组，每个实现类及其别名占一行。

## 预测

传统算法：

```bash
python -m openLLV.cli predict Gamma input.jpg -o results/gamma.png --kwargs gamma=0.6
```

深度学习模型：

```bash
python -m openLLV.cli predict ZeroDCE input.jpg -o results/zero_dce.png --device cuda
```

目录输入：

```bash
python -m openLLV.cli predict ZeroDCE images -o results/zero_dce --no-progress
```

重要选项包括 `--backend`、`--device`、`--output-dir`、`--no-save`、`--output-name` 和 `--output-ext`。

## 训练

按名称选择内置配置：

```bash
python -m openLLV.cli train ZeroDCE --kwargs root_dir=datasets/my_dataset epochs=10 batch_size=4 device=cuda
```

或者提供 YAML 路径：

```bash
python -m openLLV.cli train configs/experiment.yaml --kwargs lr=5e-5 amp=true
```

## 评估

全参考评估：

```bash
python -m openLLV.cli evaluate --en results/zero_dce --ref datasets/my_dataset/test/target --metrics PSNR SSIM --save-path results/zero_dce/eval.json
```

无参考评估：

```bash
python -m openLLV.cli eval --en results/zero_dce --metrics NIQE MUSIQ
```

`--en-img-dir` 和 `--ref-img-dir` 分别是 `--en` 和 `--ref` 的长选项别名。

## Imwrite

```bash
python -m openLLV.cli imwrite input.jpg -o results/copy.png
```

覆盖格式或目录中的文件名：

```bash
python -m openLLV.cli imwrite input.jpg -o results --save-format png --output-name converted
```

## `KEY=VALUE` 解析

`--kwargs` 接受零个或多个 `KEY=VALUE` 标记。常见 Python 字面量会被解析：

- `true`、`false` 和 `none` 会转换为 Python 值；
- 数字、列表、元组和字典使用字面量解析；
- 其他值保留为字符串。

```bash
python -m openLLV.cli train ZeroDCE --kwargs epochs=5 amp=false optimizer_params="{'weight_decay': 0.0001}"
```

未知的训练覆盖参数和格式错误的标记都会显式报错。

## 命令帮助

```bash
python -m openLLV.cli predict --help
python -m openLLV.cli train --help
python -m openLLV.cli evaluate --help
```

