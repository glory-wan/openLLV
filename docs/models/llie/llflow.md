# LLFlow

Standard LLFlow (AAAI 2022), ported from the [official LOL-pc implementation](https://github.com/wyf0912/LLFlow).

Uses a 24-RRDB condition encoder and three-level normalizing flow. Training minimizes conditional Gaussian NLL. Inference decodes the predicted color-map prior mean with official histogram/log preprocessing and symmetric padding.

All model layers and helpers are in `LLFlow.py`; preprocessing helpers are in
`LLFlowDataset.py` and are shared by training and inference.

See the [complete LLFlow reference](../../reference/models/llflow.md) for parameters, official weight conversion, training, licenses and integration limitations. Old simplified-model parameters and checkpoints are incompatible.
