# LLFormer

> Task: low-light image enhancement (`llie`)

LLFormer is the transformer-based method from **Ultra-High-Definition Low-Light Image Enhancement: A Benchmark and Transformer-Based Method** (AAAI 2023 Oral). It combines row/column axial self-attention, dual-gated feed-forward networks, cross-layer attention fusion, a four-level encoder-decoder, and learned weighted skip connections.

## Links

| Type | URL |
| --- | --- |
| Paper | https://arxiv.org/abs/2212.11548 |
| Official source code | https://github.com/TaoWangzj/LLFormer |
| Default configuration | `openLLV/deepLearning/config/LLFormer.yaml` |

## License Notice

The integrated architecture is adapted from an upstream implementation distributed for academic, non-commercial use under CC BY-NC-SA 4.0. Review the license in the official source repository before redistribution or commercial use; its terms may be narrower than openLLV's repository-level license.

## openLLV Implementation

| Item | Location |
| --- | --- |
| Model | `openLLV/deepLearning/models/LLIE/LLFormer.py` |
| Model name | `LLFormer` (alias: `LL-Former`) |
| Loss | `openLLV/deepLearning/loss/LLIELoss/LLFormer_Loss.py` |
| Loss name | `llformer` |

The default architecture uses `dim=16`, blocks `[2, 4, 8, 16]`, heads `[1, 2, 4, 8]`, two refinement blocks, WithBias layer normalization, and no global residual skip. The registered loss follows the official Smooth L1 training objective.

LLFormer pads inputs to dimensions divisible by 16 and crops predictions back to their original size.

## Training

The packaged config uses `CommonDataset` with matching files under `train/input` and `train/target`, plus the corresponding validation directories.

```bash
python -m openLLV.cli train LLFormer --kwargs root_dir=datasets/my_dataset
```

Or use Python:

```python
import openLLV as llv

result = llv.train("LLFormer", root_dir="datasets/my_dataset")
print(result["checkpoint_dir"])
```

## Prediction

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "checkpoints/LLFormer_CommonDataset/checkpoints/best.pt",
    "input.png",
    output="results/LLFormer/output.png",
    device="cuda",
)
```

For memory-efficient UHD inference, enable overlap-and-average tiling:

```python
enhanced, saved_path = llv.predict(
    "checkpoints/LLFormer_CommonDataset/checkpoints/best.pt",
    "uhd_input.png",
    output="results/LLFormer/uhd_output.png",
    device="cuda",
    config={
        "tile_size": [720, 1280],
        "tile_overlap": [360, 640],
    },
)
```

Tile dimensions must be divisible by 16. Overlapping predictions, including boundary tiles, are averaged.

## Official Checkpoints

Official checkpoints may store weights below `state_dict` and prefix keys with `module.` after `DataParallel` training. Strip that prefix before loading a raw state dictionary into `LLVModel.create_model("LLFormer")`. Checkpoints produced by openLLV already contain model metadata and can be passed directly to `llv.predict`.

