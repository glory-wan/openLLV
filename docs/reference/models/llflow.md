# LLFlow

> Task: low-light image enhancement (`llie`)

Standard LLFlow from official LOL-pc, replacing the former simplified flow.
Trainer and both Predictor interfaces are unchanged.

## Links

| Type | URL |
| --- | --- |
| Paper | https://doi.org/10.1609/aaai.v36i3.20162 |
| Official code | https://github.com/wyf0912/LLFlow |
| Reference revision | `115da161a96de868d67494a32db848e31f85bbc1` |
| Reference configuration | https://github.com/wyf0912/LLFlow/blob/115da161a96de868d67494a32db848e31f85bbc1/code/confs/LOL-pc.yml |
| Packaged configuration | `openLLV/deepLearning/config/LLFlow.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/LLFlow.py` |
| Official layers and helpers | Consolidated in `openLLV/deepLearning/models/LLIE/LLFlow.py` |
| Licenses | `openLLV/deepLearning/models/LLIE/LLFlow.LICENSE` |
| Class / registered name | LLFlow, case-insensitive, no aliases |
| Base class | LLVModel in `openLLV/deepLearning/models/BaseModel.py` |
| Loss | `openLLV/deepLearning/loss/LLIELoss/LLFlow_Loss.py` |
| Dataset adapter | `openLLV/data/datasets/LLFlowDataset.py` |

## Implementation Notes

ConEncoder1 uses 64 channels and 24 RRDBs; block features 1/3/5/7 are
concatenated with multi-scale features. Each of three flow levels squeezes
the image, applies two ActNorm/invertible 1x1 convolution steps without coupling,
then twelve CondAffineSeparatedAndCond steps. Splitting is disabled. The latent
has 192 channels at one-eighth the input height and width.

Histogram equalization, log transformation and padding helpers are defined in
`openLLV/data/datasets/LLFlowDataset.py` and shared by the dataset and model.
Preprocessing concatenates log(low + 0.001) and per-channel OpenCV histogram
equalization. The encoder also computes color and gradient/noise maps.
LLFlowDataset equalizes the full image before synchronously cropping/flipping
low, GT, and equalized images. It supports LOL our485/low, our485/high,
eval15/low, eval15/high and CommonDataset input/target layouts. Other layouts
require explicit input_dir/target_dir; automatic LOL-v2 discovery is not included.

Training minimizes only mean conditional Gaussian NLL:
`-(logdet + log N(z; mean, I)) / (log(2) * H * W)`.
One Python random draw per batch selects predicted color (probability 0.8) or
GT / (channel_sum(GT) + 0.0001) (probability 0.2) as the prior mean.
No reconstruction, L1, color or TV penalties are added. GT dequantization is
off by default, matching the actual official training call.

Inference decodes the squeezed predicted color map, deterministically.
It does not use a zero or random latent. Default padding matches official
test_unpaired.py: OpenCV BORDER_REFLECT, adding 16 - size % 16 pixels per
axis, including a full 16 when already divisible. Padding precedes histogram
equalization; output is cropped back to the original dimensions. Raw model
output is unclamped; Predictor handles image conversion.

Paired forward encodes GT first to initialize ActNorm, then decodes an image
under no_grad to satisfy the prediction contract. NLL remains differentiable.
Loaded ActNorm statistics are retained even with zero bias. Singular invertible
convolutions raise RuntimeError instead of retrying indefinitely.

## Parameters

Constructor: `LLFlow(config=None, **kwargs)`. Merge order: defaults, config,
kwargs. Unknown keys are retained but unused, except removed legacy parameters
which explicitly raise ValueError.

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| config | dict or None | None | Other types raise TypeError. |
| model_name | str | "LLFlow" | Inherited metadata name. |
| input_channels | int | 3 | Must equal 3; forward separately accepts six prepared channels. |
| save_dir | str | "./checkpoints/llie/LLFlow" | Inherited save directory. |
| nf | int | 64 | Encoder width; must equal 64. |
| nb | int | 24 | RRDB count, integer >= 8, excluding bool. |
| K | int | 12 | Coupling steps per level, integer >= 1, excluding bool. |
| L | int | 3 | Flow levels; must equal 3. |
| train_gt_ratio | float | 0.2 | Converted with float(); finite, in [0, 1]. |
| quant | float | 32 | Positive finite quantization count; only used with GT noise. |
| inference_padding | str | "reflect16" | "reflect16" or "none"; paired calls never pad. |
| mode | str | "inference" | "train" requires paired_image; "inference" allows enhancement. |

Removed: condition_channels, condition_blocks, flow_layers,
flow_hidden_channels, scale_clamp, sample_temperature. Old simplified-model
checkpoints are incompatible. Changing nb/K changes the architecture; nb=8/K=1
is useful for tests but is neither official smallNet nor the standard model.

### Forward and weight loading

`forward(x, paired_image=None, *, add_gt_noise=False)`:

| Parameter | Type | Default | Contract |
| --- | --- | --- | --- |
| x | Tensor | required | Floating BCHW RGB in [0,1] or six prepared channels; positive H/W. |
| paired_image | Tensor or None | None | RGB GT with matching batch/H/W; paired H/W divisible by 8. |
| add_gt_noise | bool | False | Add uniform (-0.5,0.5)/quant GT noise and quantization logdet correction. |

Without GT, returns original-size BCHW RGB. With GT, returns
`{"pred": Tensor, "aux": {"nll": B-vector, "latent": Tensor, "logdet": B-vector}, "meta": {"mode": str}}`.
With padding="none", unpaired H/W must also be divisible by 8.
Call eval() for direct inference; Predictor does so automatically.

- `rrdbPreprocessing(low)`: six prepared channels to feature dictionary.
- `encode(target, condition, *, add_gt_noise=False)`: RGB GT and feature
  dictionary to (latent, per-sample NLL, logdet).
- `decode(condition)`: feature dictionary to RGB from predicted prior mean.
- `load_official_weights(path)`: local pure generator state_dict, removes an
  optional module. prefix, loads strictly, returns self. No automatic download.

The loss is registered as llflow (aliases llflow_loss, low_light_flow,
normalizing_flow_loss). Its constructor has no model-specific parameters.
`LLFlow_Loss()(model_output, target)` and Trainer's compute interface reduce
aux.nll; missing GT or malformed NLL raises ValueError.

## Usage Example

```python
import openLLV as llv
from openLLV.deepLearning.models.LLIE.LLFlow import LLFlow

model = LLFlow().load_official_weights("pretrained/39000_G.pth").eval()
checkpoint = model.save_model("checkpoints/llflow")
enhanced, saved_path = llv.predict(
    checkpoint, "input.jpg", output="results/llflow/output.png",
)
```

```python
import openLLV as llv

llv.train("LLFlow", root_dir="datasets/LOL", output_dir="runs/llflow")
```

YAML uses LLFlowDataset, 160-pixel crops, horizontal flips, batch 16,
Adam lr=0.0005 and AMP. For limited VRAM reduce batch_size and recalculate
the epoch schedule.

## Limits of the unchanged integration

| Official behavior | openLLV integration |
| --- | --- |
| 40,000 iteration loop; initial iteration skips optimization | Whole epochs, every batch optimized. YAML: 1,334 epochs = 40,020 updates for 485 images, batch 16, drop_last=True. |
| 3,000-iteration warmup; decay at 20k/30k/36k/38k | No per-update scheduler hook. YAML approximates milestones by epochs 667/1000/1200/1267, gamma 0.5, without warmup. Recalculate for other data/batch sizes. |
| Validate every 200 iterations, save every 1,000 | Epoch cadence only; YAML validates each epoch and uses Trainer's best/last saves. |
| Training batch 16, validation batch 1 | Trainer shares one batch size; full-resolution validation can use considerably more VRAM. Lower the shared batch size as needed. |
| GT brightness adjustment for reported PSNR, custom SSIM, best PSNR checkpoint | Trainer validates NLL and selects best loss. Predictor performs no GT-dependent adjustment; generic metrics are not directly comparable to official results. |
| Separate generator and optimizer/scheduler files | Generator conversion is supported; official optimizer/scheduler resume files cannot be passed directly to Trainer. |
| DataParallel-specific parameter grouping | YAML matches actual official single-GPU Adam behavior: betas=(0.9,0.999), no decay. Official DataParallel matches RRDB names into a separate 1e-5 decay group; Trainer's YAML optimizer builder lacks named grouping. |

Official beta1/beta2 are unused parameter-group keys, so Adam uses
(0.9,0.999). train_RRDB=false and train_RRDB_delay do not freeze the encoder
in the official standard execution path; it is trained from the start here.
Architecture and objective are reproduced; the complete training/evaluation
driver is not. These differences preclude claiming exact paper metrics.

## Checkpoint / Official Weights

Preserves RRDB.* and flowUpsamplerNet.* names/shapes, including unused official
parameters for strict generator loading. Use LOL-pc weights, not LOL_smallNet.
Random initialization supports training; meaningful inference needs trained
weights. Official code retains CC BY-NC-SA 4.0 and bundled component licenses;
see `openLLV/deepLearning/models/LLIE/LLFlow.LICENSE`.
