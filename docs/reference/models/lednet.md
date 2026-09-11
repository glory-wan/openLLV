# LEDNet

> Task: low-light image enhancement and deblurring (LLIE)

LEDNet is openLLV's encoder-decoder combining pyramid pooling, curve attention, and dynamic filtering.

## Links

| Type | URL |
| --- | --- |
| Paper | https://arxiv.org/pdf/2202.03373 |
| Official source code | https://github.com/sczhou/LEDNet |
| Official project page | None |
| Default configuration | `openLLV/deepLearning/config/LEDNet.yaml` |

## Location in openLLV

| Item | Location |
| --- | --- |
| Implementation | `openLLV/deepLearning/models/LLIE/LEDNet.py` |
| Class name | `LEDNet` |
| Registered name | `LEDNet` (no aliases; lookup is case-insensitive and trims whitespace) |
| Base class | `LLVModel` in `openLLV/deepLearning/models/BaseModel.py` |
| Related loss | `openLLV/deepLearning/loss/LLIELoss/LEDNet_Loss.py` |

## Implementation Notes

Three downsampling stages feed pyramid-pooling and curve-attention modules; the decoder applies generated dynamic kernels and optional skip additions. Inference returns the tensor. Training returns standardized output, with `aux.side_output` only when side supervision is active. Forward `side_loss=None` enables it only when both configured `use_side_loss` and training `mode` are true; an explicit boolean overrides that decision. The YAML overrides direct-construction `use_side_loss` from `False` to `True` and omits architecture keys that retain source defaults.

The built-in YAML normalizes both input and GT with mean/std `[0.5,0.5,0.5]` after `ToTensor`, giving `[-1,1]` tensors for training and validation. Training first applies a paired random `256x256` crop, then horizontal flip, vertical flip, and spatial transpose, each independently enabled with probability 0.5. Both images share crop coordinates and augmentation decisions. Validation retains original dimensions and has no random spatial transforms. Cropping rejects undersized images and unequal paired dimensions instead of resizing them.

The `lednet` loss and YAML default to `range_norm=True`, matching the [released BasicSR settings](https://github.com/sczhou/LEDNet/blob/master/options/train_LEDNet.yml). Main and side perceptual terms apply `(x + 1) / 2`, then ImageNet mean/std normalization, without clipping predictions or targets. Pixel L1 uses the original `[-1,1]` tensors.

The YAML also sets model `image_range="minus_one_one"`, which is saved in checkpoints. Predictor converts its preprocessed `[0,1]` tensor to `[-1,1]` before forward and maps the prediction back with `(output+1)/2` before 8-bit conversion. Custom Predictor transforms must therefore still return `[0,1]` tensors; do not normalize twice. Raw `LEDNet.forward` does not convert ranges: its caller must supply the range used during training and interpret the raw output accordingly. Direct construction defaults to `image_range="zero_one"` for paper weights and older `[0,1]` checkpoints.

The YAML sets `train.seed=10`, `train.cudnn_deterministic=False`, and `train.cudnn_benchmark=True`, matching the official training script's default cuDNN behavior. Trainer seeds Python, NumPy, PyTorch, the current CUDA device, and all CUDA devices. Benchmarking does not guarantee identical results across runs. Load these settings with `llv.train(config="LEDNet", root_dir="datasets/my_dataset")`.

The shared deep-learning Predictor converts output to 8-bit images by clamping to `[0,1]`, multiplying by 255, rounding to the nearest integer (ties to even), and casting to `uint8`. This applies to grayscale, RGB, and RGBA outputs; the raw model tensor is unchanged.

## Parameters

| Parameter | Type | Default | Meaning / constraints |
| --- | --- | --- | --- |
| `config` | `Optional[Dict[str, Any]]` | `None` | Configuration mapping; a non-dictionary raises `TypeError`; `**kwargs` override it. |
| `model_name` | `str` | `"LEDNet"` | Base-class metadata. |
| `input_channels` | `int` | `3` | Input/output channels; must be a positive integer. |
| `save_dir` | `str` | `"./checkpoints/llie/LEDNet"` | Default checkpoint directory. |
| `channels` | `List[int]` | `[32, 64, 128, 128]` | Four stage widths; must contain four positive values. |
| `connection` | `bool` | `False` | Enables decoder-to-encoder skip additions. Compatible stage shapes/channels are required but not explicitly validated. |
| `use_side_loss` | `bool` | `False` | Default side-supervision switch. The YAML overrides it to `True`. |
| `mode` | `str` | `"inference"` | Must be `"train"` or `"inference"`. |
| `image_range` | `str` | `"zero_one"` | Raw network input/output convention used by Predictor. Accepts `"zero_one"` or `"minus_one_one"`; other values raise `ValueError`. YAML overrides to `"minus_one_one"`; does not change raw forward computation. |
| `kernel_size` | `int` | `5` | Dynamic convolution kernel size; must be odd. Positivity is not explicitly validated. |
| `curve_n` | `int` | `3` | Curve-attention iteration count; must be positive. |
| `ppm_bins` | `Tuple[int, ...]` | `(1, 2, 3, 6)` | Adaptive-pooling output sizes for pyramid pooling; not explicitly validated. |

## Dataset Settings

These optional `CommonDataset` arguments are forwarded by Trainer; they do not change other datasets' defaults.

| Parameter | Default | LEDNet YAML | Meaning |
| --- | --- | --- | --- |
| `mean` | `None` | `data.params.mean: [0.5,0.5,0.5]` | Per-channel tensor mean for both inputs and GT, including validation. |
| `std` | `None` | `data.params.std: [0.5,0.5,0.5]` | Positive finite standard deviation; supplied together with mean. |
| `crop_size` | `None` | `data.train_params.crop_size: 256` | Positive integer square crop, applied to the pair before tensor conversion. |
| `use_flip` | `False` | `data.train_params.use_flip: true` | Shared horizontal flip with probability 0.5. |
| `use_rot` | `False` | `data.train_params.use_rot: true` | Shared vertical flip and transpose, independently with probability 0.5. |

`resize` remains `None`. Dataset repetition, `drop_last`, epoch-based training, and the learning-rate schedule retain their existing settings.

## Loss Parameters

These are keyword-only arguments to `LEDNet_Loss`, also accepted through `loss.params`:

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `pixel_weight` | `float` | `1.0` | Pixel L1 weight. |
| `perceptual_weight` | `float` | `0.01` | Global VGG feature L1 weight; non-positive values disable the perceptual term. |
| `side_loss_weight` | `float` | `0.8` | Weight of the complete side loss. |
| `use_side_loss` | `bool` | `True` | Add side supervision when a side output is supplied. |
| `use_perceptual` | `bool` | `True` | Enable the perceptual term. |
| `pretrained_vgg` | `bool` | `True` | Load ImageNet VGG19 weights. |
| `layer_weights` | `Optional[Dict[str, float]]` | `None` | Uses `conv1_2`, `conv2_2`, `conv3_4`, `conv4_4`, each weighted 1; unsupported names raise `KeyError`. |
| `range_norm` | `bool` | `True` | Apply `(x+1)/2` before VGG, without clipping. |
| `use_input_norm` | `bool` | `True` | Apply ImageNet mean/std normalization before VGG. |

`VGG19PerceptualLoss` also defaults to `range_norm=True`. Set `loss_params={"range_norm": False}` to disable only the range conversion.

## Usage Example

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "LEDNet", "input.jpg", output="results/lednet/output.png",
    config={"connection": True, "curve_n": 4},
)
```

## Checkpoint / Official Weights

Use an openLLV checkpoint path as the prediction target. No automatic official-weight download is implemented.
