# Custom Traditional Enhancement Algorithms

Traditional low-level vision methods inherit from `LLVEnhancer` in `openLLV/tradition/algorithms/BaseModel.py`. Place an implementation in the package matching its domain, such as `BaseMethods`, `Dehazing`, or `LLIE`.

## 1. Base Class Contract

`LLVEnhancer` provides public RGB image loading/output, internal BGR conversion, value-range normalization/restoration, validation, dtype preservation, clipping, automatic registration, and factory construction. Subclasses implement only `_enhance()` plus their own parameters.

## 2. Minimal Algorithm

```python
from typing import Any

import numpy as np

from openLLV.tradition.algorithms import LLVEnhancer


class MyAlgorithm(LLVEnhancer):
    name = "my_algorithm"
    aliases = ["myalgo"]
    working_range = "byte"

    def __init__(self, strength: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        if strength < 0:
            raise ValueError("strength must be non-negative.")
        self.strength = float(strength)

    def _enhance(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        strength = float(kwargs.get("strength", self.strength))
        result = image.astype(np.float32) * strength
        return result

    def get_params(self):
        params = super().get_params()
        params["strength"] = self.strength
        return params
```

Three-channel arrays received and returned by `_enhance()` use OpenCV-style BGR order. The base class converts public RGB input to BGR before the call, then converts the result back to RGB after clipping and optional dtype restoration. Do not perform either boundary conversion in the subclass.

Set `working_range = "byte"` when `_enhance()` consumes and returns `uint8` values in `[0,255]` (the default, suitable for OpenCV histogram methods). Set `working_range = "unit"` when it consumes and returns `float32` values in `[0,1]`. The base class maps the caller's resolved input range into this working range and restores the caller's range afterward.

## 3. Base Options

| Option | Default | Meaning |
| --- | --- | --- |
| `output_type` | `"numpy"` | `numpy`, `pil`, `bytes`, `base64`, or `file` |
| `keep_dtype` | `True` | Cast the result back to the input dtype |
| `clip_output` | `True` | Clip to the resolved input value range |
| `value_range` | `"auto"` | `"auto"`, `"unit"`, `"byte"`, or custom `(min, max)`/`[min, max]`; controls input range interpretation |

`"auto"` interprets floating arrays with maximum `<= 1` as `[0,1]` and those with maximum `<= 255` as `[0,255]`. Because a very dark byte-range float image may also have maximum `<= 1`, pass `value_range="byte"` in that ambiguous case. Negative or float values above `255` require an explicit valid custom range; non-finite input values are always rejected.

## 4. Register and Use

Export the class from its domain `__init__.py` and from `openLLV/tradition/algorithms/__init__.py` so importing openLLV triggers registration.

```python
import openLLV as llv

enhanced, saved_path = llv.predict(
    "myalgo",
    "input.jpg",
    output="results/myalgo.png",
    strength=1.2,
)
```

Direct factory use is also available:

```python
from openLLV.tradition.algorithms import LLVEnhancer

enhancer = LLVEnhancer.create_enhancer(
    "my_algorithm",
    output_type="pil",
    strength=1.2,
)
result = enhancer("input.jpg")
```

Use `llv.list_algorithms()` to confirm the class name, declared name, and aliases were registered.
