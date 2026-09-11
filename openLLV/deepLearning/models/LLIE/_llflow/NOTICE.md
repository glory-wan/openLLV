# LLFlow attribution and modifications

Source: https://github.com/wyf0912/LLFlow
Revision: 115da161a96de868d67494a32db848e31f85bbc1
Reference: code/confs/LOL-pc.yml (standard LOL model).

Copyright (c) 2021 Yufei Wang. Upstream LLFlow is licensed under CC BY-NC-SA 4.0,
with academic-research-use language in its license. Copies of the original
LLFlow, SRFlow, BasicSR and Glow notices are included in licenses/.
These terms continue to apply; openLLV's top-level MIT license does not replace them.

Modules in this directory are adapted from code/models/modules.
Imports were made package-local; unused NoEncoder and module_util helpers were
omitted. options.py provides the standard options and opt_get.
Permutations.py fails on singular/non-finite matrices instead of retrying
forever or perturbing weights, and uses torch.linalg with double-precision
inversion. FlowActNorms.py marks loaded statistics initialized, including zero
biases, without adding persistent state-dict keys.

Adjacent LLFlow.py adapts LLFlow_arch.py to LLVModel and preserves the standard
encoder, flow, Gaussian objective, prior selection and parameter names.
LLFlowDataset.py and llflow_preprocessing.py adapt code/data/LoL_dataset.py and
code/test_unpaired.py. LLFlow_Loss.py reduces NLL computed by paired forward.

See docs/reference/models/llflow.md for scheduling, validation, optimizer
grouping and resume differences from the official driver.
