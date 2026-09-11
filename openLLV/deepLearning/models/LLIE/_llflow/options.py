"""Small option helpers used by the vendored official modules."""


def opt_get(options, keys, default=None):
    for key in keys:
        if not isinstance(options, dict) or key not in options:
            return default
        options = options[key]
    return options


def standard_options(config):
    """Build the architecture options of official LOL-pc.yml."""
    return {
        "scale": 1,
        "cond_encoder": "ConEncoder1",
        "concat_histeq": True,
        "concat_color_map": False,
        "gray_map": False,
        "le_curve": False,
        "sigmoid_output": False,
        "datasets": {"train": {"GT_size": 160, "quant": config["quant"]}},
        "network_G": {"flow": {
            "K": config["K"], "L": 3,
            "coupling": "CondAffineSeparatedAndCond",
            "additionalFlowNoAffine": 2,
            "split": {"enable": False}, "fea_up0": True,
            "stackRRDB": {"blocks": [1, 3, 5, 7], "concat": True},
        }},
    }
