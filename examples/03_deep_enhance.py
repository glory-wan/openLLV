"""Run deep-learning prediction with a registered openLLV model."""

from _utils import ensure_results_dir, find_example_image

import torch

import openLLV as llv


def main() -> None:
    """Run an untrained Zero-DCE model to demonstrate the prediction API."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_path = find_example_image()
    output_dir = ensure_results_dir("deep")

    enhanced, single_path = llv.predict(
        "ZeroDCE",
        image_path,
        output=output_dir / "zerodce_single.png",
        device=device,
    )

    print(f"Device: {device}")
    print(f"Single deep-learning output: {single_path}")
    print(f"Single output size: {enhanced.size}")
    print("Note: this example uses randomly initialized model weights.")


if __name__ == "__main__":
    main()
