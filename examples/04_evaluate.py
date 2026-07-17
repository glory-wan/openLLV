"""Enhance an image directory and evaluate the saved results."""

from _utils import ensure_results_dir, find_example_folder

import openLLV as llv


def main() -> None:
    """Evaluate Gamma outputs with built-in full-reference metrics."""
    input_dir = find_example_folder()
    output_dir = ensure_results_dir("evaluation")
    enhanced_dir = output_dir / "enhanced_by_gamma"
    save_path = output_dir / "evaluation.json"

    llv.predict(
        "gamma",
        input_dir,
        output=enhanced_dir,
        gamma=0.8,
        progress_bar=False,
    )

    results = llv.evaluate(
        en=enhanced_dir,
        ref=input_dir,
        metrics=["PSNR", "SSIM"],
        save_path=save_path,
        batch_size=1,
        num_workers=0,
        device="cpu",
    )

    print(f"Metrics: {list(results['statistics'])}")
    print(f"Evaluation result: {save_path}")


if __name__ == "__main__":
    main()
