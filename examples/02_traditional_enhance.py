"""Run single-image and directory prediction with traditional algorithms."""

from _utils import ensure_results_dir, find_example_folder, find_example_image

import openLLV as llv


def main() -> None:
    """Enhance a single image with Gamma and a directory with RCLAHE."""
    image_path = find_example_image()
    input_dir = find_example_folder()
    output_dir = ensure_results_dir("traditional")

    enhanced, single_path = llv.predict(
        "gamma",
        image_path,
        gamma=0.8,
        output=output_dir / "gamma_single.png",
    )
    batch_paths = llv.predict(
        "rclahe",
        input_dir,
        color_space="hsv",
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        iterations=2,
        output=output_dir / "rclahe_batch",
        progress_bar=False,
    )

    print(f"Single traditional output: {single_path}")
    print(f"Single output shape: {enhanced.shape}")
    print(f"Batch output directory: {output_dir / 'rclahe_batch'}")
    print(f"Batch image count: {len(batch_paths)}")


if __name__ == "__main__":
    main()
