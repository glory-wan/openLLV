"""Read one image in several formats and write converted copies."""

from _utils import ensure_results_dir, find_example_image

import openLLV as llv


def main() -> None:
    """Demonstrate the top-level image reading and writing API."""
    image_path = find_example_image()
    output_dir = ensure_results_dir("io")

    pil_image = llv.imread(image_path, output_format="pil")
    numpy_image = llv.imread(image_path, output_format="numpy")
    image_bytes = llv.imread(image_path, output_format="bytes")
    image_base64 = llv.imread(image_path, output_format="base64")

    copied_path = llv.imwrite(
        pil_image,
        output=output_dir,
        output_name="copied.png",
    )
    converted_path = llv.imwrite(
        pil_image,
        output=output_dir / "converted.jpg",
    )

    print(f"Input image: {image_path}")
    print(f"PIL size: {pil_image.size}")
    print(f"NumPy shape: {numpy_image.shape}")
    print(f"Bytes length: {len(image_bytes)}")
    print(f"Base64 length: {len(image_base64)}")
    print(f"Copied image: {copied_path}")
    print(f"Converted image: {converted_path}")


if __name__ == "__main__":
    main()
