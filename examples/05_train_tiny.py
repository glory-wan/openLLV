"""Train Zero-DCE for one epoch on a generated tiny dataset."""

from _utils import create_tiny_common_dataset, ensure_results_dir

import torch

import openLLV as llv


def main() -> None:
    """Demonstrate Trainer construction through the top-level API."""
    dataset_root = create_tiny_common_dataset(
        ensure_results_dir("tiny_dataset")
    )
    output_dir = ensure_results_dir("checkpoints", "ZeroDCE_tiny")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = llv.train(
        model="ZeroDCE",
        model_params={"input_channels": 3},
        dataset="CommonDataset",
        root_dir=str(dataset_root),
        train_split="train",
        val_split="val",
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        loss="zerodce",
        optimizer="adam",
        lr=1e-4,
        epochs=1,
        device=device,
        output_dir=str(output_dir),
        log_every=1,
        save_every=1,
        validate_every=1,
        progress_bar=False,
    )

    print(f"Device: {device}")
    print(f"Training history: {result['history']}")
    print(f"Checkpoint directory: {result['checkpoint_dir']}")


if __name__ == "__main__":
    main()
