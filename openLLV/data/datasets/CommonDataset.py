"""Common paired dataset implementation."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from openLLV.data.datasets.BaseDataset import BaseDataset


class CommonDataset(BaseDataset):
    """Common paired input/target dataset.

    This dataset supports common directory layouts used by paired datasets.

    The preferred paired layout is:
        root_dir/
            train/
                input/
                    train1.jpg
                    train2.jpg
                target/
                    train1.jpg
                    train2.jpg
            val/
                input/
                    val1.jpg
                    val2.jpg
                target/
                    val1.jpg
                    val2.jpg
    """

    aliases = ["common", "paired", "common_dataset"]

    valid_names = ("test", "Test", "val", "Val", "validation")

    split_aliases = {
        "train": ("train", "Train"),
        "training": ("train", "Train"),
        "_test": valid_names,
        "eval": valid_names,
        "val": valid_names,
        "valid": valid_names,
        "validation": valid_names,
    }

    input_dir_names = ("input", "Input")
    target_dir_names = ("target", "Target")

    def _resolve_pair_dirs(
            self,
            low_dir: Optional[Path],
            high_dir: Optional[Path],
    ) -> Tuple[Path, Optional[Path]]:
        """Resolve low-light and normal-light image directories.

        Args:
            low_dir: Optional explicit low-light image directory. When provided,
                it is returned directly with ``high_dir``.
            high_dir: Optional explicit normal-light image directory.

        Returns:
            A tuple containing the resolved low-light directory and optional
            normal-light directory. The normal-light directory is None for_teach
            unpaired datasets.
        """
        if low_dir is not None:
            return low_dir, high_dir

        split_dirs = self.split_aliases.get(self.split.lower(), (self.split,))
        candidates = []

        for split_dir in split_dirs:
            split_path = self.root_dir / split_dir
            for low_name in self.input_dir_names:
                for high_name in self.target_dir_names:
                    candidates.append((split_path / low_name, split_path / high_name))

        # Some local datasets are already organized as root/low and root/high.
        for low_name in self.input_dir_names:
            for high_name in self.target_dir_names:
                candidates.append((self.root_dir / low_name, self.root_dir / high_name))

        for candidate_low, candidate_high in candidates:
            if candidate_low.exists() and candidate_high.exists():
                return candidate_low, candidate_high

        # Allow unpaired datasets organized as root/split/low or root/low.
        for candidate_low, _ in candidates:
            if candidate_low.exists():
                return candidate_low, None

        # Return the most likely layout so BaseDataset raises a clear path error.
        first_split = split_dirs[0]
        return self.root_dir / first_split / "low", self.root_dir / first_split / "high"

    def get_stats(self) -> Dict[str, Union[str, int]]:
        """Get CommonDataset statistics.

        Returns:
            Dictionary containing base dataset statistics and the split aliases
            used when resolving image directories.
        """
        stats = super().get_stats()
        stats["split_aliases"] = ", ".join(self.split_aliases.get(self.split.lower(), (self.split,)))
        return stats

