"""Common paired dataset implementation."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

    aliases = ["PairedDataset"]

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
            input_dir: Optional[Path],
            target_dir: Optional[Path],
    ) -> Tuple[Path, Optional[Path]]:
        """Resolve input and target image directories.

        Args:
            input_dir: Optional explicit input image directory. When provided,
                it is returned directly with ``target_dir``.
            target_dir: Optional explicit target image directory.

        Returns:
            The resolved input directory and optional target directory. The
            target directory is ``None`` for unpaired datasets.
        """
        if input_dir is not None:
            return input_dir, target_dir

        split_dirs = self.split_aliases.get(self.split.lower(), (self.split,))
        candidates = []

        for split_dir in split_dirs:
            split_path = self.root_dir / split_dir
            for input_name in self.input_dir_names:
                for target_name in self.target_dir_names:
                    candidates.append(
                        (split_path / input_name, split_path / target_name)
                    )

        # Some datasets are already organized as root/input and root/target.
        for input_name in self.input_dir_names:
            for target_name in self.target_dir_names:
                candidates.append(
                    (self.root_dir / input_name, self.root_dir / target_name)
                )

        for candidate_input, candidate_target in candidates:
            if candidate_input.exists() and candidate_target.exists():
                return candidate_input, candidate_target

        # Allow unpaired datasets organized as root/split/input or root/input.
        for candidate_input, _ in candidates:
            if candidate_input.exists():
                return candidate_input, None

        # Return the most likely layout so BaseDataset raises a clear path error.
        first_split = split_dirs[0]
        return (
            self.root_dir / first_split / "input",
            self.root_dir / first_split / "target",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get CommonDataset statistics.

        Returns:
            Dictionary containing base dataset statistics and the split aliases
            used when resolving image directories.
        """
        stats = super().get_stats()
        split_aliases = self.split_aliases.get(
            self.split.lower(),
            (self.split,),
        )
        stats["split_aliases"] = ", ".join(split_aliases)
        return stats
