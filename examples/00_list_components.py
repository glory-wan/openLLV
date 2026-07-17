"""List all components currently registered by openLLV."""

import json

from _utils import ROOT  # noqa: F401

import openLLV as llv


def main() -> None:
    """Print models, algorithms, metrics, losses, and datasets as JSON."""
    # print(json.dumps(llv.list_available(), ensure_ascii=False, indent=2))
    print(llv.list_available())


if __name__ == "__main__":
    main()
