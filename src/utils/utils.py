import argparse

PYTHON_DATASETS = ["humaneval", "bigcodebench", "skcriteria"]
R_DATASETS = ["rmcda"]

def get_suffix_of_dataset(args: argparse.Namespace) -> str:
    """Get the file suffix based on the dataset type."""
    if args.dataset in PYTHON_DATASETS:  # python
        return "py"
    elif args.dataset in R_DATASETS:
        return "R"
    else:
        raise ValueError(
            f"Dataset '{args.dataset}' is not supported. Please use 'humaneval', 'bigcodebench', or 'rmcda'."
        )


def get_sep_of_dataset(args: argparse.Namespace) -> str:
    """
    Get the library separator based on the dataset type.
    For example, for python and R, the separator is '.', while for C/C++ it is /.
    """
    if args.dataset in PYTHON_DATASETS:
        return "."
    elif args.dataset in R_DATASETS:
        return "."
    else:
        raise ValueError(
            f"Dataset '{args.dataset}' is not supported. Please use 'humaneval', 'bigcodebench', or 'rmcda'."
        )
