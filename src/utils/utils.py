import argparse


def get_suffix_of_dataset(args: argparse.Namespace) -> str:
    if args.dataset in ["humaneval", "bigcodebench"]:
        return "py"
    elif args.dataset in ["rmcda"]:
        return "R"
    else:
        raise ValueError(
            f"Dataset '{args.dataset}' is not supported. Please use 'humaneval', 'bigcodebench', or 'rmcda'."
        )
