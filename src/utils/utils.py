import argparse
import os

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


def get_MR_file_path(args: argparse.Namespace, function_info: dict):
    """
    Get the folder path to store the MR of function.
    """
    if function_info["type"] == "function" or function_info["type"] == "local_function":
        mr_folder = os.path.join(
            args.output_dir,
            args.strategy,
            "metamorphic_relations",
            function_info["package"].replace(get_sep_of_dataset(args), os.sep),
            f"{function_info['name']}_mrs.json",
        )
    elif function_info["type"] == "class_method":
        mr_folder = os.path.join(
            args.output_dir,
            args.strategy,
            "metamorphic_relations",
            function_info["package"].replace(get_sep_of_dataset(args), os.sep),
            function_info["class_name"],
            f"{function_info['name']}_mrs.json",
        )
    else:
        raise ValueError(
            f"Function type '{function_info['type']}' is not supported. Please use 'function', 'local_function', or 'class_method'."
        )
    return mr_folder


def get_test_program_template_file_path(
    args: argparse.Namespace, function_info: dict, mr_id: str
):
    """
    Get the folder path to store the test program template of function.
    """
    file_suffix = get_suffix_of_dataset(args)
    test_program_template_file_name = f"test_{mr_id}.{file_suffix}.template"

    if function_info["type"] == "function" or function_info["type"] == "local_function":
        template_folder = os.path.join(
            args.output_dir,
            args.strategy,
            "test_program_templates",
            function_info["package"].replace(get_sep_of_dataset(args), os.sep),
            f"test_{function_info['name']}",
            test_program_template_file_name,
        )
    elif function_info["type"] == "class_method":
        template_folder = os.path.join(
            args.output_dir,
            args.strategy,
            "test_program_templates",
            function_info["package"].replace(get_sep_of_dataset(args), os.sep),
            function_info["class_name"],
            f"test_{function_info['name']}",
            test_program_template_file_name,
        )
    else:
        raise ValueError(
            f"Function type '{function_info['type']}' is not supported. Please use 'function', 'local_function', or 'class_method'."
        )
    return template_folder


def get_mr_evaluate_result_file_path(
    args: argparse.Namespace, function_info: dict
) -> str:
    """
    Get the path to store the MR evaluation results.
    """
    if function_info["type"] == "function" or function_info["type"] == "local_function":
        mr_evaluate_result_path = os.path.join(
            args.output_dir,
            args.strategy,
            "mr_evaluate_results",
            function_info["package"].replace(get_sep_of_dataset(args), os.sep),
            f"{function_info['name']}_mr_evaluate_results.json",
        )
    elif function_info["type"] == "class_method":
        mr_evaluate_result_path = os.path.join(
            args.output_dir,
            args.strategy,
            "mr_evaluate_results",
            function_info["package"].replace(get_sep_of_dataset(args), os.sep),
            function_info["class_name"],
            f"{function_info['name']}_mr_evaluate_results.json",
        )
    else:
        raise ValueError(
            f"Function type '{function_info['type']}' is not supported. Please use 'function', 'local_function', or 'class_method'."
        )

    return mr_evaluate_result_path


def get_test_program_instance_folder_path(
    args: argparse.Namespace, function_info: dict, mr_id: str
) -> str:

    if function_info["type"] == "function" or function_info["type"] == "local_function":
        test_program_instance_folder = os.path.join(
            args.output_dir,
            args.strategy,
            "test_program_instances",
            function_info["package"].replace(get_sep_of_dataset(args), os.sep),
            f"test_{function_info['name']}",
            f"mr_{mr_id}",
        )
    elif function_info["type"] == "class_method":
        test_program_instance_folder = os.path.join(
            args.output_dir,
            args.strategy,
            "test_program_instances",
            function_info["package"].replace(get_sep_of_dataset(args), os.sep),
            function_info["class_name"],
            f"test_{function_info['name']}",
            f"mr_{mr_id}",
        )
    else:
        raise ValueError(
            f"Function type '{function_info['type']}' is not supported. Please use 'function', 'local_function', or 'class_method'."
        )

    return test_program_instance_folder
