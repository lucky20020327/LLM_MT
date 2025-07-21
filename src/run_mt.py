import os
import json
import datetime
import sys
import traceback
import argparse

from loguru import logger

from utils.utils import (
    get_MR_file_path,
    get_test_program_template_file_path,
    PYTHON_DATASETS,
    R_DATASETS,
)

from template import (
    get_prompt_template_formatter,
    BaseFormatter,
)

from mr_evaluator import get_evaluator, BaseEvaluator
from mr_generator import get_generator, BaseGenerator

pwd = os.path.dirname(os.path.abspath(__file__))

# global variables for formatter
template_formatter: BaseFormatter = None

# global variable for mr generator
mr_generator: BaseGenerator = None

# global variable for mr evaluator
mr_evaluator: BaseEvaluator = None


# TODO: Note that global variables in the test prefix are not taken into consideration now.


def test_program_construction(
    args: argparse.Namespace,
    function_info: dict,
    mr: dict,
    source_input_generator: str,
    followup_input_generator: str,
    valid_code: str,
):
    """
    Construct a test program for a function using the generated source input, follow-up input, and valid code.
    This function should return a string representing the test program.
    """
    logger.info(
        f"Constructing test program for function {function_info['name']} with type {function_info['type']} with metamorphic relation {mr['mr_input_relation']}"
    )

    test_program = template_formatter.test_program_template_formatter(
        function_info=function_info,
        mr=mr,
        source_input_generator=source_input_generator,
        followup_input_generator=followup_input_generator,
        valid_code=valid_code,
    )

    logger.debug(f"Generated test program: {test_program}")
    logger.info(
        f"Test program for function {function_info['name']} with metamorphic relation {mr['mr_input_relation']} constructed successfully."
    )

    return test_program


def gen_test_template(args: argparse.Namespace, function_info: dict):
    """
    This function will build a test program template for the given function using metamorphic relations.
    The template can be used to generate test program instances for mutants of the function by formatting {function_source_code}.
    """

    mr_file_path = get_MR_file_path(args, function_info)
    os.makedirs(os.path.dirname(mr_file_path), exist_ok=True)

    # save the metamorphic relations to a JSON file

    if os.path.exists(mr_file_path):
        logger.info(
            f"Metamorphic relations file {mr_file_path} already exists. Loading from file."
        )
        with open(mr_file_path, "r", encoding="utf-8") as f:
            MRs = json.load(f)
        if not isinstance(MRs, list):
            logger.error("Metamorphic relations should be a list.")
            raise ValueError("Metamorphic relations should be a list.")
        logger.info(f"Loaded {len(MRs)} metamorphic relations from {mr_file_path}")
    else:
        MRs = mr_generator.gen_MR(function_info)
        logger.info(f"Writing metamorphic relations to {mr_file_path}")
        with open(mr_file_path, "w", encoding="utf-8") as f:
            json.dump(MRs, f, indent=4)

    for mr_id, mr in enumerate(MRs):
        test_program_template_file_path = get_test_program_template_file_path(
            args, function_info, mr_id
        )
        os.makedirs(os.path.dirname(test_program_template_file_path), exist_ok=True)

        if os.path.exists(test_program_template_file_path):
            # read the content and check if the mr is inside the file
            with open(test_program_template_file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if mr["mr_input_relation"] in content:
                logger.info(
                    f"Test program template file {test_program_template_file_path} already exists. Skipping."
                )
                continue
            else:
                logger.warning(
                    f"Test program template file {test_program_template_file_path} exists but does not contain the metamorphic relation. Overwriting."
                )

        source_input_code = mr_generator.gen_source_input_generator(mr, function_info)
        followup_input_code = mr_generator.gen_followup_input_generator(
            mr, function_info
        )
        valid_code = mr_generator.gen_valid_code(mr, function_info)

        test_program = test_program_construction(
            args,
            function_info,
            mr,
            source_input_code,
            followup_input_code,
            valid_code,
        )
        logger.info(
            f"Writing test program template to {test_program_template_file_path}"
        )
        with open(test_program_template_file_path, "w", encoding="utf-8") as f:
            f.write(test_program)


def arg_parser():
    parser = argparse.ArgumentParser(description="Simple MT.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=PYTHON_DATASETS + R_DATASETS,
    )
    parser.add_argument(
        "--api_file",
        type=str,
        required=True,
        help="Path to the API info file.",
    )
    parser.add_argument(
        "--baseLLM",
        type=str,
        required=True,
        choices=["deepseek"],
        help="Base LLM to use for generating metamorphic relations and test programs.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for the LLM service.",
    )
    parser.add_argument(
        "--test_count_per_mr",
        type=int,
        default=10,
        help="Number of test cases to generate for each metamorphic relation.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Timeout for executing the test program in seconds.",
    )
    parser.add_argument(
        "--log_base_dir",
        type=str,
        default=os.path.join(pwd, "..", "logs", "simple_mr"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated test programs and metamorphic relations.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="simple",
        choices=["simple"],  # , "func_deep_report"],
    )
    return parser


def logger_init(args: argparse.Namespace):
    logger.remove()
    now_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.add(
        os.path.join(
            args.log_base_dir,
            args.strategy,
            "run_mt",
            f"{now_time_str}.log",
        ),
        level="DEBUG",
        format="{time} {level} {file}|{line}: {message}",
    )
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time}</green> <level>{level}</level> <cyan>{file}</cyan>|<magenta>{line}</magenta>: <level>{message}</level>",
        colorize=True,
    )


def global_init(args: argparse.Namespace):
    global template_formatter
    global mr_generator
    global mr_evaluator

    # Load template_formatter and parsers
    template_formatter = get_prompt_template_formatter(args=args)

    # Load MR generator
    mr_generator = get_generator(args=args)

    # Load MR evaluator
    mr_evaluator = get_evaluator(args=args)


if __name__ == "__main__":

    args = arg_parser().parse_args()
    logger.info(f"Starting Simple MT for functions with arguments: {args}")

    logger_init(args)
    global_init(args)

    api_file = args.api_file
    api_infos = json.load(open(api_file, "r", encoding="utf-8"))

    logger.info(f"Loaded {len(api_infos)} API infos from {api_file}")

    # generate test program templates for local functions/functions/classes
    for api_info in api_infos:
        logger.info(f"Processing: {api_info['name']} with type {api_info['type']}")
        try:
            if api_info["type"] in ["function", "local_function"]:
                gen_test_template(args, api_info)
            elif api_info["type"] == "class":
                for method in api_info["methods"]:
                    api_info_with_class = {
                        "type": "class_method",
                        "package": api_info["package"],
                        "class_name": api_info["name"],
                        "class_signature": api_info["signature"],
                        "class_docstring": api_info["docstring"],
                        **method,
                        "mutations": method.get("mutations", []),
                    }
                    gen_test_template(args, api_info_with_class)
            else:
                logger.error(
                    f"Function type {api_info['type']} is not supported. Skipping."
                )
                continue
        except Exception as e:
            logger.error(
                f"Error generating test template for function {api_info['name']}: {e}"
            )
            logger.debug(traceback.format_exc())
            continue

    # evaluate metamorphic relations for local functions/functions/classes
    for api_info in api_infos:
        logger.info(
            f"Evaluating metamorphic relations for function: {api_info['name']}"
        )
        try:
            if api_info["type"] in ["function", "local_function"]:
                api_info = {
                    **api_info,
                    "mutations": api_info.get("mutations", []),
                }
                mr_evaluator.evaluate(api_info)
            elif api_info["type"] == "class":
                for method in api_info["methods"]:
                    api_info_with_class = {
                        "type": "class_method",
                        "package": api_info["package"],
                        "class_name": api_info["name"],
                        "class_signature": api_info["signature"],
                        "class_docstring": api_info["docstring"],
                        **method,
                        "mutations": method.get("mutations", []),
                    }
                    mr_evaluator.evaluate(api_info_with_class)
            else:
                logger.error(
                    f"Function type {api_info['type']} is not supported for evaluation. Skipping."
                )
                continue
        except Exception as e:
            logger.error(
                f"Error evaluating metamorphic relations for function {api_info['name']}: {e}"
            )
            logger.error(traceback.format_exc())
            continue
    logger.info("Finished processing all functions.")
