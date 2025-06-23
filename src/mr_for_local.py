import os
import json
import datetime
import sys
import traceback
import subprocess
import argparse

from loguru import logger
from openai import OpenAI

from utils.utils import get_suffix_of_dataset
from utils.llm import call_LLM
from utils.extract_info import process_api

from template import (
    get_prompt_template_formatter,
    get_parser,
    BaseFormatter,
    BaseParser,
)

from mr_evaluator import get_evaluator, BaseEvaluator

pwd = os.path.dirname(os.path.abspath(__file__))

# global variables for formatter and parser
template_formatter: BaseFormatter = None
response_parser: BaseParser = None

# global variable for mr evaluator
mr_evaluator: BaseEvaluator = None


# TODO: Note that global variables in the test prefix are not taken into consideration now.


def gen_MR_for_function(args: argparse.Namespace, function_info: dict):
    """
    Generate metamorphic relations for a function.
    This function should return a list of metamorphic relations that can be used to generate test cases.

    Return value is a list of dictionaries with the following keys:
    - "mr_input_relation": A string describing the metamorphic input relation .
    - "mr_input_transformation_steps": A string describing the metamorphic input transformation steps.
    - "mr_output_relation": A string describing the metamorphic output relation.
    - "mr_output_validation_steps": A string describing the metamorphic output validation steps.
    - "source_input_constraints": A string describing the constraints on the source input.
    - "followup_input_constraints": A string describing the constraints on the follow-up input.
    """

    logger.info(
        f"Generating metamorphic relations for function {function_info['name']}"
    )

    if args.strategy == "func_deep_report":
        process_api(function_info)
        function_analysis_report = (
            template_formatter.function_deep_report_template_formatter(
                args=args, function_info=function_info
            )
        )
    elif args.strategy == "simple":
        function_analysis_report = ""
    else:
        raise ValueError(
            f"Unknown strategy {args.strategy}. Please use 'simple' or 'func_deep_report'."
        )

    mr_prompt = template_formatter.function_mr_prompt_formatter(
        args=args,
        function_info=function_info,
        function_analysis_report=function_analysis_report,
    )

    logger.debug(f"MR generator prompt: {mr_prompt}")
    response = call_LLM(mr_prompt, baseLLM=args.baseLLM, api_key=args.api_key)
    logger.debug(f"Response from LLM: {response}")

    MRs = response_parser.parse_response(response, target="mr")

    logger.info(
        f"Generated {len(MRs)} metamorphic relations for function {function_info['name']}"
    )
    return MRs


def gen_source_input_for_function(
    args: argparse.Namespace, mr: dict, function_info: dict
):
    """
    Generate source input generator for a function based on the given metamorphic relation and function information.
    This function should return a generator function as a string that can be used to generate source input.
    """
    logger.info(
        f"Generating source input generator for function {function_info['name']} with metamorphic relation {mr['mr_input_relation']}"
    )

    source_input_generator_prompt = (
        template_formatter.function_source_input_generator_prompt_formatter(
            args=args, mr=mr, function_info=function_info
        )
    )

    logger.debug(f"Source input generator prompt: {source_input_generator_prompt}")
    response = call_LLM(
        source_input_generator_prompt, baseLLM=args.baseLLM, api_key=args.api_key
    )
    logger.debug(f"Response from LLM: {response}")

    generator = response_parser.parse_response(
        response,
        target="source_input_generator",
    )
    logger.info(f"Generated source input generator: {generator}")
    return generator


def gen_followup_input_for_function(
    args: argparse.Namespace, mr: dict, function_info: dict
):
    """
    Generate follow-up input generator for a function based on the given metamorphic relation, function information, and source input.
    This function should return a generator function as a string that can be used to generate follow-up input.
    """
    logger.info(
        f"Generating follow-up input generator for function {function_info['name']} with metamorphic relation {mr['mr_input_relation']}, {mr['mr_output_relation']}"
    )
    followup_input_generator_prompt = (
        template_formatter.function_followup_input_generator_prompt_formatter(
            args=args, mr=mr, function_info=function_info
        )
    )

    logger.debug(f"Follow-up input generator prompt: {followup_input_generator_prompt}")
    response = call_LLM(
        followup_input_generator_prompt, baseLLM=args.baseLLM, api_key=args.api_key
    )
    logger.debug(f"Response from LLM: {response}")

    generator = response_parser.parse_response(
        response,
        target="followup_input_generator",
    )
    logger.info(f"Generated follow-up input generator: {generator}")

    return generator


def gen_valid_code_for_function(
    args: argparse.Namespace, mr: dict, function_info: dict
):
    logger.info(
        f"Generating valid code for function {function_info['name']} with metamorphic relation {mr['mr_input_relation']}"
    )
    valid_code_prompt = template_formatter.function_valid_code_prompt_formatter(
        args=args, mr=mr, function_info=function_info
    )

    logger.debug(f"Valid code prompt: {valid_code_prompt}")
    response = call_LLM(valid_code_prompt, baseLLM=args.baseLLM, api_key=args.api_key)
    logger.debug(f"Response from LLM: {response}")

    valid_code = response_parser.parse_response(
        response,
        target="validate_MR_result",
    )

    logger.info(f"Generated valid code: {valid_code}")
    return valid_code


def test_program_construction_for_local_function(
    args: argparse.Namespace,
    function_info: dict,
    mr: dict,
    source_input_generator: str,
    followup_input_generator: str,
    valid_code: str,
):
    """
    Construct a test program for a local function using the generated source input, follow-up input, and valid code.
    This function should return a string representing the test program.
    """
    logger.info(
        f"Constructing test program for local function {function_info['name']} with metamorphic relation {mr['mr_input_relation']}"
    )

    test_program = template_formatter.local_function_test_program_template_formatter(
        args=args,
        function_info=function_info,
        mr=mr,
        source_input_generator=source_input_generator,
        followup_input_generator=followup_input_generator,
        valid_code=valid_code,
    )

    logger.debug(f"Generated test program: {test_program}")
    logger.info(
        f"Test program for local function {function_info['name']} with metamorphic relation {mr['mr_input_relation']} constructed successfully."
    )

    return test_program


def gen_test_template_for_local_function(args: argparse.Namespace, function_info: dict):
    """
    This function will build a test program template for the given local function using metamorphic relations.
    The template can be used to generate test program instances for mutants of the function by formatting {function_source_code}.
    """

    function_name = function_info["name"]
    # For local functions, the module name is the customized path representing the dataset architecture.
    # For example, the function from dataset humaneval is named as humaneval.<function_name>.
    module_name = function_info["package"]
    test_program_template_folder = os.path.join(
        args.output_dir,
        args.strategy,
        "test_program_templates",
        module_name.replace(".", os.sep),
        f"test_{function_name}",
    )
    mr_folder = os.path.join(
        args.output_dir,
        args.strategy,
        "metamorphic_relations",
        module_name.replace(".", os.sep),
    )
    os.makedirs(test_program_template_folder, exist_ok=True)
    os.makedirs(mr_folder, exist_ok=True)

    # save the metamorphic relations to a JSON file
    mr_file_name = f"{function_name}_mrs.json"
    mr_file_path = os.path.join(mr_folder, mr_file_name)

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
        MRs = gen_MR_for_function(args, function_info)
        logger.info(f"Writing metamorphic relations to {mr_file_path}")
        with open(mr_file_path, "w", encoding="utf-8") as f:
            json.dump(MRs, f, indent=4)

    file_suffix = get_suffix_of_dataset(args)

    for mr_id, mr in enumerate(MRs):
        test_program_template_file_name = f"test_{mr_id}.{file_suffix}.template"
        test_program_template_file_path = os.path.join(
            test_program_template_folder, test_program_template_file_name
        )

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

        source_input_code = gen_source_input_for_function(args, mr, function_info)
        followup_input_code = gen_followup_input_for_function(args, mr, function_info)
        valid_code = gen_valid_code_for_function(args, mr, function_info)

        test_program = test_program_construction_for_local_function(
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
    parser = argparse.ArgumentParser(description="Simple MR for local functions.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
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
        choices=["simple", "func_deep_report"],
    )
    return parser


def logger_init(args: argparse.Namespace):
    logger.remove()
    now_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.add(
        os.path.join(
            args.log_base_dir,
            args.strategy,
            "mr_for_local",
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


# TODO: extend this function to support other languages in the future.
def global_init(args: argparse.Namespace):
    global template_formatter
    global response_parser
    global mr_evaluator

    dataset = args.dataset

    # Load template_formatter and parsers
    template_formatter = get_prompt_template_formatter(dataset)
    response_parser = get_parser(dataset)

    # Load MR evaluator
    mr_evaluator = get_evaluator(dataset)


if __name__ == "__main__":

    args = arg_parser().parse_args()
    logger.info(f"Starting Simple MR for local functions with arguments: {args}")

    logger_init(args)
    global_init(args)

    api_file = args.api_file
    api_infos = json.load(open(api_file, "r", encoding="utf-8"))

    logger.info(f"Loaded {len(api_infos)} API infos from {api_file}")

    # generate test program templates for local functions
    for api_info in api_infos:
        assert (
            api_info["type"] == "local_function"
        ), "Only local functions are supported in this script."
        logger.info(f"Processing local function: {api_info['name']}")
        try:
            gen_test_template_for_local_function(args, api_info)
        except Exception as e:
            logger.error(
                f"Error generating test template for function {api_info['name']}: {e}"
            )
            logger.debug(traceback.format_exc())
            continue

    # evaluate metamorphic relations for local functions
    for api_info in api_infos:
        assert "mutations" in api_info, "Mutations field is missing in the API info."
        logger.info(
            f"Evaluating metamorphic relations for function: {api_info['name']}"
        )
        try:
            mr_evaluator.evaluate(args, api_info)
        except Exception as e:
            logger.error(
                f"Error evaluating metamorphic relations for function {api_info['name']}: {e}"
            )
            logger.error(traceback.format_exc())
            continue
    logger.info("Finished processing all local functions.")
