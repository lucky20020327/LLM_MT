import os
import json
import datetime
import sys
import signal
import traceback
import subprocess

from loguru import logger
from openai import OpenAI


# TODO: Note that global variables in the test prefix are not taken into consideration now.

function_source_input_generator_prompt = open(
    "./template/function_source_input_generator.prompt", "r", encoding="utf-8"
).read()
function_followup_input_generator_prompt = open(
    "./template/function_followup_input_generator.prompt", "r", encoding="utf-8"
).read()
function_valid_code_prompt = open(
    "./template/function_valid_code.prompt", "r", encoding="utf-8"
).read()
function_mr_prompt = open("./template/function_mr.prompt", "r", encoding="utf-8").read()
# This is used for functions are not in a public package. So the source_code of the function is needed to be imported.
local_function_test_program_template = open(
    "./template/local_function_test_program.template", "r", encoding="utf-8"
).read()


baseLLM = "deepseek"
api_key = "sk-ca15c4d0bec041c0b118a2ec0f69d388"

TEST_COUNT_PER_MR = 10


# Define a timeout exception
class TimeoutException(Exception):
    pass


# Define signal handler
def handler(signum, frame):
    raise TimeoutException("Execution timed out!")


def parse_mr_response(response: str):
    """
    Parse the metamorphic relations from the response string.
    The response is expected to be in JSON format.
    """
    identifier = "```mrs"
    if "```mrs" not in response:
        identifier = "```python"
    try:
        mr_list = response.split(identifier)[-1].split("```")[0].strip()
        MRs = json.loads(mr_list)
        if not isinstance(MRs, list):
            raise ValueError("Metamorphic relations should be a list.")
        return MRs
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse metamorphic relations: {e}")
        logger.error(f"Response content: {response}")
        raise ValueError("Failed to parse metamorphic relations from the response.")


def parse_function_response(
    response: str, identifier: str, target_function_signature: str
):
    """
    Parse the function code from the response string.
    The response is expected to contain a code block with the specified identifier.
    """
    if identifier not in response:
        identifier = "```python"
    try:
        function_code = response.split(identifier)[-1].split("```")[0].strip()
        if target_function_signature not in function_code:
            logger.error(
                f"Function signature '{target_function_signature}' not found in the response."
            )
            raise ValueError(
                f"Function signature '{target_function_signature}' not found in the response."
            )
        return function_code
    except Exception as e:
        logger.error(f"Failed to parse function code: {e}")
        logger.error(f"Response content: {response}")
        raise ValueError("Failed to parse function code from the response.")


def call_LLM(prompt, api_key, baseLLM="deepseek"):
    if baseLLM == "deepseek":
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        model_name = "deepseek-chat"
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def gen_MR_for_function(function_info: dict):
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

    function_full_name = function_info["name"]
    function_signature = function_info["signature"]
    function_docstring = function_info["docstring"]

    function_name = function_full_name.split(".")[-1]
    module_name = ".".join(function_full_name.split(".")[:-1])

    mr_prompt = function_mr_prompt.format(
        function_name=function_name,
        module_name=module_name,
        function_signature=function_signature,
        function_docstring=function_docstring,
    )

    logger.debug(f"MR generator prompt: {mr_prompt}")
    response = call_LLM(mr_prompt, baseLLM=baseLLM, api_key=api_key)
    logger.debug(f"Response from LLM: {response}")

    MRs = parse_mr_response(response)

    logger.info(
        f"Generated {len(MRs)} metamorphic relations for function {function_info['name']}"
    )
    return MRs


def gen_source_input_for_function(mr: dict, function_info: dict):
    """
    Generate source input generator for a function based on the given metamorphic relation and function information.
    This function should return a generator function as a string that can be used to generate source input.
    """
    logger.info(
        f"Generating source input generator for function {function_info['name']} with metamorphic relation {mr['mr_input_relation']}"
    )

    function_full_name = function_info["name"]
    function_signature = function_info["signature"]
    function_docstring = function_info["docstring"]

    function_name = function_full_name.split(".")[-1]
    module_name = ".".join(function_full_name.split(".")[:-1])

    source_input_constraints = mr["source_input_constraints"]

    source_input_generator_prompt = function_source_input_generator_prompt.format(
        function_name=function_name,
        module_name=module_name,
        function_signature=function_signature,
        function_docstring=function_docstring,
        input_constraints=source_input_constraints,
    )

    logger.debug(f"Source input generator prompt: {source_input_generator_prompt}")
    response = call_LLM(source_input_generator_prompt, baseLLM=baseLLM, api_key=api_key)
    logger.debug(f"Response from LLM: {response}")

    generator = parse_function_response(
        response,
        identifier="```source_input_generator",
        target_function_signature="def source_input_generator(k: int)",
    )
    logger.info(f"Generated source input generator: {generator}")
    return generator


def gen_followup_input_for_function(mr: dict, function_info: dict):
    """
    Generate follow-up input generator for a function based on the given metamorphic relation, function information, and source input.
    This function should return a generator function as a string that can be used to generate follow-up input.
    """
    logger.info(
        f"Generating follow-up input generator for function {function_info['name']} with metamorphic relation {mr['mr_input_relation']}, {mr['mr_output_relation']}"
    )

    function_full_name = function_info["name"]
    function_signature = function_info["signature"]
    function_docstring = function_info["docstring"]

    function_name = function_full_name.split(".")[-1]
    module_name = ".".join(function_full_name.split(".")[:-1])

    followup_input_constraints = mr["followup_input_constraints"]

    followup_input_generator_prompt = function_followup_input_generator_prompt.format(
        function_name=function_name,
        module_name=module_name,
        function_signature=function_signature,
        function_docstring=function_docstring,
        input_metamorphic_relation=mr["mr_input_relation"],
        input_transformation_steps=mr["mr_input_transformation_steps"],
        input_constraints=followup_input_constraints,
    )

    logger.debug(f"Follow-up input generator prompt: {followup_input_generator_prompt}")
    response = call_LLM(
        followup_input_generator_prompt, baseLLM=baseLLM, api_key=api_key
    )
    logger.debug(f"Response from LLM: {response}")

    generator = parse_function_response(
        response,
        identifier="```followup_input_generator",
        target_function_signature="def followup_input_generator(source_input: dict)",
    )
    logger.info(f"Generated follow-up input generator: {generator}")

    return generator


def gen_valid_code_for_function(mr: dict, function_info: dict):
    logger.info(
        f"Generating valid code for function {function_info['name']} with metamorphic relation {mr['mr_input_relation']}"
    )
    function_full_name = function_info["name"]
    function_signature = function_info["signature"]
    function_docstring = function_info["docstring"]

    function_name = function_full_name.split(".")[-1]
    module_name = ".".join(function_full_name.split(".")[:-1])

    valid_code_prompt = function_valid_code_prompt.format(
        function_name=function_name,
        module_name=module_name,
        function_signature=function_signature,
        function_docstring=function_docstring,
        input_metamorphic_relation=mr["mr_input_relation"],
        input_transformation_steps=mr["mr_input_transformation_steps"],
        output_metamorphic_relation=mr["mr_output_relation"],
        output_validation_steps=mr["mr_output_validation_steps"],
    )

    logger.debug(f"Valid code prompt: {valid_code_prompt}")
    response = call_LLM(valid_code_prompt, baseLLM=baseLLM, api_key=api_key)
    logger.debug(f"Response from LLM: {response}")

    valid_code = parse_function_response(
        response,
        identifier="```validate_MR_result",
        target_function_signature="def validate_MR_result(source_input, followup_input, source_result, followup_result)",
    )

    logger.info(f"Generated valid code: {valid_code}")
    return valid_code


def test_program_construction_for_local_function(
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

    mr_str = json.dumps(mr, indent=4)

    function_full_name = function_info["name"]
    function_source_code = function_info["source_code"]

    function_name = function_full_name.split(".")[-1]

    test_program = local_function_test_program_template.format(
        metamorphic_relation=mr_str,
        function_source_code=function_source_code,
        source_input_code=source_input_generator,
        followup_input_code=followup_input_generator,
        validate_result_code=valid_code,
        function_name=function_name,
        input_count=TEST_COUNT_PER_MR,
    )

    logger.debug(f"Generated test program: {test_program}")
    logger.info(
        f"Test program for local function {function_info['name']} with metamorphic relation {mr['mr_input_relation']} constructed successfully."
    )

    return test_program


def gen_test_template_for_local_function(function_info: dict):
    """
    This function will build a test program template for the given local function using metamorphic relations.
    The template can be used to generate test program instances for mutants of the function by formatting {function_source_code}.
    """

    function_full_name = function_info["name"]
    function_name = function_full_name.split(".")[-1]
    # For local functions, the module name is the customized path representing the dataset architecture.
    # For example, the function from dataset humaneval is named as humaneval.<function_name>.
    module_name = ".".join(function_full_name.split(".")[:-1])
    test_program_template_folder = os.path.join(
        ".",
        "simple_output",
        "test_program_templates",
        module_name.replace(".", os.sep),
        f"test_{function_name}",
    )
    mr_folder = os.path.join(
        ".", "simple_output", "metamorphic_relations", module_name.replace(".", os.sep)
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
        MRs = gen_MR_for_function(function_info)
        logger.info(f"Writing metamorphic relations to {mr_file_path}")
        with open(mr_file_path, "w", encoding="utf-8") as f:
            json.dump(MRs, f, indent=4)

    for mr_id, mr in enumerate(MRs):
        test_program_template_file_name = f"test_{mr_id}.py.template"
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

        source_input_code = gen_source_input_for_function(mr, function_info)
        followup_input_code = gen_followup_input_for_function(mr, function_info)
        valid_code = gen_valid_code_for_function(mr, function_info)

        test_program = test_program_construction_for_local_function(
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


def execute_test_program(test_program_file_path: str):
    """
    Execute the test program and return the result.
    This function should handle any exceptions that occur during execution.
    """
    try:
        # execute the test program using subprocess
        logger.info(f"Executing test program {test_program_file_path}")
        result = subprocess.run(
            ["python", test_program_file_path],
            capture_output=True,
            text=True,
            timeout=5,  # Set a timeout for the execution
        )
        if result.returncode != 0:
            logger.error(
                f"Test program {test_program_file_path} failed with return code {result.returncode}"
            )
            return False, result.stderr.strip()
        logger.info(f"Test program {test_program_file_path} executed successfully.")
        return True, result.stdout.strip()

    except Exception as e:
        logger.warning(f"Error executing test program {test_program_file_path}: {e}")
        return False, str(e)


def evaluate_mr(function_info: dict):
    """
    Evaluate the metamorphic relations on a function and its mutants.
    This function should execute the test program and check if the metamorphic relations hold.
    """
    function_full_name = function_info["name"]
    function_name = function_full_name.split(".")[-1]
    module_name = ".".join(function_full_name.split(".")[:-1])

    test_program_template_folder = os.path.join(
        ".",
        "simple_output",
        "test_program_templates",
        module_name.replace(".", os.sep),
        f"test_{function_name}",
    )

    mr_folder = os.path.join(
        ".", "simple_output", "metamorphic_relations", module_name.replace(".", os.sep)
    )

    mr_file_name = f"{function_name}_mrs.json"
    mr_file_path = os.path.join(mr_folder, mr_file_name)

    if not os.path.exists(mr_file_path):
        logger.error(f"Metamorphic relations file {mr_file_path} does not exist.")
        raise FileNotFoundError(
            f"Metamorphic relations file {mr_file_path} does not exist."
        )

    mr_evaluate_results = {}
    mr_evaluate_results_file_path = os.path.join(
        ".",
        "simple_output",
        "mr_evaluate_results",
        module_name.replace(".", os.sep),
        f"{function_name}_mr_evaluate_results.json",
    )
    os.makedirs(os.path.dirname(mr_evaluate_results_file_path), exist_ok=True)

    with open(mr_file_path, "r", encoding="utf-8") as f:
        MRs = json.load(f)

    for mr_id, mr in enumerate(MRs):

        mr_evaluate_results[mr_id] = {
            "mr": mr,
            "valid_mr": False,
            "mutant_detection_results": {},
        }

        test_program_template_file_name = f"test_{mr_id}.py.template"
        test_program_template_file_path = os.path.join(
            test_program_template_folder, test_program_template_file_name
        )

        if not os.path.exists(test_program_template_file_path):
            logger.error(
                f"Test program template file {test_program_template_file_path} does not exist."
            )
            raise FileNotFoundError(
                f"Test program template file {test_program_template_file_path} does not exist."
            )

        test_program_instance_folder = os.path.join(
            ".",
            "simple_output",
            "test_program_instances",
            module_name.replace(".", os.sep),
            f"test_{function_name}",
            f"mr_{mr_id}",
        )
        os.makedirs(test_program_instance_folder, exist_ok=True)

        test_program_template = open(
            test_program_template_file_path, "r", encoding="utf-8"
        ).read()

        # Execute the test program on original function and check if the metamorphic relations hold
        original_test_program_file_path = test_program_template.replace(
            "{function_source_code}",
            function_info["source_code"],
        )
        original_test_program_file_name = f"original.py"
        with open(
            os.path.join(test_program_instance_folder, original_test_program_file_name),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(original_test_program_file_path)
        logger.info(
            f"Writing original test program instance to {os.path.join(test_program_instance_folder, original_test_program_file_name)}"
        )

        execute_result, execute_output = execute_test_program(
            os.path.join(test_program_instance_folder, original_test_program_file_name)
        )
        if not execute_result:
            logger.error(f"Test program for original function failed.")
            mr_evaluate_results[mr_id]["error_message"] = execute_output
            continue
        mr_evaluate_results[mr_id]["valid_mr"] = True
        # Now, we need to execute the test program on each mutant of the function
        if "mutations" not in function_info:
            logger.error(
                f"Mutations field is missing in the function info for {function_full_name}."
            )
            raise ValueError(
                f"Mutations field is missing in the function info for {function_full_name}."
            )
        for mutation in function_info["mutations"]:
            mutation_name = mutation["name"]
            mutation_source_code = mutation["source_code"]

            test_program_instance_file_path = test_program_template.replace(
                "{function_source_code}",
                mutation_source_code,
            )  # Replace the function source code with the mutation source code
            test_program_instance_file_name = f"{mutation_name}.py"
            with open(
                os.path.join(
                    test_program_instance_folder, test_program_instance_file_name
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(test_program_instance_file_path)
            logger.info(
                f"Writing test program instance for mutation {mutation_name} to {os.path.join(test_program_instance_folder, test_program_instance_file_name)}"
            )

            # Execute the test program on the mutant and check if the metamorphic relation holds
            execute_result, execute_output = execute_test_program(
                os.path.join(
                    test_program_instance_folder, test_program_instance_file_name
                )
            )
            if not execute_result:
                logger.error(f"Test program for mutation {mutation_name} passed.")
                mr_evaluate_results[mr_id]["mutant_detection_results"][
                    mutation_name
                ] = False
            else:
                logger.info(
                    f"Test program for mutation {mutation_name} failed. It indicates that the metamorphic relation detects the mutation."
                )
                # If the test program passed, we can check if the metamorphic relation holds
                mr_evaluate_results[mr_id]["mutant_detection_results"][
                    mutation_name
                ] = True

    with open(mr_evaluate_results_file_path, "w", encoding="utf-8") as f:
        json.dump(mr_evaluate_results, f, indent=4)
    logger.info(
        f"Metamorphic relation evaluation results saved to {mr_evaluate_results_file_path}"
    )


if __name__ == "__main__":

    logger.remove()
    now_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.add(
        os.path.join(
            ".",
            "logs",
            "simple_mr",
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

    api_file = "/Users/lucky/work/ZJU/2025_04_23_metamorphic_testing/LLM_based_MT/dataset/humaneval/humaneval_mutated.json"
    api_infos = json.load(open(api_file, "r", encoding="utf-8"))

    logger.info(f"Loaded {len(api_infos)} API infos from {api_file}")

    for api_info in api_infos:
        # if api_info["name"] != "humaneval.add_elements":
        #     continue  # For testing, we only process the add_elements function.
        assert (
            api_info["type"] == "local_function"
        ), "Only local functions are supported in this script."
        logger.info(f"Processing local function: {api_info['name']}")
        try:
            gen_test_template_for_local_function(api_info)
        except Exception as e:
            logger.error(
                f"Error generating test template for function {api_info['name']}: {e}"
            )
            logger.debug(traceback.format_exc())
            continue

    for api_info in api_infos:
        # if api_info["name"] != "humaneval.add_elements":
        #     continue  # For testing, we only process the add_elements function.
        assert "mutations" in api_info, "Mutations field is missing in the API info."
        logger.info(
            f"Evaluating metamorphic relations for function: {api_info['name']}"
        )
        try:
            evaluate_mr(api_info)
        except Exception as e:
            logger.error(
                f"Error evaluating metamorphic relations for function {api_info['name']}: {e}"
            )
            logger.error(traceback.format_exc())
            continue
    logger.info("Finished processing all local functions.")
