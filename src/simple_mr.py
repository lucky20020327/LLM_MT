import os
import json
import datetime
import sys

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
# The following templates are used to generate the test program for a function.
function_test_program_template = open(
    "./template/function_test_program.template", "r", encoding="utf-8"
).read()


baseLLM = "deepseek"
api_key = 

TEST_COUNT_PER_MR = 10


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


def gen_MR_for_class_method(method_info: dict, class_info: dict):
    """
    Generate metamorphic relations for a class method.
    This function should return a list of metamorphic relations that can be used to generate test cases.

    Return value is a list of dictionaries with the following keys:
    - "mr_input_relation": A string describing the metamorphic relation input.
    - "mr_output_relation": A string describing the metamorphic relation output.
    - "source_input_constraints": A string describing the constraints on the source input.
    - "followup_input_constraints": A string describing the constraints on the follow-up input.
    """
    return []


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


def test_program_construction_for_function(
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
        f"Constructing test program for function {function_info['name']} with metamorphic relation {mr['mr_input_relation']}"
    )

    mr_str = json.dumps(mr, indent=4)

    function_full_name = function_info["name"]

    function_name = function_full_name.split(".")[-1]
    module_name = ".".join(function_full_name.split(".")[:-1])

    import_statement = f"from {module_name} import {function_name}\n"

    test_program = function_test_program_template.format(
        import_statements=import_statement,
        metamorphic_relation=mr_str,
        source_input_code=source_input_generator,
        followup_input_code=followup_input_generator,
        validate_result_code=valid_code,
        function_name=function_name,
        input_count=TEST_COUNT_PER_MR,
    )

    logger.debug(f"Generated test program: {test_program}")
    logger.info(
        f"Test program for function {function_info['name']} with metamorphic relation {mr['mr_input_relation']} constructed successfully."
    )

    return test_program


def MT_for_function(function_info: dict):
    """
    MT for a function.
    This function will build a test program for the given function using metamorphic relations.
    """

    function_full_name = function_info["name"]
    function_name = function_full_name.split(".")[-1]
    module_name = ".".join(function_full_name.split(".")[:-1])
    test_program_folder = os.path.join(
        ".", "simple_output", "test_program_templates", module_name.replace(".", os.sep)
    )
    mr_folder = os.path.join(
        ".", "simple_output", "metamorphic_relations", module_name.replace(".", os.sep)
    )
    os.makedirs(test_program_folder, exist_ok=True)

    # save the metamorphic relations to a JSON file
    mr_file_name = f"{function_name}_mrs.json"
    mr_file_path = os.path.join(mr_folder, mr_file_name)
    os.makedirs(os.path.dirname(mr_file_path), exist_ok=True)

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
        test_program_file_name = f"test_{mr_id}.py"
        test_program_file_path = os.path.join(
            test_program_folder, f"test_{function_name}", test_program_file_name
        )

        if os.path.exists(test_program_file_path):
            logger.info(
                f"Test program file {test_program_file_path} already exists. Skipping."
            )
            continue

        source_input_code = gen_source_input_for_function(mr, function_info)
        followup_input_code = gen_followup_input_for_function(mr, function_info)
        valid_code = gen_valid_code_for_function(mr, function_info)

        test_program = test_program_construction_for_function(
            function_info,
            mr,
            source_input_code,
            followup_input_code,
            valid_code,
        )

        logger.info(f"Writing test program to {test_program_file_path}")
        with open(test_program_file_path, "w", encoding="utf-8") as f:
            f.write(test_program)




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
        format="{time} {level} {file}|{line}: {message}",
    )

    api_file = 
    api_infos = json.load(open(api_file, "r", encoding="utf-8"))

    logger.info(f"Loaded {len(api_infos)} API infos from {api_file}")

    for api_info in api_infos:
        if api_info["type"] == "function":
            logger.info(f"Processing function: {api_info['name']}")
            MT_for_function(api_info)
        elif api_info["type"] == "class":
            logger.info(f"Processing class: {api_info['name']}")
            for method_info in api_info["methods"]:
                logger.info(
                    f"Processing method: {method_info['name']} in class {api_info['name']}"
                )
                # MT_for_class_method(method_info, api_info)
                pass
        else:
            raise ValueError(
                f"Unknown type {api_info['type']} for API info {api_info['name']}"
            )
