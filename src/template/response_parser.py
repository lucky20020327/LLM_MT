import json

from loguru import logger

# # #
# This file contains functions to parse responses from LLMs. 
# If you change the prompt, you may need to change the parsing logic accordingly.
# # #

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
