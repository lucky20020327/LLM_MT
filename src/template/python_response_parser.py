import json
import os
import sys

from loguru import logger

_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_pwd, ".."))

from template.base_response_parser import BaseParser


class PythonParser(BaseParser):
    """
    PythonParser is a class that parses responses from LLMs specifically for Python.
    """

    def _parse_source_input_generator(self, response):
        return self._python_parse_function_response(
            response,
            identifier="```source_input_generator",
            target_function_signature="def source_input_generator(",
        )

    def _parse_followup_input_generator(self, response):
        return self._python_parse_function_response(
            response,
            identifier="```followup_input_generator",
            target_function_signature="def followup_input_generator(",
        )

    def _parse_validate_mr_result(self, response):
        return self._python_parse_function_response(
            response,
            identifier="```validate_MR_result",
            target_function_signature="def validate_MR_result(",
        )

    def _parse_mr(self, response):
        """
        Parse the metamorphic relations from the response string.
        The response is expected to be in JSON format.
        """
        return self._python_parse_mr_response(response)

    def _python_parse_mr_response(self, response: str):
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

    def _python_parse_function_response(
        self, response: str, identifier: str, target_function_signature: str
    ):
        """
        Parse the function code from the response string.
        The response is expected to contain a code block with the specified identifier.
        """
        if identifier not in response:
            identifier = "```python"
        # get all the blocks inside indentifier and ```
        all_possible_blocks = response.split(identifier)[1:]
        all_possible_blocks = [
            block.split("```")[0].strip()
            for block in all_possible_blocks
            if block.strip()
        ]
        try:
            for function_code in all_possible_blocks:
                # Check if the target function signature is in the function code
                if target_function_signature in function_code:
                    # If found, return the function code
                    logger.info(f"Function code found for {target_function_signature}")
                    return function_code
            else:
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
