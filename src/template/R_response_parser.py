import json
import os
import sys

from loguru import logger

_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_pwd)

from base_response_parser import BaseParser


class RParser(BaseParser):
    """
    RParser is a class that parses responses from LLMs specifically for R.
    """

    def _parse_source_input_generator(self, response):
        return self._R_parse_function_response(
            response,
            identifier="```source_input_generator",
            target_function_signature="source_input_generator <- function(",
        )

    def _parse_followup_input_generator(self, response):
        return self._R_parse_function_response(
            response,
            identifier="```followup_input_generator",
            target_function_signature="followup_input_generator <- function(",
        )

    def _parse_validate_mr_result(self, response):
        return self._R_parse_function_response(
            response,
            identifier="```validate_MR_result",
            target_function_signature="validate_MR_result <- function(",
        )

    def _parse_mr(self, response):
        """
        Parse the metamorphic relations from the response string.
        The response is expected to be in JSON format.
        """
        return self._R_parse_mr_response(response)

    def _R_parse_mr_response(self, response: str):
        """
        Parse the metamorphic relations from the response string.
        The response is expected to be in JSON format.
        """
        identifier = "```mrs"
        if "```mrs" not in response:
            identifier = "```R"
        if identifier not in response:
            identifier = "```r"
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

    def _R_parse_function_response(
        self, response: str, identifier: str, target_function_signature: str
    ):
        """
        Parse the function code from the response string.
        The response is expected to contain a code block with the specified identifier.
        """
        if identifier not in response:
            identifier = "```R"
        if identifier not in response:
            identifier = "```r"
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
