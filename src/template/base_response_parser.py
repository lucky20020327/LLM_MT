from abc import ABC, abstractmethod
import os
import sys

_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_pwd)


class BaseParser(ABC):
    """
    Base class for response parsers.
    
    This class defines the interface for parsing responses from LLMs.
    It provides a method to parse the response based on the target type, which can be:
    
    - source_input_generator: to parse the source input generator function.
    - followup_input_generator: to parse the followup input generator function.
    - validate_MR_result: to parse the validation of metamorphic relations result.
    - mr: to parse the metamorphic relations.

    The actual parsing logic for each target type should be correspondingly implemented to the concrete prompts.
    """

    def __init__(self):
        """
        Initialize the parser.
        """
        pass

    def parse_response(self, response: str, target: str) -> str:
        """
        Parse the response string to extract the relevant information.

        :param response: The response string from the LLM.
        :param target: The target information to extract from the response.
        :return: The parsed information as a string.
        """
        if target == "source_input_generator":
            return self._parse_source_input_generator(response)
        elif target == "followup_input_generator":
            return self._parse_followup_input_generator(response)
        elif target == "validate_MR_result":
            return self._parse_validate_mr_result(response)
        elif target == "mr":
            return self._parse_mr(response)
        else:
            raise ValueError(f"Unknown target: {target}")

    @abstractmethod
    def _parse_source_input_generator(self, response: str) -> str:
        """
        Parse the source input generator from the response string.

        :param response: The response string from the LLM.
        :return: The parsed source input generator as a string.
        """
        raise NotImplementedError(
            "Subclasses must implement the _parse_source_input_generator method."
        )

    @abstractmethod
    def _parse_followup_input_generator(self, response: str) -> str:
        """
        Parse the followup input generator from the response string.

        :param response: The response string from the LLM.
        :return: The parsed followup input generator as a string.
        """
        raise NotImplementedError(
            "Subclasses must implement the _parse_followup_input_generator method."
        )

    @abstractmethod
    def _parse_validate_mr_result(self, response: str) -> str:
        """
        Parse the validate MR result from the response string.

        :param response: The response string from the LLM.
        :return: The parsed validate MR result as a string.
        """
        raise NotImplementedError(
            "Subclasses must implement the _parse_validate_mr_result method."
        )

    @abstractmethod
    def _parse_mr(self, response: str) -> str:
        """
        Parse the function metamorphic relations from the response string.

        :param response: The response string from the LLM.
        :return: The parsed function metamorphic relations as a string.
        """
        raise NotImplementedError("Subclasses must implement the _parse_mr method.")
