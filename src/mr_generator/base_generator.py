from abc import ABC, abstractmethod
import os
import sys
import argparse


class BaseGenerator(ABC):
    """
    Abstract base class for generating metamorphic relations (MRs) for datasets.

    You must implement the following methods in subclasses:
    - gen_MR: Generate metamorphic relations for a function.
    - gen_source_input_generator: Generate source input generator for a function.
    - gen_followup_input_generator: Generate follow-up input generator for a function.
    - gen_valid_code_for_function: Generate valid code for a function based on the metamorphic relation and function information.

    Each of these methods should return the appropriate data as specified in the docstrings.
    The class also requires an `args` parameter during initialization, which contains command line arguments.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the mr generator with command line arguments.
        :param args: Command line arguments.
        """
        self.args = args

    @abstractmethod
    def gen_MR(function_info: dict):
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
        raise NotImplementedError(
            "gen_MR must be implemented by subclasses."
        )

    @abstractmethod
    def gen_source_input_generator(mr: dict, function_info: dict):
        """
        Generate source input generator for a function based on the given metamorphic relation and function information.
        This function should return a generator function as a string that can be used to generate source input.
        """
        raise NotImplementedError(
            "gen_source_input_generator must be implemented by subclasses."
        )


    @abstractmethod
    def gen_followup_input_generator(mr: dict, function_info: dict):
        """
        Generate follow-up input generator for a function based on the given metamorphic relation, function information, and source input.
        This function should return a generator function as a string that can be used to generate follow-up input.
        """
        raise NotImplementedError(
            "gen_followup_input_generator must be implemented by subclasses."
        )

    @abstractmethod
    def gen_valid_code(mr: dict, function_info: dict):
        """
        Generate valid code for a function based on the given metamorphic relation and function information.
        The generated function returns True if the function comply with the metamorphic relation, otherwise returns False.
        This function should return the valid code as a string.
        """
        raise NotImplementedError("gen_valid_code must be implemented by subclasses.")
