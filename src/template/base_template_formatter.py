from abc import ABC, abstractmethod
import os
import sys

import argparse


class BaseFormatter(ABC):
    """
    Base class for tmplate/prompt formatters of different languages.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the formatter and load the templates.
        """
        self.args = args

    def load_template(self, language: str):
        _pwd = os.path.dirname(os.path.abspath(__file__))
        with open(
            os.path.join(_pwd, language, "function_followup_input_generator.prompt"),
            "r",
            encoding="utf-8",
        ) as f:
            self.function_followup_input_generator_prompt = f.read()
        with open(
            os.path.join(_pwd, language, "function_source_input_generator.prompt"),
            "r",
            encoding="utf-8",
        ) as f:
            self.function_source_input_generator_prompt = f.read()
        with open(
            os.path.join(_pwd, language, "function_mr.prompt"),
            "r",
            encoding="utf-8",
        ) as f:
            self.function_mr_prompt = f.read()
        with open(
            os.path.join(_pwd, language, "function_valid_code.prompt"),
            "r",
            encoding="utf-8",
        ) as f:
            self.function_valid_code_prompt = f.read()
        with open(
            os.path.join(_pwd, language, "local_function_test_program.template"),
            "r",
            encoding="utf-8",
        ) as f:
            self.local_function_test_program_template = f.read()
        with open(
            os.path.join(_pwd, language, "function_test_program.template"),
            "r",
            encoding="utf-8",
        ) as f:
            self.function_test_program_template = f.read()
        with open(
            os.path.join(_pwd, language, "function_deep_report.template"),
            "r",
            encoding="utf-8",
        ) as f:
            self.function_deep_report_template = f.read()

    @abstractmethod
    def function_followup_input_generator_prompt_formatter(self, **kwargs) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    @abstractmethod
    def function_source_input_generator_prompt_formatter(self, **kwargs) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    @abstractmethod
    def function_mr_prompt_formatter(self, **kwargs) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    @abstractmethod
    def function_valid_code_prompt_formatter(self, **kwargs) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    @abstractmethod
    def local_function_test_program_template_formatter(self, **kwargs) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    @abstractmethod
    def function_test_program_template_formatter(self, **kwargs) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    @abstractmethod
    def function_deep_report_template_formatter(self, **kwargs) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")
