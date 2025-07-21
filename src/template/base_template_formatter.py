from abc import ABC, abstractmethod
import os
import sys

import argparse

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd, ".."))

from utils.utils import get_language_of_dataset


class BaseFormatter(ABC):
    """
    Base class for tmplate/prompt formatters of different languages.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the formatter and load the templates.
        """
        self.args = args

    @property
    def function_followup_input_generator_prompt(self) -> str:
        """
        The prompt template for generating follow-up inputs for functions.
        """
        if not hasattr(self, "_function_followup_input_generator_prompt"):
            self._function_followup_input_generator_prompt = open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    get_language_of_dataset(self.args),
                    "function_followup_input_generator.prompt",
                ),
                "r",
                encoding="utf-8",
            ).read()
        return self._function_followup_input_generator_prompt

    @property
    def class_method_followup_input_generator_prompt(self) -> str:
        """
        The prompt template for generating follow-up inputs for class methods.
        """
        if not hasattr(self, "_class_method_followup_input_generator_prompt"):
            self._class_method_followup_input_generator_prompt = open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    get_language_of_dataset(self.args),
                    "class_method_followup_input_generator.prompt",
                ),
                "r",
                encoding="utf-8",
            ).read()
        return self._class_method_followup_input_generator_prompt

    @property
    def function_source_input_generator_prompt(self) -> str:
        """
        The prompt template for generating source inputs for functions.
        """
        if not hasattr(self, "_function_source_input_generator_prompt"):
            self._function_source_input_generator_prompt = open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    get_language_of_dataset(self.args),
                    "function_source_input_generator.prompt",
                ),
                "r",
                encoding="utf-8",
            ).read()
        return self._function_source_input_generator_prompt

    @property
    def class_method_source_input_generator_prompt(self) -> str:
        """
        The prompt template for generating source inputs for class methods.
        """
        if not hasattr(self, "_class_method_source_input_generator_prompt"):
            self._class_method_source_input_generator_prompt = open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    get_language_of_dataset(self.args),
                    "class_method_source_input_generator.prompt",
                ),
                "r",
                encoding="utf-8",
            ).read()
        return self._class_method_source_input_generator_prompt

    @property
    def function_mr_prompt(self) -> str:
        """
        The prompt template for generating metamorphic relations for functions.
        """
        if not hasattr(self, "_function_mr_prompt"):
            self._function_mr_prompt = open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    get_language_of_dataset(self.args),
                    "function_mr.prompt",
                ),
                "r",
                encoding="utf-8",
            ).read()
        return self._function_mr_prompt

    @property
    def class_method_mr_prompt(self) -> str:
        """
        The prompt template for generating metamorphic relations for class methods.
        """
        if not hasattr(self, "_class_method_mr_prompt"):
            self._class_method_mr_prompt = open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    get_language_of_dataset(self.args),
                    "class_method_mr.prompt",
                ),
                "r",
                encoding="utf-8",
            ).read()
        return self._class_method_mr_prompt

    @property
    def function_valid_code_prompt(self) -> str:
        """
        The prompt template for validating code for functions.
        """
        if not hasattr(self, "_function_valid_code_prompt"):
            self._function_valid_code_prompt = open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    get_language_of_dataset(self.args),
                    "function_valid_code.prompt",
                ),
                "r",
                encoding="utf-8",
            ).read()
        return self._function_valid_code_prompt

    @property
    def class_method_valid_code_prompt(self) -> str:
        """
        The prompt template for validating code for class methods.
        """
        if not hasattr(self, "_class_method_valid_code_prompt"):
            self._class_method_valid_code_prompt = open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    get_language_of_dataset(self.args),
                    "class_method_valid_code.prompt",
                ),
                "r",
                encoding="utf-8",
            ).read()
        return self._class_method_valid_code_prompt

    @property
    def local_function_test_program_template(self) -> str:
        """
        The template for generating test programs for local functions.
        """
        if not hasattr(self, "_local_function_test_program_template"):
            self._local_function_test_program_template = open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    get_language_of_dataset(self.args),
                    "local_function_test_program.template",
                ),
                "r",
                encoding="utf-8",
            ).read()
        return self._local_function_test_program_template

    @property
    def function_test_program_template(self) -> str:
        """
        The template for generating test programs for functions.
        """
        if not hasattr(self, "_function_test_program_template"):
            self._function_test_program_template = open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    get_language_of_dataset(self.args),
                    "function_test_program.template",
                ),
                "r",
                encoding="utf-8",
            ).read()
        return self._function_test_program_template

    @property
    def class_method_test_program_template(self) -> str:
        """
        The template for generating test programs for class methods.
        """
        if not hasattr(self, "_class_method_test_program_template"):
            self._class_method_test_program_template = open(
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    get_language_of_dataset(self.args),
                    "class_method_test_program.template",
                ),
                "r",
                encoding="utf-8",
            ).read()
        return self._class_method_test_program_template

    # @property
    # def function_deep_report_template(self) -> str:
    #     """
    #     The template for generating deep reports for functions.
    #     """
    #     if not hasattr(self, "_function_deep_report_template"):
    #         self._function_deep_report_template = open(
    #             os.path.join(
    #                 os.path.dirname(os.path.abspath(__file__)),
    #                 get_language_of_dataset(self.args),
    #                 "function_deep_report.template",
    #             ),
    #             "r",
    #             encoding="utf-8",
    #         ).read()
    #     return self._function_deep_report_template

    def followup_input_generator_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        if (
            function_info["type"] == "local_function"
            or function_info["type"] == "function"
        ):
            return self.function_followup_input_generator_prompt_formatter(
                mr=mr, function_info=function_info
            )
        elif function_info["type"] == "class_method":
            return self.class_method_followup_input_generator_prompt_formatter(
                mr=mr, function_info=function_info
            )

    def function_followup_input_generator_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    def class_method_followup_input_generator_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    def source_input_generator_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        if (
            function_info["type"] == "local_function"
            or function_info["type"] == "function"
        ):
            return self.function_source_input_generator_prompt_formatter(
                mr=mr, function_info=function_info
            )
        elif function_info["type"] == "class_method":
            return self.class_method_source_input_generator_prompt_formatter(
                mr=mr, function_info=function_info
            )

    def function_source_input_generator_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    def class_method_source_input_generator_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    def mr_prompt_formatter(
        self,
        function_info: dict,
        function_analysis_report: str,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        if (
            function_info["type"] == "local_function"
            or function_info["type"] == "function"
        ):
            return self.function_mr_prompt_formatter(
                function_info=function_info,
                function_analysis_report=function_analysis_report,
            )
        elif function_info["type"] == "class_method":
            return self.class_method_mr_prompt_formatter(
                function_info=function_info,
                function_analysis_report=function_analysis_report,
            )

    def function_mr_prompt_formatter(
        self,
        function_info: dict,
        function_analysis_report: str,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    def class_method_mr_prompt_formatter(
        self,
        function_info: dict,
        function_analysis_report: str,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    def valid_code_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        if (
            function_info["type"] == "local_function"
            or function_info["type"] == "function"
        ):
            return self.function_valid_code_prompt_formatter(
                mr=mr, function_info=function_info
            )
        elif function_info["type"] == "class_method":
            return self.class_method_valid_code_prompt_formatter(
                mr=mr, function_info=function_info
            )

    def function_valid_code_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    def class_method_valid_code_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    def test_program_template_formatter(
        self,
        function_info: dict,
        mr: dict,
        source_input_generator: str,
        followup_input_generator: str,
        valid_code: str,
    ) -> str:
        """
        Format the test program template with the provided keyword arguments.
        """
        if function_info["type"] == "local_function":
            return self.local_function_test_program_template_formatter(
                function_info=function_info,
                mr=mr,
                source_input_generator=source_input_generator,
                followup_input_generator=followup_input_generator,
                valid_code=valid_code,
            )
        elif function_info["type"] == "function":
            return self.function_test_program_template_formatter(
                function_info=function_info,
                mr=mr,
                source_input_generator=source_input_generator,
                followup_input_generator=followup_input_generator,
                valid_code=valid_code,
            )
        elif function_info["type"] == "class_method":
            return self.class_method_test_program_template_formatter(
                function_info=function_info,
                mr=mr,
                source_input_generator=source_input_generator,
                followup_input_generator=followup_input_generator,
                valid_code=valid_code,
            )

    def local_function_test_program_template_formatter(
        self,
        function_info: dict,
        mr: dict,
        source_input_generator: str,
        followup_input_generator: str,
        valid_code: str,
    ) -> str:
        """
        Format the local function test program template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    def function_test_program_template_formatter(
        self,
        function_info: dict,
        mr: dict,
        source_input_generator: str,
        followup_input_generator: str,
        valid_code: str,
    ) -> str:
        """
        Format the function test program template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    def class_method_test_program_template_formatter(
        self,
        function_info: dict,
        mr: dict,
        source_input_generator: str,
        followup_input_generator: str,
        valid_code: str,
    ) -> str:
        """
        Format the class method test program template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    # @abstractmethod
    # def deep_report_template_formatter(self, **kwargs) -> str:
    #     """
    #     Format the template with the provided keyword arguments.
    #     """
    #     raise NotImplementedError("Subclasses must implement the format method.")
