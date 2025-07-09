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

        # followup input generator prompts
        with open(
            os.path.join(_pwd, language, "function_followup_input_generator.prompt"),
            "r",
            encoding="utf-8",
        ) as f:
            self.function_followup_input_generator_prompt = f.read()
        with open(
            os.path.join(
                _pwd, language, "class_method_followup_input_generator.prompt"
            ),
            "r",
            encoding="utf-8",
        ) as f:
            self.class_method_followup_input_generator_prompt = f.read()

        # source input generator prompts
        with open(
            os.path.join(_pwd, language, "function_source_input_generator.prompt"),
            "r",
            encoding="utf-8",
        ) as f:
            self.function_source_input_generator_prompt = f.read()
        with open(
            os.path.join(_pwd, language, "class_method_source_input_generator.prompt"),
            "r",
            encoding="utf-8",
        ) as f:
            self.class_method_source_input_generator_prompt = f.read()

        # metamorphic relation prompts
        with open(
            os.path.join(_pwd, language, "function_mr.prompt"),
            "r",
            encoding="utf-8",
        ) as f:
            self.function_mr_prompt = f.read()
        with open(
            os.path.join(_pwd, language, "class_method_mr.prompt"),
            "r",
            encoding="utf-8",
        ) as f:
            self.class_method_mr_prompt = f.read()

        # valid code prompts
        with open(
            os.path.join(_pwd, language, "function_valid_code.prompt"),
            "r",
            encoding="utf-8",
        ) as f:
            self.function_valid_code_prompt = f.read()
        with open(
            os.path.join(_pwd, language, "class_method_valid_code.prompt"),
            "r",
            encoding="utf-8",
        ) as f:
            self.class_method_valid_code_prompt = f.read()
            
        # test program templates
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
            os.path.join(_pwd, language, "class_method_test_program.template"),
            "r",
            encoding="utf-8",
        ) as f:
            self.class_method_test_program_template = f.read()
            
        # # deep report templates
        # with open(
        #     os.path.join(_pwd, language, "function_deep_report.template"),
        #     "r",
        #     encoding="utf-8",
        # ) as f:
        #     self.function_deep_report_template = f.read()

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

    @abstractmethod
    def function_followup_input_generator_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    @abstractmethod
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

    @abstractmethod
    def function_source_input_generator_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    @abstractmethod
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

    @abstractmethod
    def function_mr_prompt_formatter(
        self,
        function_info: dict,
        function_analysis_report: str,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    @abstractmethod
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

    @abstractmethod
    def function_valid_code_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        raise NotImplementedError("Subclasses must implement the format method.")

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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
