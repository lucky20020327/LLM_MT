import os
import sys
import argparse
import json

_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_pwd, ".."))

from template.base_template_formatter import BaseFormatter


class PythonFormatter(BaseFormatter):
    """
    PythonFormatter is a class that formats various prompts and templates for Python language.
    It inherits from BaseFormatter and implements methods to format prompts and templates
    specific to Python programming tasks.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the PythonFormatter and load the templates.
        """
        super().__init__(args=args)
        self.load_template("python")

    def function_followup_input_generator_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the follow-up input generator prompt for a function based on the metamorphic relation and function information.
        """

        function_name = function_info["name"]
        function_signature = function_info["signature"]
        function_docstring = function_info["docstring"]

        module_name = function_info["package"]

        followup_input_constraints = mr["followup_input_constraints"]

        followup_input_generator_prompt = (
            self.function_followup_input_generator_prompt.format(
                function_name=function_name,
                module_name=module_name,
                function_signature=function_signature,
                function_docstring=function_docstring,
                input_metamorphic_relation=mr["mr_input_relation"],
                input_transformation_steps=mr["mr_input_transformation_steps"],
                input_constraints=followup_input_constraints,
            )
        )

        return followup_input_generator_prompt

    def class_method_followup_input_generator_prompt_formatter(self, mr, function_info):
        """
        Format the follow-up input generator prompt for a class method based on the metamorphic relation and function information.
        """
        function_name = function_info["name"]
        function_signature = function_info["signature"]
        function_docstring = function_info["docstring"]

        class_name = function_info["class_name"]
        class_signature = function_info["class_signature"]
        class_docstring = function_info["class_docstring"]

        module_name = function_info["package"]

        followup_input_constraints = mr["followup_input_constraints"]

        followup_input_generator_prompt = (
            self.class_method_followup_input_generator_prompt.format(
                function_name=function_name,
                module_name=module_name,
                function_signature=function_signature,
                function_docstring=function_docstring,
                class_name=class_name,
                class_signature=class_signature,
                class_docstring=class_docstring,
                input_metamorphic_relation=mr["mr_input_relation"],
                input_transformation_steps=mr["mr_input_transformation_steps"],
                input_constraints=followup_input_constraints,
            )
        )

        return followup_input_generator_prompt

    def function_source_input_generator_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        function_name = function_info["name"]
        function_signature = function_info["signature"]
        function_docstring = function_info["docstring"]

        module_name = function_info["package"]

        source_input_constraints = mr["source_input_constraints"]

        source_input_generator_prompt = (
            self.function_source_input_generator_prompt.format(
                function_name=function_name,
                module_name=module_name,
                function_signature=function_signature,
                function_docstring=function_docstring,
                input_constraints=source_input_constraints,
            )
        )

        return source_input_generator_prompt

    def class_method_source_input_generator_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        function_name = function_info["name"]
        function_signature = function_info["signature"]
        function_docstring = function_info["docstring"]

        class_name = function_info["class_name"]
        class_signature = function_info["class_signature"]
        class_docstring = function_info["class_docstring"]

        module_name = function_info["package"]

        source_input_constraints = mr["source_input_constraints"]

        source_input_generator_prompt = (
            self.class_method_source_input_generator_prompt.format(
                function_name=function_name,
                module_name=module_name,
                function_signature=function_signature,
                function_docstring=function_docstring,
                class_name=class_name,
                class_signature=class_signature,
                class_docstring=class_docstring,
                input_constraints=source_input_constraints,
            )
        )

        return source_input_generator_prompt

    def function_mr_prompt_formatter(
        self,
        function_info: dict,
        function_analysis_report: str,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        function_name = function_info["name"]
        function_signature = function_info["signature"]
        function_docstring = function_info["docstring"]

        module_name = function_info["package"]

        mr_prompt = self.function_mr_prompt.format(
            function_name=function_name,
            module_name=module_name,
            function_analysis=function_analysis_report,
            function_signature=function_signature,
            function_docstring=function_docstring,
        )

        return mr_prompt

    def class_method_mr_prompt_formatter(
        self,
        function_info: dict,
        function_analysis_report: str,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        function_name = function_info["name"]
        function_signature = function_info["signature"]
        function_docstring = function_info["docstring"]

        class_name = function_info["class_name"]
        class_signature = function_info["class_signature"]
        class_docstring = function_info["class_docstring"]

        module_name = function_info["package"]

        mr_prompt = self.class_method_mr_prompt.format(
            function_name=function_name,
            module_name=module_name,
            function_analysis=function_analysis_report,
            function_signature=function_signature,
            function_docstring=function_docstring,
            class_name=class_name,
            class_signature=class_signature,
            class_docstring=class_docstring,
        )

        return mr_prompt

    def function_valid_code_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        function_name = function_info["name"]
        function_signature = function_info["signature"]
        function_docstring = function_info["docstring"]

        module_name = function_info["package"]

        valid_code_prompt = self.function_valid_code_prompt.format(
            function_name=function_name,
            module_name=module_name,
            function_signature=function_signature,
            function_docstring=function_docstring,
            input_metamorphic_relation=mr["mr_input_relation"],
            input_transformation_steps=mr["mr_input_transformation_steps"],
            output_metamorphic_relation=mr["mr_output_relation"],
            output_validation_steps=mr["mr_output_validation_steps"],
        )

        return valid_code_prompt

    def class_method_valid_code_prompt_formatter(
        self,
        mr: dict,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        function_name = function_info["name"]
        function_signature = function_info["signature"]
        function_docstring = function_info["docstring"]

        class_name = function_info["class_name"]
        class_signature = function_info["class_signature"]
        class_docstring = function_info["class_docstring"]

        module_name = function_info["package"]

        valid_code_prompt = self.class_method_valid_code_prompt.format(
            function_name=function_name,
            module_name=module_name,
            function_signature=function_signature,
            function_docstring=function_docstring,
            class_name=class_name,
            class_signature=class_signature,
            class_docstring=class_docstring,
            input_metamorphic_relation=mr["mr_input_relation"],
            input_transformation_steps=mr["mr_input_transformation_steps"],
            output_metamorphic_relation=mr["mr_output_relation"],
            output_validation_steps=mr["mr_output_validation_steps"],
        )

        return valid_code_prompt

    def local_function_test_program_template_formatter(
        self,
        function_info: dict,
        mr: dict,
        source_input_generator: str,
        followup_input_generator: str,
        valid_code: str,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        mr_str = json.dumps(mr, indent=4)

        test_program = self.local_function_test_program_template.format(
            metamorphic_relation=mr_str,
            source_input_code=source_input_generator,
            followup_input_code=followup_input_generator,
            validate_result_code=valid_code,
            function_name=function_info["name"],
            input_count=self.args.test_count_per_mr,
        )

        return test_program

    def function_test_program_template_formatter(
        self,
        function_info: dict,
        mr: dict,
        source_input_generator: str,
        followup_input_generator: str,
        valid_code: str,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        mr_str = json.dumps(mr, indent=4)

        test_program = self.function_test_program_template.format(
            metamorphic_relation=mr_str,
            source_input_code=source_input_generator,
            followup_input_code=followup_input_generator,
            validate_result_code=valid_code,
            function_name=function_info["name"],
            module_name=function_info["package"],
            input_count=self.args.test_count_per_mr,
        )

        return test_program

    def class_method_test_program_template_formatter(
        self,
        function_info: dict,
        mr: dict,
        source_input_generator: str,
        followup_input_generator: str,
        valid_code: str,
    ) -> str:
        mr_str = json.dumps(mr, indent=4)

        test_program = self.class_method_test_program_template.format(
            metamorphic_relation=mr_str,
            source_input_code=source_input_generator,
            followup_input_code=followup_input_generator,
            validate_result_code=valid_code,
            function_name=function_info["name"],
            module_name=function_info["package"],
            class_name=function_info["class_name"],
            input_count=self.args.test_count_per_mr,
        )

        return test_program

    def function_deep_report_template_formatter(
        self,
        function_info: dict,
    ) -> str:
        """
        Format the template with the provided keyword arguments.
        """
        assert (
            "deep_report" in function_info
        ), "Function information must contain 'deep_report' key."

        func_deep_report = function_info["deep_report"]

        deep_report = self.function_deep_report_template.format(
            control_flow=json.dumps(func_deep_report["control_flow"], indent=2),
            parameter_relations=json.dumps(
                func_deep_report["parameter_relations"], indent=2
            ),
            state_mutations=json.dumps(func_deep_report["state_mutations"], indent=2),
            computational_properties=json.dumps(
                func_deep_report["computational_properties"], indent=2
            ),
            parameter_sensitivity=json.dumps(
                func_deep_report["parameter_sensitivity"], indent=2
            ),
            ast_summary="\n".join(func_deep_report["ast_summary"]),
        )

        return deep_report
