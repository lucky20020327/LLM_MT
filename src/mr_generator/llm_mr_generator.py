import os
import sys
import argparse

from loguru import logger

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd, ".."))

from mr_generator.base_generator import BaseGenerator
from utils.llm import call_LLM
from utils.extract_info import process_api
from template import (
    get_prompt_template_formatter,
    get_parser,
)


class LLM_MR_Generator(BaseGenerator):
    """
    LLM_MR_Generator is a class that generates metamorphic relations (MRs) using a language model.
    It uses the language model to generate MRs based on the provided function information.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the LLM_MR_Generator with command line arguments.
        :param args: Command line arguments.
        """
        super().__init__(args)

        # Load template_formatter and parsers
        self.template_formatter = get_prompt_template_formatter(args)
        self.response_parser = get_parser(args)

    def gen_MR(self, function_info: dict):
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

        ## TODO: currently only simple strategy is supported.
        # if self.args.strategy == "func_deep_report":
        #     process_api(function_info)
        #     function_analysis_report = (
        #         self.template_formatter.function_deep_report_template_formatter(
        #             function_info=function_info
        #         )
        #     )
        # elif self.args.strategy == "simple":
        if self.args.strategy == "simple":
            function_analysis_report = ""
        else:
            raise ValueError(
                f"Unknown strategy {self.args.strategy}. Please use 'simple' or 'func_deep_report'."
            )

        mr_prompt = self.template_formatter.mr_prompt_formatter(
            function_info=function_info,
            function_analysis_report=function_analysis_report,
        )

        logger.debug(f"MR generator prompt: {mr_prompt}")
        response = call_LLM(
            mr_prompt, baseLLM=self.args.baseLLM, api_key=self.args.api_key
        )
        logger.debug(f"Response from LLM: {response}")

        MRs = self.response_parser.parse_response(response, target="mr")

        logger.info(
            f"Generated {len(MRs)} metamorphic relations for function {function_info['name']}"
        )
        return MRs

    def gen_source_input_generator(self, mr: dict, function_info: dict):
        """
        Generate source input generator for a function based on the given metamorphic relation and function information.
        This function should return a generator function as a string that can be used to generate source input.
        """
        logger.info(
            f"Generating source input generator for function {function_info['name']} with metamorphic relation {mr['mr_input_relation']}"
        )

        source_input_generator_prompt = (
            self.template_formatter.source_input_generator_prompt_formatter(
                mr=mr, function_info=function_info
            )
        )

        logger.debug(f"Source input generator prompt: {source_input_generator_prompt}")
        response = call_LLM(
            source_input_generator_prompt,
            baseLLM=self.args.baseLLM,
            api_key=self.args.api_key,
        )
        logger.debug(f"Response from LLM: {response}")

        generator = self.response_parser.parse_response(
            response,
            target="source_input_generator",
        )
        logger.info(f"Generated source input generator: {generator}")
        return generator

    def gen_followup_input_generator(self, mr: dict, function_info: dict):
        """
        Generate follow-up input generator for a function based on the given metamorphic relation, function information, and source input.
        This function should return a generator function as a string that can be used to generate follow-up input.
        """
        logger.info(
            f"Generating follow-up input generator for function {function_info['name']} with metamorphic relation {mr['mr_input_relation']}, {mr['mr_output_relation']}"
        )
        followup_input_generator_prompt = (
            self.template_formatter.followup_input_generator_prompt_formatter(
                mr=mr, function_info=function_info
            )
        )

        logger.debug(
            f"Follow-up input generator prompt: {followup_input_generator_prompt}"
        )
        response = call_LLM(
            followup_input_generator_prompt,
            baseLLM=self.args.baseLLM,
            api_key=self.args.api_key,
        )
        logger.debug(f"Response from LLM: {response}")

        generator = self.response_parser.parse_response(
            response,
            target="followup_input_generator",
        )
        logger.info(f"Generated follow-up input generator: {generator}")

        return generator

    def gen_valid_code(self, mr: dict, function_info: dict):
        """
        Generate valid code for a function based on the given metamorphic relation and function information.
        The generated function returns True if the function comply with the metamorphic relation, otherwise returns False.
        This function should return the valid code as a string.
        """
        logger.info(
            f"Generating valid code for function {function_info['name']} with metamorphic relation {mr['mr_input_relation']}"
        )
        valid_code_prompt = (
            self.template_formatter.valid_code_prompt_formatter(
                mr=mr, function_info=function_info
            )
        )

        logger.debug(f"Valid code prompt: {valid_code_prompt}")
        response = call_LLM(
            valid_code_prompt, baseLLM=self.args.baseLLM, api_key=self.args.api_key
        )
        logger.debug(f"Response from LLM: {response}")

        valid_code = self.response_parser.parse_response(
            response,
            target="validate_MR_result",
        )

        logger.info(f"Generated valid code: {valid_code}")
        return valid_code
