from abc import ABC, abstractmethod
import os
import sys
import argparse
import json
import re
from loguru import logger
from typing import Optional, Dict, Any

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd, ".."))

from utils.utils import get_suffix_of_dataset, get_sep_of_dataset


class BaseEvaluator(ABC):
    """
    Base class for evaluators.

    This class defines the interface for evaluating mrs.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Initialize the evaluator.
        """
        self.args = args

    def evaluate(self, function_info: dict):
        """
        Evaluate the metamorphic relations on a function and its mutants.
        """
        if function_info["type"] == "local_function":
            self.evaluate_local_function(function_info)
        elif function_info["type"] == "function":
            self.evaluate_function(function_info)
        else:
            logger.error(
                f"Function type {function_info['type']} is not supported for evaluation."
            )
            raise ValueError(
                f"Function type {function_info['type']} is not supported for evaluation."
            )

    def evaluate_local_function(self, function_info: dict):
        """
        Evaluate the metamorphic relations on a function and its mutants.
        This function should execute the test program and check if the metamorphic relations hold.
        """
        function_name = function_info["name"]
        module_name = function_info["package"]

        test_program_template_folder = os.path.join(
            self.args.output_dir,
            self.args.strategy,
            "test_program_templates",
            module_name.replace(get_sep_of_dataset(self.args), os.sep),
            f"test_{function_name}",
        )

        mr_folder = os.path.join(
            self.args.output_dir,
            self.args.strategy,
            "metamorphic_relations",
            module_name.replace(get_sep_of_dataset(self.args), os.sep),
        )

        mr_file_name = f"{function_name}_mrs.json"
        mr_file_path = os.path.join(mr_folder, mr_file_name)

        if not os.path.exists(mr_file_path):
            logger.error(f"Metamorphic relations file {mr_file_path} does not exist.")
            raise FileNotFoundError(
                f"Metamorphic relations file {mr_file_path} does not exist."
            )

        mr_evaluate_results = {}
        mr_evaluate_results_file_path = os.path.join(
            self.args.output_dir,
            self.args.strategy,
            "mr_evaluate_results",
            module_name.replace(get_sep_of_dataset(self.args), os.sep),
            f"{function_name}_mr_evaluate_results.json",
        )
        os.makedirs(os.path.dirname(mr_evaluate_results_file_path), exist_ok=True)

        with open(mr_file_path, "r", encoding="utf-8") as f:
            MRs = json.load(f)

        file_suffix = get_suffix_of_dataset(args=self.args)

        for mr_id, mr in enumerate(MRs):

            mr_evaluate_results[mr_id] = {
                "mr": mr,
                "valid_mr": False,
                "mutant_detection_results": {},
                "error_message": "",
                "coverage_info": {},
            }

            test_program_template_file_name = f"test_{mr_id}.{file_suffix}.template"
            test_program_template_file_path = os.path.join(
                test_program_template_folder, test_program_template_file_name
            )

            if not os.path.exists(test_program_template_file_path):
                logger.error(
                    f"Test program template file {test_program_template_file_path} does not exist."
                )
                raise FileNotFoundError(
                    f"Test program template file {test_program_template_file_path} does not exist."
                )

            test_program_instance_folder = os.path.join(
                self.args.output_dir,
                self.args.strategy,
                "test_program_instances",
                module_name.replace(get_sep_of_dataset(self.args), os.sep),
                f"test_{function_name}",
                f"mr_{mr_id}",
            )
            os.makedirs(test_program_instance_folder, exist_ok=True)

            test_program_template = open(
                test_program_template_file_path, "r", encoding="utf-8"
            ).read()

            # Execute the test program on original function and check if the metamorphic relations hold
            original_test_program_file_path = test_program_template.replace(
                "{function_source_code}",
                function_info["source_code"],
            )
            original_test_program_file_name = f"origina.{file_suffix}"
            with open(
                os.path.join(
                    test_program_instance_folder, original_test_program_file_name
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(original_test_program_file_path)
            logger.info(
                f"Writing original test program instance to {os.path.join(test_program_instance_folder, original_test_program_file_name)}"
            )

            execute_result, execute_output, coverage_info = self.execute_test_program(
                os.path.join(
                    test_program_instance_folder, original_test_program_file_name
                ),
                self.args.timeout,
                with_coverage=True,
                function_name=function_name,
            )
            if not execute_result:
                logger.error(f"Test program for original function failed.")
                mr_evaluate_results[mr_id]["error_message"] = execute_output
                continue
            mr_evaluate_results[mr_id]["valid_mr"] = True
            mr_evaluate_results[mr_id]["coverage_info"] = coverage_info
            # Now, we need to execute the test program on each mutant of the function
            if "mutations" not in function_info:
                logger.error(
                    f"Mutations field is missing in the function info for {function_name}."
                )
                raise ValueError(
                    f"Mutations field is missing in the function info for {function_name}."
                )
            for mutation in function_info["mutations"]:
                mutation_name = mutation["name"]
                mutation_source_code = mutation["source_code"]

                test_program_instance_file_path = test_program_template.replace(
                    "{function_source_code}",
                    mutation_source_code,
                )  # Replace the function source code with the mutation source code
                test_program_instance_file_name = f"{mutation_name}.{file_suffix}"
                with open(
                    os.path.join(
                        test_program_instance_folder, test_program_instance_file_name
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(test_program_instance_file_path)
                logger.info(
                    f"Writing test program instance for mutation {mutation_name} to {os.path.join(test_program_instance_folder, test_program_instance_file_name)}"
                )

                # Execute the test program on the mutant and check if the metamorphic relation holds
                execute_result, execute_output, _ = self.execute_test_program(
                    os.path.join(
                        test_program_instance_folder, test_program_instance_file_name
                    ),
                    self.args.timeout,
                )
                if not execute_result:
                    logger.error(f"Test program for mutation {mutation_name} passed.")
                    mr_evaluate_results[mr_id]["mutant_detection_results"][
                        mutation_name
                    ] = False
                else:
                    logger.info(
                        f"Test program for mutation {mutation_name} failed. It indicates that the metamorphic relation detects the mutation."
                    )
                    # If the test program passed, we can check if the metamorphic relation holds
                    mr_evaluate_results[mr_id]["mutant_detection_results"][
                        mutation_name
                    ] = True

        with open(mr_evaluate_results_file_path, "w", encoding="utf-8") as f:
            json.dump(mr_evaluate_results, f, indent=4)
        logger.info(
            f"Metamorphic relation evaluation results saved to {mr_evaluate_results_file_path}"
        )

    def get_decorated_function(self, function_info: dict) -> str:
        """
        Get the decorated function source code from the function info.
        """

        if self.args.dataset in [
            "humaneval",
            "bigcodebench",
            "skcriteria",
        ]:  # python datasets
            function_name = function_info["name"]
            signature = function_info["signature"]
            # using re to extract parameters from the signature
            # the signature is like "(param1[: type1], param2[: type2]) -> return_type"
            # the parameters are (param1, param2)
            match = re.search(r"\((.*?)\)", signature)
            if match:
                params_str = match.group(1)  # "param1: type1, param2: type2"

                # Split by comma, strip each part, and extract param name before colon
                params = [
                    re.split(r":\s*", param.strip())[0]
                    for param in params_str.split(",")
                    if param.strip()
                ]
                after_star_in_params = False
                params_postprocess = []
                for param in params:
                    if param == "*":
                        after_star_in_params = True
                    elif after_star_in_params:
                        params_postprocess.append(f"{param}={param}")
                    else:
                        params_postprocess.append(param)
                params = params_postprocess
            else:
                raise ValueError(f"Invalid signature format: {signature}")
            return f"""def my_{function_name}{signature}:
    return {function_name}{",".join(params)}"""

        elif self.args.dataset in ["rmcda"]:  # R datasets
            function_name = function_info["name"]
            signature = function_info["signature"]
            return f"""my_{function_name} <- function({signature}) {{
    {function_name}({signature})
}}"""
        else:
            logger.error(
                f"Dataset {self.args.dataset} is not supported for getting decorated function."
            )
            raise ValueError(
                f"Dataset {self.args.dataset} is not supported for getting decorated function."
            )

    def evaluate_function(self, function_info: dict):
        """
        Evaluate the metamorphic relations on a function and its mutants.
        This function should execute the test program and check if the metamorphic relations hold.
        """
        function_name = function_info["name"]
        module_name = function_info["package"]

        test_program_template_folder = os.path.join(
            self.args.output_dir,
            self.args.strategy,
            "test_program_templates",
            module_name.replace(get_sep_of_dataset(self.args), os.sep),
            f"test_{function_name}",
        )

        mr_folder = os.path.join(
            self.args.output_dir,
            self.args.strategy,
            "metamorphic_relations",
            module_name.replace(get_sep_of_dataset(self.args), os.sep),
        )

        mr_file_name = f"{function_name}_mrs.json"
        mr_file_path = os.path.join(mr_folder, mr_file_name)

        if not os.path.exists(mr_file_path):
            logger.error(f"Metamorphic relations file {mr_file_path} does not exist.")
            raise FileNotFoundError(
                f"Metamorphic relations file {mr_file_path} does not exist."
            )

        mr_evaluate_results = {}
        mr_evaluate_results_file_path = os.path.join(
            self.args.output_dir,
            self.args.strategy,
            "mr_evaluate_results",
            module_name.replace(get_sep_of_dataset(self.args), os.sep),
            f"{function_name}_mr_evaluate_results.json",
        )
        os.makedirs(os.path.dirname(mr_evaluate_results_file_path), exist_ok=True)

        with open(mr_file_path, "r", encoding="utf-8") as f:
            MRs = json.load(f)

        file_suffix = get_suffix_of_dataset(args=self.args)

        for mr_id, mr in enumerate(MRs):

            mr_evaluate_results[mr_id] = {
                "mr": mr,
                "valid_mr": False,
                "mutant_detection_results": {},
                "error_message": "",
                "coverage_info": {},
            }

            test_program_template_file_name = f"test_{mr_id}.{file_suffix}.template"
            test_program_template_file_path = os.path.join(
                test_program_template_folder, test_program_template_file_name
            )

            if not os.path.exists(test_program_template_file_path):
                logger.error(
                    f"Test program template file {test_program_template_file_path} does not exist."
                )
                raise FileNotFoundError(
                    f"Test program template file {test_program_template_file_path} does not exist."
                )

            test_program_instance_folder = os.path.join(
                self.args.output_dir,
                self.args.strategy,
                "test_program_instances",
                module_name.replace(get_sep_of_dataset(self.args), os.sep),
                f"test_{function_name}",
                f"mr_{mr_id}",
            )
            os.makedirs(test_program_instance_folder, exist_ok=True)

            test_program_template = open(
                test_program_template_file_path, "r", encoding="utf-8"
            ).read()

            # Execute the test program on original function and check if the metamorphic relations hold
            original_test_program_file_path = test_program_template.replace(
                "{function_source_code}",
                self.get_decorated_function(function_info),
            )
            original_test_program_file_name = f"origina.{file_suffix}"
            with open(
                os.path.join(
                    test_program_instance_folder, original_test_program_file_name
                ),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(original_test_program_file_path)
            logger.info(
                f"Writing original test program instance to {os.path.join(test_program_instance_folder, original_test_program_file_name)}"
            )

            execute_result, execute_output, coverage_info = self.execute_test_program(
                os.path.join(
                    test_program_instance_folder, original_test_program_file_name
                ),
                self.args.timeout,
                with_coverage=True,
                function_name=function_name,
            )
            if not execute_result:
                logger.error(f"Test program for original function failed.")
                mr_evaluate_results[mr_id]["error_message"] = execute_output
                continue
            mr_evaluate_results[mr_id]["valid_mr"] = True
            mr_evaluate_results[mr_id]["coverage_info"] = coverage_info
            # Now, we need to execute the test program on each mutant of the function
            if "mutations" not in function_info:
                logger.error(
                    f"Mutations field is missing in the function info for {function_name}."
                )
                raise ValueError(
                    f"Mutations field is missing in the function info for {function_name}."
                )
            for mutation in function_info["mutations"]:
                mutation_name = mutation["name"]
                mutation_source_code = mutation["source_code"]

                test_program_instance_file_path = test_program_template.replace(
                    "{function_source_code}",
                    mutation_source_code,
                )  # Replace the function source code with the mutation source code
                test_program_instance_file_name = f"{mutation_name}.{file_suffix}"
                with open(
                    os.path.join(
                        test_program_instance_folder, test_program_instance_file_name
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(test_program_instance_file_path)
                logger.info(
                    f"Writing test program instance for mutation {mutation_name} to {os.path.join(test_program_instance_folder, test_program_instance_file_name)}"
                )

                # Execute the test program on the mutant and check if the metamorphic relation holds
                execute_result, execute_output, _ = self.execute_test_program(
                    os.path.join(
                        test_program_instance_folder, test_program_instance_file_name
                    ),
                    self.args.timeout,
                )
                if not execute_result:
                    logger.error(f"Test program for mutation {mutation_name} passed.")
                    mr_evaluate_results[mr_id]["mutant_detection_results"][
                        mutation_name
                    ] = False
                else:
                    logger.info(
                        f"Test program for mutation {mutation_name} failed. It indicates that the metamorphic relation detects the mutation."
                    )
                    # If the test program passed, we can check if the metamorphic relation holds
                    mr_evaluate_results[mr_id]["mutant_detection_results"][
                        mutation_name
                    ] = True

        with open(mr_evaluate_results_file_path, "w", encoding="utf-8") as f:
            json.dump(mr_evaluate_results, f, indent=4)
        logger.info(
            f"Metamorphic relation evaluation results saved to {mr_evaluate_results_file_path}"
        )

    @abstractmethod
    def execute_test_program(
        self,
        test_program_file_path: str,
        timeout: int = 5,
        with_coverage: bool = False,
        function_name: Optional[str] = None,
    ) -> tuple[bool, str, Dict[str, Any]]:
        """
        Execute the test program and return the result.
        This function should handle any exceptions that occur during execution.

        Parameters:
            test_program_file_path: The path to the test program file to be executed.
            timeout: The maximum time to wait for the execution to complete.
            with_coverage: A boolean indicating whether to collect coverage information.
            function_name: The name of the function being tested, if applicable.

        Returns:
            A tuple containing:
            - A boolean indicating if the execution was successful.
            - A string with the output or error message from the execution.
            - A dictionary with coverage information if requested.
        """
        raise NotImplementedError(
            "Subclasses must implement the execute_test_program method."
        )
