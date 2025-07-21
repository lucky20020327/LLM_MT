import os
import sys
from typing import Optional, Dict, Any

from loguru import logger

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd, ".."))

from mr_evaluator.base_evaluator import BaseEvaluator


class REvaluator(BaseEvaluator):
    """
    REvaluator is a class that evaluates metamorphic relations (MRs) for R functions.
    It executes test programs generated from MRs and get the evaluation metrics for the MRs.
    """

    def get_decorated_function(self, function_info):
        if function_info["type"] == "local_function":
            return function_info["source_code"]

        function_name = function_info["name"]
        signature = function_info["signature"]
        if function_info["type"] == "function":
            return f"""my_{function_name} <- function({signature}) {{
    {function_name}({signature})
}}"""
        else:
            raise NotImplementedError("Error")

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

        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr

        # Import the base package for R
        base = importr("base")

        try:
            # Read the test program file
            with open(test_program_file_path, "r") as file:
                test_program_code = file.read()

            # Execute the test program code
            robjects.r(test_program_code)

            # If coverage is requested, collect coverage information
            if with_coverage:
                # Not implemented
                coverage_info = {}
            else:
                coverage_info = {}

            return True, "Test program executed successfully.", coverage_info

        except Exception as e:
            logger.error(f"Error executing test program: {e}")
            return False, str(e), {}
