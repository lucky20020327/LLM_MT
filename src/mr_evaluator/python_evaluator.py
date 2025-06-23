import argparse
import json
import os
import sys
import tempfile
import subprocess
import inspect
from typing import Optional, Dict, Any

from loguru import logger

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pwd, ".."))
from mr_evaluator.base_evaluator import BaseEvaluator


class PythonEvaluator(BaseEvaluator):
    """
    PythonEvaluator is a class that evaluates metamorphic relations (MRs) for Python functions.
    It executes test programs generated from MRs and get the evaluation metrics for the MRs.
    """

    def evaluate(self, args: argparse.Namespace, function_info: dict):
        """
        Evaluate the metamorphic relations on a function and its mutants.
        This function should execute the test program and check if the metamorphic relations hold.
        """
        function_name = function_info["name"]
        module_name = function_info["package"]

        test_program_template_folder = os.path.join(
            args.output_dir,
            args.strategy,
            "test_program_templates",
            module_name.replace(".", os.sep),
            f"test_{function_name}",
        )

        mr_folder = os.path.join(
            args.output_dir,
            args.strategy,
            "metamorphic_relations",
            module_name.replace(".", os.sep),
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
            args.output_dir,
            args.strategy,
            "mr_evaluate_results",
            module_name.replace(".", os.sep),
            f"{function_name}_mr_evaluate_results.json",
        )
        os.makedirs(os.path.dirname(mr_evaluate_results_file_path), exist_ok=True)

        with open(mr_file_path, "r", encoding="utf-8") as f:
            MRs = json.load(f)

        for mr_id, mr in enumerate(MRs):

            mr_evaluate_results[mr_id] = {
                "mr": mr,
                "valid_mr": False,
                "mutant_detection_results": {},
                "error_message": "",
                "coverage_info": {},
            }

            test_program_template_file_name = f"test_{mr_id}.py.template"
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
                args.output_dir,
                args.strategy,
                "test_program_instances",
                module_name.replace(".", os.sep),
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
            original_test_program_file_name = f"original.py"
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

            execute_result, execute_output, coverage_info = self._execute_test_program(
                os.path.join(
                    test_program_instance_folder, original_test_program_file_name
                ),
                args.timeout,
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
                test_program_instance_file_name = f"{mutation_name}.py"
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
                execute_result, execute_output, _ = self._execute_test_program(
                    os.path.join(
                        test_program_instance_folder, test_program_instance_file_name
                    ),
                    args.timeout,
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

    def _run_coverage_on_script(
        self, script_path: str, timeout: int = 5
    ) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = os.path.join(tmpdir, ".coverage")
            json_report_path = os.path.join(tmpdir, "coverage.json")

            # Run coverage to collect data
            subprocess.run(
                [
                    "coverage",
                    "run",
                    f"--data-file={data_file}",
                    "--branch",
                    script_path,
                ],
                timeout=timeout,
            )

            # Write JSON report into json_report_path
            subprocess.run(
                [
                    "coverage",
                    "json",
                    f"--data-file={data_file}",
                    f"-o{json_report_path}",
                    "-q",
                ],
                check=True,
            )

            # Read and return JSON
            with open(json_report_path, "r") as f:
                return json.load(f)

    def _analyze_function_coverage(
        self, coverage_data: Dict[str, Any], module_path: str, function_name: str
    ) -> Dict[str, Any]:
        """
        Given coverage JSON and a function name, return statement and branch coverage stats.
        """
        logger.debug(
            f"Analyzing function '{function_name}' in module '{module_path}' for coverage."
        )
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        abs_module_path = os.path.abspath(module_path)

        # Normalize all paths from coverage report
        file_matches = {
            os.path.abspath(k): v for k, v in coverage_data["files"].items()
        }
        try:
            file_data = file_matches[abs_module_path]
        except KeyError:
            raise ValueError(
                f"File '{abs_module_path}' not found in coverage data. Available files: {list(file_matches.keys())}"
            )

        # Use inspect to find line numbers
        import importlib.util

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if not hasattr(mod, function_name):
            raise ValueError(
                f"Function '{function_name}' not found in module '{module_name}'."
            )

        func_obj = getattr(mod, function_name)
        source_lines, start_line = inspect.getsourcelines(func_obj)
        end_line = start_line + len(source_lines) - 1
        function_lines = set(range(start_line, end_line + 1))

        # Global stats
        all_executed = set(file_data.get("executed_lines", []))
        all_missing = set(file_data.get("missing_lines", []))

        all_executed_branches = file_data.get("executed_branches", [])
        all_missing_branches = file_data.get("missing_branches", [])
        all_branches = all_executed_branches + all_missing_branches

        # Function-specific stats
        func_executed = all_executed & function_lines
        func_missing = all_missing & function_lines
        func_statements = func_executed | func_missing

        func_executed_branches = 0
        func_missing_branches_details = []
        total_func_branches = 0

        for branch in all_branches:
            branch_start, branch_end = branch
            if branch_start in function_lines:
                total_func_branches += 1
                if branch in all_executed_branches:
                    func_executed_branches += 1
                else:
                    func_missing_branches_details.append((branch_start, branch_end))

        result = {
            "function": function_name,
            "file": abs_module_path,
            "lines": {
                "start": start_line,
                "end": end_line,
            },
            "statement_coverage": {
                "executed": len(func_executed),
                "missing": len(func_missing),
                "total": len(func_statements),
                "percent": (
                    (len(func_executed) / len(func_statements) * 100)
                    if func_statements
                    else 0.0
                ),
                "missing_details": sorted(func_missing),
            },
            "branch_coverage": {
                "executed": func_executed_branches,
                "missing": len(func_missing_branches_details),
                "total": total_func_branches,
                "percent": (
                    (func_executed_branches / total_func_branches * 100)
                    if total_func_branches
                    else 0.0
                ),
                "missing_details": func_missing_branches_details,
            },
        }

        return result

    def _execute_test_program(
        self,
        test_program_file_path: str,
        timeout: int = 5,
        with_coverage: bool = False,
        function_name: Optional[str] = None,
    ) -> tuple[bool, str, Dict[str, Any]]:
        """
        Execute the test program and return the result.
        This function should handle any exceptions that occur during execution.
        """
        try:
            # execute the test program using subprocess
            logger.info(f"Executing test program {test_program_file_path}")
            result = subprocess.run(
                ["python", test_program_file_path],
                capture_output=True,
                text=True,
                timeout=timeout,  # Set a timeout for the execution
            )
            if result.returncode != 0:
                logger.error(
                    f"Test program {test_program_file_path} failed with return code {result.returncode}"
                )
                return False, result.stderr.strip(), {}
            logger.info(f"Test program {test_program_file_path} executed successfully.")

        except Exception as e:
            logger.warning(
                f"Error executing test program {test_program_file_path}: {e}"
            )
            return False, str(e), {}

        try:
            if with_coverage:
                # If coverage is requested, we can return the coverage data
                try:
                    coverage_data = self._run_coverage_on_script(
                        test_program_file_path, timeout=timeout
                    )
                except Exception as e:
                    logger.warning(
                        f"Error running coverage on {test_program_file_path}: {e}"
                    )
                    return False, str(e), {}
                coverage_result = self._analyze_function_coverage(
                    coverage_data, test_program_file_path, function_name
                )
                return True, result.stdout.strip(), coverage_result
            else:
                return True, result.stdout.strip(), {}

        except Exception as e:
            logger.warning(
                f"Error analyzing coverage for {test_program_file_path}: {e}"
            )
            return False, str(e), {}
