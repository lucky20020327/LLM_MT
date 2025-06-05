import tempfile
import subprocess
import json
import os
import inspect
from typing import Optional, Dict, Any

from loguru import logger


def run_coverage_on_script(script_path: str, timeout: int = 5) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, ".coverage")
        json_report_path = os.path.join(tmpdir, "coverage.json")

        # Run coverage to collect data
        subprocess.run(
            ["coverage", "run", f"--data-file={data_file}", "--branch", script_path],
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


def analyze_function_coverage(
    coverage_data: Dict[str, Any], module_path: str, function_name: str
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
    file_matches = {os.path.abspath(k): v for k, v in coverage_data["files"].items()}
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


def execute_test_program(
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
        logger.warning(f"Error executing test program {test_program_file_path}: {e}")
        return False, str(e), {}

    try:
        if with_coverage:
            # If coverage is requested, we can return the coverage data
            try:
                coverage_data = run_coverage_on_script(
                    test_program_file_path, timeout=timeout
                )
            except Exception as e:
                logger.warning(
                    f"Error running coverage on {test_program_file_path}: {e}"
                )
                return False, str(e), {}
            coverage_result = analyze_function_coverage(
                coverage_data, test_program_file_path, function_name
            )
            return True, result.stdout.strip(), coverage_result
        else:
            return True, result.stdout.strip(), {}

    except Exception as e:
        logger.warning(f"Error analyzing coverage for {test_program_file_path}: {e}")
        return False, str(e), {}
