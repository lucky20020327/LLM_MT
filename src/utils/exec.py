import subprocess

from loguru import logger


def execute_test_program(test_program_file_path: str, timeout: int = 5):
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
            return False, result.stderr.strip()
        logger.info(f"Test program {test_program_file_path} executed successfully.")
        return True, result.stdout.strip()

    except Exception as e:
        logger.warning(f"Error executing test program {test_program_file_path}: {e}")
        return False, str(e)
