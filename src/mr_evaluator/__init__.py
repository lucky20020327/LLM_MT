import os
import sys

_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_pwd, ".."))
from mr_evaluator.base_evaluator import BaseEvaluator
from mr_evaluator.python_evaluator import PythonEvaluator
from mr_evaluator.R_evaluator import REvaluator

evaluator_per_language = {
    "python": PythonEvaluator(),
    "R": REvaluator(),
}


def get_evaluator(dataset: str) -> BaseEvaluator:
    """
    Get the evaluator for the specified language.
    Returns an instance of BaseEvaluator or its subclass.
    """
    if dataset in ["humaneval", "bigcodebench"]:
        language = "python"
    elif dataset in ["rmcda"]:
        language = "R"
    else:
        raise ValueError(f"Dataset '{dataset}' is not supported.")

    if language not in evaluator_per_language:
        raise ValueError(f"Language '{language}' is not supported.")

    return evaluator_per_language[language]
