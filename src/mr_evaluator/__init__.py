import os
import sys
import argparse


_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_pwd, ".."))

from mr_evaluator.base_evaluator import BaseEvaluator
from mr_evaluator.python_evaluator import PythonEvaluator
from mr_evaluator.R_evaluator import REvaluator

from utils.utils import PYTHON_DATASETS, R_DATASETS


def get_evaluator(args: argparse.Namespace) -> BaseEvaluator:
    """
    Get the evaluator for the specified language.
    Returns an instance of BaseEvaluator or its subclass.
    """
    if args.dataset in PYTHON_DATASETS:
        return PythonEvaluator(args)
    elif args.dataset in R_DATASETS:
        return REvaluator(args)
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")

