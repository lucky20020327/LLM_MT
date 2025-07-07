import os
import sys
import argparse

_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_pwd, ".."))

from template.base_response_parser import BaseParser
from template.python_response_parser import PythonParser
from template.R_response_parser import RParser

from template.base_template_formatter import BaseFormatter
from template.python_template_formatter import PythonFormatter
from template.R_template_formatter import RFormatter

from utils.utils import PYTHON_DATASETS, R_DATASETS

# get template formatter based on language
def get_prompt_template_formatter(args: argparse.Namespace) -> BaseFormatter:
    """
    Get the prompt template formatter for the specified language.
    Returns a dictionary with keys as prompt types and values as the template formatters.
    """
    if args.dataset in PYTHON_DATASETS:
        return PythonFormatter(args=args)
    elif args.dataset in R_DATASETS:
        return RFormatter(args=args)
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")



# get response parser based on language
def get_parser(args: argparse.Namespace) -> BaseParser:
    """
    Get the response parsers for the specified language.
    Returns a tuple with the metamorphic relations parser and the function response parser.
    """
    if args.dataset in PYTHON_DATASETS:
        return PythonParser(args=args)
    elif args.dataset in R_DATASETS:
        return RParser(args=args)
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")

