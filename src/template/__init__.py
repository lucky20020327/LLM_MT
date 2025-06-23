import os
import sys

_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_pwd)

from base_response_parser import BaseParser
from python_response_parser import PythonParser
from R_response_parser import RParser

from base_template_formatter import BaseFormatter
from python_template_formatter import PythonFormatter
from R_template_formatter import RFormatter

parser_per_language = {
    "python": PythonParser(),
    "R": RParser(),
}

template_formatter_per_language = {
    "python": PythonFormatter(),
    "R": RFormatter(),
}


# get template formatter based on language
def get_prompt_template_formatter(dataset: str) -> BaseFormatter:
    """
    Get the prompt template formatter for the specified language.
    Returns a dictionary with keys as prompt types and values as the template formatters.
    """
    if dataset in ["humaneval", "bigcodebench"]:
        language = "python"
    elif dataset in ["rmcda"]:
        language = "R"
    else:
        raise ValueError(f"Dataset '{dataset}' is not supported.")

    if language not in template_formatter_per_language:
        raise ValueError(f"Language '{language}' is not supported.")

    return template_formatter_per_language[language]


# get response parser based on language
def get_parser(dataset: str) -> BaseParser:
    """
    Get the response parsers for the specified language.
    Returns a tuple with the metamorphic relations parser and the function response parser.
    """
    if dataset in ["humaneval", "bigcodebench"]:
        language = "python"
    elif dataset in ["rmcda"]:
        language = "R"
    else:
        raise ValueError(f"Dataset '{dataset}' is not supported.")

    if language not in parser_per_language:
        raise ValueError(f"Language '{language}' is not supported.")

    return parser_per_language[language]
