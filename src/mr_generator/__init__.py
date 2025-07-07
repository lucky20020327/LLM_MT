import os
import sys
import argparse

_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_pwd, ".."))
from mr_generator.base_generator import BaseGenerator
from mr_generator.llm_mr_generator import LLM_MR_Generator


def get_generator(args: argparse.Namespace) -> BaseGenerator:
    """
    Get the generator. Currently only LLM_MR_Generator is supported.
    """

    return LLM_MR_Generator(args=args)
