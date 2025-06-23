from abc import ABC, abstractmethod
import os
import sys

_pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_pwd)

class BaseEvaluator(ABC):
    """
    Base class for evaluators.
    
    This class defines the interface for evaluating mrs.
    """
    
    def __init__(self):
        """
        Initialize the evaluator.
        """
        pass

    @abstractmethod
    def evaluate(self, function_info: dict) -> dict:
        """
        Evaluate the metamorphic relation (mr) for the given API information.
        :param function_info: A dictionary containing the API information.
        :return: A dictionary containing the evaluation results.
        """
        raise NotImplementedError(
            "Subclasses must implement the evaluate method."
        )