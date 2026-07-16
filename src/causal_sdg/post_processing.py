"""
================================================================================
Author: Andrea Iommi
Code Ownership:
    - All Python source code in this file is written solely by the author.
Documentation Notice:
    - All docstrings and inline documentation are written by ChatGPT,
      but thoroughly checked and approved by the author for accuracy.
================================================================================
"""

import ast
from typing import List, Union, Any, Callable

import numpy as np

from .processing import ProcessingFunction


class YearExpProcessing(ProcessingFunction):
    """
    Processes experience years by converting ranges into a random integer within the specified range.

    Examples:
    ---------
    >>> processing = YearExpProcessing(
    ...     feature_name="years_experience",
    ...     apply_on=["years_experience_range"],
    ...     replace=True,
    ... )
    >>> value = processing.transform("2 - 5 years")
    >>> 2 <= value <= 5
    True
    >>> processing.transform("Not specified")
    0
    """

    @staticmethod
    def transform(x: str) -> int:

        if x == "Not specified":
            return 0
        elif x == "+15 years":
            start, end = 15, 30
        else:
            x = x.split()
            start, end = int(x[0]), int(x[2])

        return int(np.random.randint(start, end + 1, 1)[0])


class AgeProcessing(ProcessingFunction):
    """
    Processes age ranges by converting them into a random integer within the specified range.

    Examples:
    ---------
    >>> processing = AgeProcessing(
    ...     feature_name="age",
    ...     apply_on=["age_range"],
    ...     replace=True,
    ... )
    >>> value = processing.transform("25-34")
    >>> 25 <= value <= 34
    True
    >>> processing.transform("No answer")
    'No answer'
    """

    def transform(self, x: str) -> Union[str, int]:

        if x == "No answer":
            return "No answer"
        if x == "65+":
            start, end = 65, 75
        else:
            x = x.split("-")
            start, end = int(x[0]), int(x[1])

        return int(np.random.randint(start, end + 1, 1)[0])


class ReplacementProcessing(ProcessingFunction):
    """
    Replaces matching values with a fixed value or with the output of a callable.

    Parameters:
    -----------
    origin : Any
        Value to replace. If a list, tuple, set, or frozenset is provided, any
        matching value is replaced.

    replacement : Any or Callable
        Replacement value. If it is callable, it is evaluated each time a match
        is found. Callables with zero or one parameter are supported.

    Examples:
    ---------
    >>> processing = ReplacementProcessing(
    ...     feature_name="contract",
    ...     apply_on=["contract"],
    ...     replace=True,
    ...     origin="Not specified",
    ...     replacement="Unknown",
    ... )
    >>> processing.transform("Not specified")
    'Unknown'
    >>> processing.transform("Permanent")
    'Permanent'

    >>> random_age = ReplacementProcessing(
    ...     feature_name="age",
    ...     apply_on=["age"],
    ...     replace=True,
    ...     origin="No answer",
    ...     replacement=lambda: int(np.random.randint(30, 56)),
    ... )
    >>> value = random_age.transform("No answer")
    >>> 30 <= value <= 55
    True
    """

    def __init__(self, feature_name: str, apply_on: List[str], replace: bool,
                 origin: Any, replacement: Union[Any, Callable]):
        super().__init__(feature_name, apply_on, replace)
        self.origin = origin
        self.replacement = replacement

    def transform(self, x: Any) -> Any:
        if self._matches_origin(x):
            return self._get_replacement(x)

        return x

    def _matches_origin(self, x: Any) -> bool:
        if isinstance(self.origin, (list, tuple, set, frozenset)):
            return x in self.origin

        return x == self.origin

    def _get_replacement(self, x: Any) -> Any:
        if not callable(self.replacement):
            return self.replacement

        try:
            return self.replacement(x)
        except TypeError:
            return self.replacement()

class Str2ListProcessing(ProcessingFunction):
    """
    Converts a list-like value into an actual list.

    Strings containing a Python list literal are parsed with `ast.literal_eval`.
    Plain scalar strings are wrapped into a one-element list, which keeps this
    processing step compatible with mechanisms that return a single category.

    Examples:
    ---------
    >>> processing = Str2ListProcessing(
    ...     feature_name="skills",
    ...     apply_on=["raw_skills"],
    ...     replace=True,
    ... )
    >>> processing.transform("['Python', 'SQL']")
    ['Python', 'SQL']
    >>> processing.transform("Python")
    ['Python']

    >>> first_item = Str2ListProcessing(
    ...     feature_name="main_language",
    ...     apply_on=["languages"],
    ...     replace=True,
    ...     take_first=True,
    ... )
    >>> first_item.transform("['English', 'Italian']")
    'English'
    """

    def __init__(self, feature_name: str, apply_on: List[str], replace: bool,
                 take_first: bool = False):
        super().__init__(feature_name, apply_on, replace)

        self.take_first = take_first

    def transform(self, x: str) -> List[str]:

        if isinstance(x, list):
            to_list = x
        elif isinstance(x, tuple):
            to_list = list(x)
        elif isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
            except (ValueError, SyntaxError):
                parsed = x

            if isinstance(parsed, list):
                to_list = parsed
            elif isinstance(parsed, tuple):
                to_list = list(parsed)
            else:
                to_list = [parsed]
        else:
            raise ValueError("Input must be of type str, list or tuple")

        return (to_list[0] if len(to_list) > 0 else None) if self.take_first else to_list


class ProcessingCompose(ProcessingFunction):
    """
    Applies multiple processing functions sequentially to the same input value.

    All composed functions must share the same `feature_name`, `apply_on`, and
    `replace` settings.

    Examples:
    ---------
    >>> processing = ProcessingCompose([
    ...     Str2ListProcessing(
    ...         feature_name="main_skill",
    ...         apply_on=["skills"],
    ...         replace=True,
    ...         take_first=True,
    ...     )
    ... ])
    >>> processing.transform("['Python', 'SQL']")
    'Python'
    """

    def __init__(self, compose: List[ProcessingFunction]):
        """
        Initializes the ProcessingCompose with a list of ProcessingFunction objects.
        """
        feature_name = compose[0].feature_name
        apply_on = compose[0].apply_on
        replace = compose[0].replace

        if not (all(obj.feature_name == feature_name for obj in compose) and
                all(obj.apply_on == apply_on for obj in compose) and
                all(obj.replace == replace for obj in compose)):
            raise ValueError(
                "All functions in compose must have the same feature_name, apply_on, and replace attributes.")

        self.compose = compose
        super().__init__(feature_name, apply_on, replace)

    def transform(self, x: str) -> Any:

        for func in self.compose:
            x = func.transform(x)

        return x


class RandomProficiency(ProcessingFunction):
    """
    Assigns a random proficiency score to each element of a list.

    Scores are sampled from the discrete set `{0.25, 0.50, 0.75, 1}`.

    Examples:
    ---------
    >>> processing = RandomProficiency(
    ...     feature_name="skill_proficiency",
    ...     apply_on=["skills"],
    ...     replace=False,
    ... )
    >>> result = processing.transform(["Python", "SQL"])
    >>> set(result)
    {'Python', 'SQL'}
    >>> all(score in {0.25, 0.50, 0.75, 1} for score in result.values())
    True
    """

    def __init__(self, feature_name: str, apply_on: List[str], replace: bool):
        super().__init__(feature_name, apply_on, replace)

    def transform(self, x: List[str]) -> dict[str, float]:
        return {i: np.random.choice([0.25, 0.50, 0.75, 1]) for i in x}
