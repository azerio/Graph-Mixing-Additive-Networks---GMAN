import ast
import pandas as pd
import math
from typing import Any

def convert_to_tuple(value: Any) -> Any:
    """
    Converts a string representation of a tuple into a tuple of mixed types.

    The string is expected to be a tuple-like format, e.g., "(1.0, 'a', 2.0, 3.0, 4.0)".
    'nan' values are converted to None.
    Elements at indices 0, 3, and 4 are converted to floats, while other elements are kept as strings.

    Args:
        value (Any): The input value. If it's a string, it will be processed. Otherwise, it will be returned as is.

    Returns:
        Any: The converted tuple or the original value.
    """
    if isinstance(value, str):
        # Replace 'nan' with 'None' to allow for safe parsing with literal_eval
        safe_value = value.replace("nan", "None")
        parsed = ast.literal_eval(safe_value)

        if isinstance(parsed, tuple):
            # Convert elements to their respective types
            float_values = [
                    math.nan if x is None else float(x) if i in {0, 3, 4} else str(x)
                    for i, x in enumerate(parsed)
                ]

            return tuple(float_values)

        return value
        return value
