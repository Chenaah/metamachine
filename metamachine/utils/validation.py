"""Input validation utilities.

This module provides helper functions for validating and checking
the types of input arguments in a robust way.

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Sequence, Any

import numpy as np


def is_list_like(variable: Any) -> bool:
    """Check if variable is list-like (sequence or array) but not string.
    
    Args:
        variable: Variable to check
        
    Returns:
        True if variable is list-like, False otherwise
        
    Examples:
        >>> is_list_like([1, 2, 3])
        True
        >>> is_list_like(np.array([1, 2, 3]))
        True
        >>> is_list_like("hello")
        False
        >>> is_list_like(42)
        False
    """
    return isinstance(variable, (Sequence, np.ndarray)) and not isinstance(variable, (str, bytes))

def is_number(variable: Any) -> bool:
    """Check if variable can be converted to a number.
    
    Args:
        variable: Variable to check
        
    Returns:
        True if variable can be converted to float, False otherwise
        
    Examples:
        >>> is_number(42)
        True
        >>> is_number("3.14")
        True
        >>> is_number("hello")
        False
        >>> is_number([1, 2, 3])
        False
    """
    try:
        float(variable)
        return True
    except (ValueError, TypeError):
        return False