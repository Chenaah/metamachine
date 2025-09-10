"""Type definitions and aliases for the metamachine library.

This module provides common type aliases used throughout the codebase
to improve readability and type safety.

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

from typing import List, Union

import numpy as np

# Type aliases for improved readability and type safety
Vector3 = Union[List[float], np.ndarray]  # 3D vector representation
Quaternion = Union[List[float], np.ndarray]  # Quaternion rotation representation [w,x,y,z] or [x,y,z,w]
Color = Union[List[float], np.ndarray]  # RGB(A) color representation with values in [0,1]