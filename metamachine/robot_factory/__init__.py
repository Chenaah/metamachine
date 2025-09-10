"""
Metamachine Robot Factory Module
This module provides the core functionality for generating MuJoCo XML files
from modular robot configurations using a modern factory pattern.

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

# Import the new factory system
from .factory_registry import (
    get_robot_factory,
    register_factory,
    list_factories,
    get_factories_by_type,
    search_factories,
    get_registry
)

# Import base classes for custom factory development
from .base_factory import (
    BaseRobotFactory,
    BaseRobot,
    RobotType,
    RobotSpec
)

# Import specific factories
from .modular_legs.modular_legs_factory import ModularLegsFactory, ModularLegsRobot

# Legacy compatibility imports
from .modular_legs.meta_designer import ModularLegs

# Export public API
__all__ = [
    # Factory registry functions
    'get_robot_factory',
    'register_factory', 
    'list_factories',
    'get_factories_by_type',
    'search_factories',
    'get_registry',
    
    # Base classes
    'BaseRobotFactory',
    'BaseRobot',
    'RobotType',
    'RobotSpec',
    
    # Specific factories
    'ModularLegsFactory',
    'ModularLegsRobot',
    
    # Legacy compatibility
    'ModularLegs',
]


# Legacy compatibility function
def get_robot_factory_legacy(factory_name: str = "modular_legs"):
    """
    Legacy compatibility function for getting robot factories.
    
    This function maintains backward compatibility with the old factory system.
    New code should use the registry-based get_robot_factory function.
    """
    import warnings
    
    warnings.warn(
        "get_robot_factory_legacy is deprecated. Use get_robot_factory instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Map old factory names to new ones
    legacy_mapping = {
        "modular_legs": "modular_legs",
        "mini_modular_legs": "mini_modular_legs"
    }
    
    new_name = legacy_mapping.get(factory_name)
    if new_name is None:
        raise ValueError(f"Unknown factory name: {factory_name}")
    
    factory = get_robot_factory(new_name)
    
    # For regular modular_legs, return the legacy ModularLegs
    return ModularLegs