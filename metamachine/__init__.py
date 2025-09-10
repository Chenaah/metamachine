"""
MetaMachine - A Modular Robotic Simulation Framework
MetaMachine is a simulation framework for modular robots. It provides flexible
environments and tools for reinforcement learning research, evolutionary robotics
research, and development of adaptive robotic systems with modular morphologies.
Key modules:
- environments: Core simulation environments and components
- robot_factory: Robot design and morphology generation tools
- utils: Utility functions and helper classes
Example usage:
from metamachine.environments.configs.config_registry import ConfigRegistry
from metamachine.environments.env_sim import MetaMachine
cfg = ConfigRegistry.create_from_name("basic_quadruped")
env = MetaMachine(cfg)

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

import os

# Package version
__version__ = "0.1.0"

# Root directory for asset and configuration file paths
METAMACHINE_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# Package metadata
__author__ = "Chen Yu"
__email__ = "chenyu@u.northwestern.edu"
__description__ = "A simulation framework for modular robots"