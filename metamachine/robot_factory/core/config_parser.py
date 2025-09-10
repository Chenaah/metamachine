"""
Configuration parser for metamachine modular robots.
Converts integer sequences to robot configurations.

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

from typing import List, Dict, Any


class ConfigParser:
    """
    Parses integer sequences representing robot configurations.
    
    The configuration format is a sequence of 4-tuples:
    [parent_module_id, parent_position_id, child_position_id, rotation_id, ...]
    """
    
    def __init__(self, robot_cfg: Dict[str, Any], mesh_dict: Dict[str, Any]):
        """
        Initialize the configuration parser.
        
        Args:
            robot_cfg: Robot configuration dictionary (e.g., ROBOT_CFG_AIR1S)
            mesh_dict: Mesh dictionary for robot parts (e.g., MESH_DICT_FINE)
        """
        self.robot_cfg = robot_cfg
        self.mesh_dict = mesh_dict
        
    def parse_config(self, config_sequence: List[int]) -> Dict[str, Any]:
        """
        Parse integer sequence into robot configuration.
        
        Args:
            config_sequence: List of integers representing robot configuration
            
        Returns:
            Dictionary containing parsed robot configuration
        """
        if len(config_sequence) % 4 != 0:
            raise ValueError("Configuration sequence must be divisible by 4")
            
        modules = []
        for i in range(0, len(config_sequence), 4):
            module_config = {
                'parent_module_id': config_sequence[i],
                'parent_position_id': config_sequence[i + 1],
                'child_position_id': config_sequence[i + 2],
                'rotation_id': config_sequence[i + 3]
            }
            modules.append(module_config)
            
        return {
            'modules': modules,
            'robot_cfg': self.robot_cfg,
            'mesh_dict': self.mesh_dict
        }
    
    def validate_config(self, parsed_config: Dict[str, Any]) -> bool:
        """
        Validate the parsed configuration for consistency.
        
        Args:
            parsed_config: Parsed robot configuration
            
        Returns:
            True if configuration is valid, False otherwise
        """
        # Add validation logic here
        # Check for valid module IDs, position IDs, etc.
        return True
    
    def get_config_info(self, parsed_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract information about the configuration.
        
        Args:
            parsed_config: Parsed robot configuration
            
        Returns:
            Dictionary with configuration statistics and properties
        """
        modules = parsed_config['modules']
        return {
            'num_modules': len(modules),
            'unique_parent_ids': len(set(m['parent_module_id'] for m in modules)),
            'config_complexity': len(modules) * 2,  # Simple complexity metric
        }