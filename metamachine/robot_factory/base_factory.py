"""
Abstract base class for robot factories.
This module provides the base interface that all robot factories should implement,
ensuring consistency across different robot types while allowing for specific
implementations.

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

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class RobotType(Enum):
    """Enumeration of supported robot types."""
    MODULAR_LEGS = "modular_legs"
    MINI_MODULAR_LEGS = "mini_modular_legs"
    HUMANOID = "humanoid"
    QUADRUPED = "quadruped"
    MANIPULATOR = "manipulator"
    CUSTOM = "custom"


@dataclass
class RobotSpec:
    """Specification for a robot design."""
    robot_type: RobotType
    name: str
    description: str
    default_config: Dict[str, Any]
    supported_morphologies: List[str]
    capabilities: List[str]


class BaseRobotFactory(ABC):
    """
    Abstract base class for robot factories.
    
    This class defines the common interface that all robot factories must implement.
    It provides standardized methods for robot creation, configuration, and output
    generation while allowing specialized implementations for different robot types.
    
    Attributes:
        robot_type (RobotType): The type of robot this factory creates
        name (str): Human-readable name of the factory
        description (str): Description of what robots this factory creates
        default_config (Dict): Default configuration parameters
        supported_morphologies (List[str]): List of supported morphology types
    """
    
    def __init__(
        self,
        robot_type: RobotType,
        name: str,
        description: str,
        default_config: Optional[Dict[str, Any]] = None,
        supported_morphologies: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the base robot factory.
        
        Args:
            robot_type: The type of robot this factory creates
            name: Human-readable name of the factory
            description: Description of what robots this factory creates
            default_config: Default configuration parameters
            supported_morphologies: List of supported morphology types
            **kwargs: Additional factory-specific parameters
        """
        self.robot_type = robot_type
        self.name = name
        self.description = description
        self.default_config = default_config or {}
        self.supported_morphologies = supported_morphologies or []
        
        # Initialize factory-specific attributes
        self._initialize_factory(**kwargs)
    
    @abstractmethod
    def _initialize_factory(self, **kwargs):
        """Initialize factory-specific attributes and components."""
        pass
    
    @abstractmethod
    def create_robot(
        self,
        morphology: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'BaseRobot':
        """
        Create a new robot instance.
        
        Args:
            morphology: Robot morphology specification
            config: Configuration parameters
            **kwargs: Additional creation parameters
            
        Returns:
            BaseRobot: A robot instance
        """
        pass
    
    @abstractmethod
    def get_available_configurations(self) -> List[Dict[str, Any]]:
        """
        Get list of available robot configurations.
        
        Returns:
            List of configuration dictionaries
        """
        pass
    
    @abstractmethod
    def validate_morphology(self, morphology: Any) -> bool:
        """
        Validate if a morphology is supported by this factory.
        
        Args:
            morphology: The morphology to validate
            
        Returns:
            bool: True if morphology is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_default_morphology(self) -> Any:
        """
        Get the default morphology for this robot type.
        
        Returns:
            Default morphology specification
        """
        pass
    
    def get_robot_spec(self) -> RobotSpec:
        """
        Get the specification for this robot factory.
        
        Returns:
            RobotSpec: Complete specification of this factory
        """
        return RobotSpec(
            robot_type=self.robot_type,
            name=self.name,
            description=self.description,
            default_config=self.default_config,
            supported_morphologies=self.supported_morphologies,
            capabilities=self.get_capabilities()
        )
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get the capabilities supported by this factory.
        
        Returns:
            List of capability strings
        """
        pass
    
    def merge_config(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration dictionaries with override priority.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        merged.update(override_config)
        return merged


class BaseRobot(ABC):
    """
    Abstract base class for robot instances.
    
    This class defines the common interface that all robot instances must implement,
    providing standardized methods for robot manipulation, simulation, and output.
    """
    
    def __init__(
        self,
        factory: BaseRobotFactory,
        morphology: Any,
        config: Dict[str, Any],
        **kwargs
    ):
        """
        Initialize the base robot instance.
        
        Args:
            factory: The factory that created this robot
            morphology: Robot morphology specification
            config: Configuration parameters
            **kwargs: Additional robot-specific parameters
        """
        self.factory = factory
        self.morphology = morphology
        self.config = config
        self._initialize_robot(**kwargs)
    
    @abstractmethod
    def _initialize_robot(self, **kwargs):
        """Initialize robot-specific attributes and components."""
        pass
    
    @abstractmethod
    def reset(self, **kwargs):
        """Reset the robot to its initial state."""
        pass
    
    @abstractmethod
    def add_module(self, module_spec: Any) -> bool:
        """
        Add a module to the robot.
        
        Args:
            module_spec: Module specification
            
        Returns:
            bool: True if module was added successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def remove_module(self, module_id: Any) -> bool:
        """
        Remove a module from the robot.
        
        Args:
            module_id: Module identifier
            
        Returns:
            bool: True if module was removed successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_available_connections(self) -> List[Any]:
        """
        Get available connection points for new modules.
        
        Returns:
            List of available connection specifications
        """
        pass
    
    @abstractmethod
    def save(self, save_dir: str, **kwargs) -> str:
        """
        Save the robot to files.
        
        Args:
            save_dir: Directory to save files
            **kwargs: Additional save parameters
            
        Returns:
            str: Path to the saved main file
        """
        pass
    
    @abstractmethod
    def get_xml_string(self) -> str:
        """
        Get the robot as an XML string.
        
        Returns:
            str: XML representation of the robot
        """
        pass
    
    @abstractmethod
    def render(self, **kwargs):
        """
        Render the robot for visualization.
        
        Args:
            **kwargs: Rendering parameters
        """
        pass
    
    @abstractmethod
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the robot configuration.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        pass
    
    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """
        Get robot properties and statistics.
        
        Returns:
            Dictionary of robot properties
        """
        pass
    
    def get_robot_type(self) -> RobotType:
        """Get the robot type."""
        return self.factory.robot_type
    
    def get_config(self) -> Dict[str, Any]:
        """Get the robot configuration."""
        return self.config.copy()
    
    def get_morphology(self) -> Any:
        """Get the robot morphology."""
        return self.morphology
