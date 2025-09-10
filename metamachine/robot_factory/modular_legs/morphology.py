"""
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

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

class ModuleType(Enum):
    """Represents different types of modules in the robot."""
    BALL = 0
    STICK = 1

class DockPosition(Enum):
    """Represents different docking positions on modules."""
    
    
    # Stick positions (upper stick)
    UPPER_STICK_1 = 0
    UPPER_STICK_2 = 1
    UPPER_STICK_3 = 2
    UPPER_STICK_4 = 3
    UPPER_STICK_5 = 4
    UPPER_STICK_6 = 5
    UPPER_STICK_7 = 6

    # Ball positions
    BALL_TOP = 7
    BALL_BOTTOM = 8
    LOWER_BALL_TOP = 9
    LOWER_BALL_BOTTOM = 10
    
    # Stick positions (lower stick)
    LOWER_STICK_1 = 11
    LOWER_STICK_2 = 12
    LOWER_STICK_3 = 13
    LOWER_STICK_4 = 14
    LOWER_STICK_5 = 15
    LOWER_STICK_6 = 16
    LOWER_STICK_7 = 17

@dataclass
class ModuleConnection:
    """Represents a connection between two modules in the robot."""
    parent_module_id: int
    parent_dock: Union[DockPosition, int]
    child_dock: Union[DockPosition, int]
    orientation: int
    module_id: Optional[int] = -1
    
    @classmethod
    def from_sequence(cls, seq: List[int]) -> 'ModuleConnection':
        """Create a ModuleConnection from a sequence of integers."""
        if len(seq) != 4:
            raise ValueError("Connection sequence must have exactly 4 elements")
        return cls(
            parent_module_id=seq[0],
            parent_dock=seq[1],
            child_dock=seq[2],
            orientation=seq[3]
        )
    
    def to_sequence(self) -> List[int]:
        """Convert the connection to a sequence of integers."""
        return [
            self.parent_module_id,
            self.parent_dock if isinstance(self.parent_dock, int) else self.parent_dock.value,
            self.child_dock if isinstance(self.child_dock, int) else self.child_dock.value,
            self.orientation
        ]

class RobotMorphology:
    """Manages the physical structure/form of a modular robot."""
    
    def __init__(self, connections: Optional[List[ModuleConnection]] = None):
        """Initialize a robot morphology.
        
        Args:
            connections: List of module connections defining the robot structure
        """
        self.connections = connections or []
        self.module_count = 0
        # self._validate_connections()

    def _update_module_id(self):
        """Update module IDs based on the current connections."""
        for i, conn in enumerate(self.connections):
            conn.module_id = i+1
    
    # def _validate_connections(self):
    #     """Validate that the connections form a valid tree structure."""
    #     if not self.connections:
    #         return
        
    #     # Update module IDs
    #     self._update_module_id()
            
    #     # Update module count
    #     self.module_count = max(
    #         max(conn.parent_module_id for conn in self.connections) + 1,
    #         self.module_count
    #     )
        
    #     # Validate tree structure (no cycles, single parent)
    #     parent_counts = {}
    #     for conn in self.connections:
    #         child_id = conn.parent_module_id + 1  # Next module ID after parent
    #         if child_id in parent_counts:
    #             raise ValueError(f"Module {child_id} has multiple parents")
    #         parent_counts[child_id] = conn.parent_module_id
    
    def add_connection(self, connection: ModuleConnection):
        """Add a new connection to the morphology.
        
        Args:
            connection: The module connection to add
        """
        if not self.validate_connection(connection):
            raise ValueError("Invalid connection: " + str(connection))
        
        self.connections.append(connection)
        # self._validate_connections()

        self.module_count = len(self.connections) + 1

    
    @classmethod
    def from_sequence(cls, seq: List[int]) -> 'RobotMorphology':
        """Create a RobotMorphology from a sequence of integers.
        
        Args:
            seq: Sequence of integers [parent_id, parent_dock, child_dock, orientation, ...]
                The sequence length must be a multiple of 4, where each group of 4 integers
                represents a single connection in the robot's structure.
            
        Returns:
            RobotMorphology instance
        """

        if isinstance(seq[0], int):
            if not seq:
                return cls()
                
            if len(seq) % 4 != 0:
                raise ValueError(f"Sequence length must be a multiple of 4, got {len(seq)}")
                
            connections = []
            for i in range(0, len(seq), 4):
                connections.append(ModuleConnection.from_sequence(seq[i:i+4]))
        elif isinstance(seq[0], ModuleConnection):
            connections = seq
        else:
            raise ValueError(f"Invalid sequence type: {type(seq[0])}")
            
        return cls(connections)
    
    def to_sequence(self) -> List[int]:
        """Convert the morphology to a sequence of integers.
        
        Returns:
            List of integers representing the morphology
        """
        sequence = []
        for conn in self.connections:
            sequence.extend(conn.to_sequence())
        return sequence
    
    def get_available_docks(self, module_id: int) -> List[DockPosition]:
        """Get list of available docking positions for a module.
        
        Args:
            module_id: ID of the module
            
        Returns:
            List of available DockPosition values
        """
        self._update_module_id()


        used_as_parents = {
            conn.parent_dock if isinstance(conn.parent_dock, int) 
            else conn.parent_dock.value
            for conn in self.connections 
            if conn.parent_module_id == module_id
        }
        print(f"[Generating] used_as_parents: {used_as_parents}")

        used_as_children = {
            conn.child_dock if isinstance(conn.child_dock, int) 
            else conn.child_dock.value
            for conn in self.connections 
            if conn.module_id == module_id
        }
        print(f"[Generating] used_as_children: {used_as_children}")

        used_docks = used_as_parents.union(used_as_children)
        print(f"[Generating] Used docks for module {module_id}: {used_docks}")
        
        return [
            dock for dock in DockPosition 
            if dock.value not in used_docks
        ]
    
    def validate_connection(self, connection: ModuleConnection) -> bool:
        """Validate if a connection can be added to the morphology.
        
        Args:
            connection: The connection to validate
            
        Returns:
            bool: Whether the connection is valid
        """
        self._update_module_id()

        

        # Check parent module exists
        parent_module_id = connection.parent_module_id if isinstance(connection.parent_module_id, int) else connection.parent_module_id.value
        if parent_module_id > self.module_count:
            pdb.set_trace()
            print(f"Parent module {parent_module_id} does not exist in the morphology")
            return False
            
        # Check parent dock is available
        dock_value = (connection.parent_dock if isinstance(connection.parent_dock, int) 
                     else connection.parent_dock.value)
        used_as_parents = {
            conn.parent_dock if isinstance(conn.parent_dock, int) 
            else conn.parent_dock.value
            for conn in self.connections 
            if conn.parent_module_id == connection.parent_module_id
        }

        used_as_children = {
            conn.child_dock if isinstance(conn.child_dock, int) 
            else conn.child_dock.value
            for conn in self.connections 
            if conn.module_id == connection.parent_module_id
        }

        used_docks = used_as_parents.union(used_as_children)
        # print(f"[Checking] Used docks: {used_docks}")

        if dock_value in used_docks:
            print(f"Dock {dock_value} already used in parent or child position for module {parent_module_id}")
            pdb.set_trace()
            return False
        
        # Check child dock is available
        child_dock_value = (connection.child_dock if isinstance(connection.child_dock, int) 
                            else connection.child_dock.value)
        if child_dock_value not in [i for i in range(9)]:
            # Ball docks are always available
            print(f"Child dock {child_dock_value} is not in the first 9 positions")
            return False
        
        # Check orientation is valid
        available_rotations = self._get_available_rotation_ids(
            dock_value, child_dock_value)
        orientation_value = (connection.orientation if isinstance(connection.orientation, int)
                             else connection.orientation.value)
        if orientation_value not in available_rotations:
            print(f"Orientation {orientation_value} is not valid for parent dock {dock_value} and child dock {child_dock_value}")
            return False

        return True
    
    def _get_available_rotation_ids(self, parent_dock, child_dock) -> List[int]:
        """Get available rotation IDs for given positions.
        
        Args:
            posa (int): Position A ID
            posb (int): Position B ID
            
        Returns:
            list: List of available rotation IDs
        """
        stick_side = [1,2,3,4,5,6,11,12,13,14,15,16]
        if parent_dock in stick_side and child_dock in stick_side:
            return [0,1]
        return [0,1,2]
            