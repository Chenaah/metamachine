"""
Module for designing modular robots with balls and sticks.
This module provides the RobotDesigner class for creating and manipulating
modular robot designs using a combination of ball and stick components.

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
from collections import defaultdict
from typing import Dict, List, Union, Optional, Tuple, Any
import numpy as np
import mujoco
from lxml import etree

from .connect import ConnectAsym, generate_stick_docks, generate_stick_to_stick_child_connections
from .constants import MESH_DICT_FINE, ROBOT_CFG_AIR1S

from ...utils.designs import movability_check, self_collision_check, stable_state_check
from .robot_builder import RobotBuilder
from ...utils.math_utils import calculate_transformation_matrix, matrix_to_pos_quat, quaternion_from_vectors
from ...utils.visual_utils import is_headless, vec2string
from ...utils.rendering import view

from ... import METAMACHINE_ROOT_DIR
if not is_headless():
    import mujoco.viewer




class RobotDesigner:
    """
    A class for designing modular robots with balls and sticks.

    This class provides functionality to create and manipulate robot designs by
    connecting ball and stick modules in various configurations. It manages the
    available connection ports and maintains the robot's structure.

    Attributes:
        robot_cfg (Dict): Robot configuration parameters
        mesh_dict (Dict): Dictionary mapping module types to mesh files
        allow_overlapping (bool): Whether to allow overlapping connections
        lite (bool): Whether to use lite version of components
        available_ports (defaultdict): Available connection ports for each module
        parent_id_to_type (Dict[int, str]): Mapping of parent IDs to module types
        node_ids (List[int]): List of node IDs in the robot
        robot_properties (Dict): Properties of the robot design
        builder (RobotBuilder): Builder instance for constructing the robot
        connecting (ConnectAsym): Connection manager instance

    Module Types:
        - Ball: Can be connected at the top (0) or bottom (1)
        - Stick: Single type (2)

    Connection Types:
        - "00": Ball to Ball (top)
        - "02": Ball to Stick
        - "10": Ball to Ball (bottom)
        - "12": Ball to Stick
        - "20": Stick to Ball
        - "22": Stick to Stick
    """

    def __init__(
        self, 
        init_pipeline: Optional[List[int]] = None,
        robot_cfg: Optional[Dict] = None,
        mesh_dict: Optional[Dict] = None,
        allow_overlapping: bool = False,
        sim_cfg: Optional[Dict] = None
    ):
        """
        Initialize the RobotDesigner.

        Args:
            init_pipeline: Initial sequence of module connections
            robot_cfg: Robot configuration parameters
            mesh_dict: Dictionary mapping module types to mesh files
            allow_overlapping: Whether to allow overlapping connections
        """
        self.robot_cfg = robot_cfg or ROBOT_CFG_AIR1S.copy()
        self.robot_cfg["l"] = (
            self.robot_cfg["l_"] - 
            (self.robot_cfg["R"] - np.sqrt(self.robot_cfg["R"]**2 - self.robot_cfg["r"]**2))
        )
        
        self.mesh_dict = mesh_dict
        self.allow_overlapping = allow_overlapping
        self.lite = True
        self.sim_cfg = sim_cfg

        if init_pipeline is not None:
            self.reset()
            for step in np.reshape(init_pipeline, (-1, 4)):
                self.step(step)

    def _get_quat(self, pos: np.ndarray) -> np.ndarray:
        """
        Calculate quaternion for given position vector.

        Args:
            pos: Position vector [x, y, z]

        Returns:
            Quaternion representing rotation from [0, 0, 1] to -pos
        """
        target_vec = -np.array(pos)
        return quaternion_from_vectors([0, 0, 1], target_vec)
    


    def _add_ports(self, parent_id: int) -> None:
        """
        Add available connection ports for a given parent module.

        Args:
            parent_id: ID of the parent module
        """
        parent_type = self.parent_id_to_type[parent_id]
        connection_types = [i for i in self.connection_types if i[0] == parent_type]

        for connection_type in connection_types:
            docks = ConnectAsym(connection_type, self.robot_cfg, self.lite)
            for dock in docks:
                position_a = dock.position_a
                position_b = dock.position_b
                T_A_B = calculate_transformation_matrix(
                    dock.pos_a, dock.rotate_a, 
                    dock.pos_b, dock.rotate_b
                )
                pos, quat = matrix_to_pos_quat(T_A_B)
                self.available_ports[parent_id][position_a][position_b][connection_type].append((pos, quat))


    def _add_a_ball(
        self,
        parent_id: Optional[int] = None,
        position_id_a: Optional[int] = None,
        position_id_b: Optional[int] = None,
        rotation_id: Optional[int] = None,
        passive: bool = False
    ) -> List[int]:
        """
        Add a ball module to the robot design.

        A ball module consists of two parts: upper and lower hemispheres.
        If no parent is specified, creates a new standalone ball.
        Otherwise, connects the ball to the specified parent module.

        Args:
            parent_id: ID of parent module to connect to
            position_id_a: Position ID on parent module
            position_id_b: Position ID on child module
            rotation_id: Rotation variant to use for connection
            passive: Whether the ball is passive (no actuation)

        Returns:
            List of node IDs for the created ball module [upper_id, lower_id]
        """
        if parent_id is None:
            # Initialize a new standalone ball
            node_ids = self.builder.add_simple_ball(color=self.color, range=None, passive=passive)
        else:
            assert rotation_id is not None, "Ball connection requires rotation ID"
            parent_type = self.parent_id_to_type[parent_id]
            connection_type = parent_type + "0"  # Child uses upper half to connect
            ports = self.available_ports[parent_id][position_id_a][position_id_b][connection_type]
            port = ports[rotation_id]
            node_ids = self.builder.add_simple_ball(
                parent_id, port[0], port[1], 
                range=None, color=self.color, 
                passive=passive
            )

            if not self.allow_overlapping:
                del self.available_ports[parent_id][position_id_a]

        # Set up node types and ports
        types = ["0", "1"]  # Upper and lower hemisphere
        for i, t in zip(node_ids, types):
            self.parent_id_to_type[i] = t
            self._add_ports(i)
            self.node_ids.append(i)

        if parent_id is not None and not self.allow_overlapping:
            del self.available_ports[node_ids[0]][position_id_b]

        return node_ids

    def _add_a_stick(
        self,
        parent_id: int,
        position_id_a: Optional[int] = None,
        position_id_b: Optional[int] = None,
        rotation_id: Optional[int] = None,
        broken: int = 0,
        pos_offset: float = 0
    ) -> List[int]:
        """
        Add a stick module to the robot design.

        Args:
            parent_id: ID of parent module to connect to
            position_id_a: Position ID on parent module
            position_id_b: Position ID on child module
            rotation_id: Rotation variant to use for connection
            broken: Whether the stick is broken (0=no, 1=yes)
            pos_offset: Offset from nominal connection position

        Returns:
            List containing the node ID of the created stick [stick_id]
        """
        parent_type = self.parent_id_to_type[parent_id]
        connection_type = parent_type + "2"  # 2 indicates stick module
        ports = self.available_ports[parent_id][position_id_a][position_id_b][connection_type]
        port = ports[rotation_id]

        node_id, xml_body = self.builder.add_independent_stick(
            parent_id,
            pos=port[0],
            quat=port[1],
            color=self.color,
            broken=broken,
            pos_offset=pos_offset
        )

        self.parent_id_to_type[node_id] = "2"
        self._add_ports(node_id)
        self._add_sites(xml_body) # New dock management for testing

        self.node_ids.append(node_id)

        if not self.allow_overlapping:
            del self.available_ports[parent_id][position_id_a]
            del self.available_ports[node_id][position_id_b]

        return [node_id]
    

    def _add_sites(self, xml_body: etree.Element) -> None:

        docks = generate_stick_docks(self.robot_cfg)
        generate_stick_to_stick_child_connections(self.robot_cfg)
        for dock in docks:
            site_pos = dock.position
            site_quat = dock.quaternion
            site = etree.SubElement(xml_body, "site", 
                                    pos=vec2string(site_pos), 
                                    quat=vec2string(site_quat),
                                    name=f"{xml_body.get('name')}_{dock.name}") # TODO: move to RobotBuilder

    
    def add_dummy_node(self, type: int) -> List[int]:
        """
        Add a dummy node to the robot design.

        Args:
            type: Type of dummy node (0 for ball, 1 for stick)

        Returns:
            List of created node IDs
        """
        n_nodes = 2 if type == 0 else 1
        for _ in range(n_nodes):
            self.builder.leg_nodes.append(None)
            self.node_ids.append(len(self.builder.leg_nodes)-1)
        return self.node_ids[-n_nodes:]
    
    def get_pos_id_list(self, module: int, parent_id: int) -> List[int]:
        """
        Get list of valid position IDs for connecting a module.

        Args:
            module: Type of module to connect (0=ball, 1=stick)
            parent_id: ID of parent module to connect to

        Returns:
            List of valid position IDs
        """
        parent_type = self.parent_id_to_type[parent_id]
        child_type = "0" if module == 0 else "2"
        connection_type = parent_type + child_type
        
        pos_list = []
        for position_id, value in self.available_ports[parent_id].items():
            if connection_type in value:
                pos_list.append(position_id)
        return pos_list
    
    def get_rotation_id_list(
        self,
        module: int,
        parent_id: int,
        position_id_a: int,
        position_id_b: int
    ) -> List[int]:
        """
        Get list of valid rotation IDs for a connection.

        Args:
            module: Type of module to connect (0=ball, 1=stick)
            parent_id: ID of parent module
            position_id_a: Position ID on parent module
            position_id_b: Position ID on child module

        Returns:
            List of valid rotation indices
        """
        parent_type = self.parent_id_to_type[parent_id]
        child_type = "0" if module == 0 else "2"
        connection_type = parent_type + child_type
        return list(range(len(self.available_ports[parent_id][position_id_a][position_id_b][connection_type])))
    

    def reset(self, init_a_ball: bool = True) -> None:
        """
        Reset the robot designer to initial state.

        Args:
            init_a_ball: Whether to initialize with a ball module
        """
        self.builder = RobotBuilder(mesh_dict=self.mesh_dict, robot_cfg=self.robot_cfg, sim_cfg=self.sim_cfg)
        self.connection_types = ["00", "02", "10", "12", "20", "22"]
        self.color = 3
        self.available_ports = defaultdict(lambda: defaultdict(lambda: defaultdict((lambda: defaultdict(list)))))
        self.parent_id_to_type = {}
        self.node_ids = []
        self.robot_properties = {}

        if init_a_ball:
            self._add_a_ball()


    def step(self, pipeline: List[int]) -> List[int]:
        """
        Execute one step of the design pipeline.

        Args:
            pipeline: List containing [module_type, parent_id, position_id, rotation_id]

        Returns:
            List of created node IDs

        Raises:
            ValueError: If invalid module type specified
        """
        module, parent, pos, rotation = pipeline
        if module == 0:
            return self._add_a_ball(parent, pos, rotation)
        elif module == 1:
            return self._add_a_stick(parent, pos, rotation)
        else:
            raise ValueError(f"Invalid module type: {module}")
        
    def step_sequence(self, pipelines: List[List[int]]) -> None:
        """
        Execute a sequence of design pipeline steps.

        Args:
            pipelines: List of pipeline steps, each containing
                      [module_type, parent_id, position_id, rotation_id]
        """
        for pipeline in np.reshape(pipelines, (-1, 4)):
            self.step(pipeline)
        
    def get_xml(self) -> str:
        """
        Get the XML string representation of the robot.

        Returns:
            MuJoCo XML string describing the robot
        """
        return self.builder.get_xml()

    def compile(
        self,
        render: bool = False,
        self_collision_test: bool = True,
        stable_state_test: bool = True,
        movability_test: bool = False,
        joint_level_optimization: bool = False,
        config_dict: Optional[Dict] = None
    ) -> None:
        """
        Compile the robot design and test its properties.

        This method creates a MuJoCo simulation of the robot and runs various
        tests to evaluate its properties like self-collision, stability, and
        movability.

        Args:
            render: Whether to render the simulation
            self_collision_test: Whether to check for self-collisions
            stable_state_test: Whether to check for stable states
            movability_test: Whether to test robot mobility
            joint_level_optimization: Whether to optimize joint parameters
            config_dict: Additional configuration parameters
        """
        xml = self.builder.get_xml(fix_file_path=True)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        
        viewer = None
        if render and not is_headless():
            viewer = mujoco.viewer.launch_passive(m, d)
            viewer.__enter__()

        try:
            self.robot_properties["num_joints"] = m.nu

            if self_collision_test:
                self_collision_check(m, d, self.robot_properties, viewer=viewer)

            if stable_state_test:
                stable_state_check(
                    m, d, self.robot_properties,
                    self.robot_cfg["theta"],
                    viewer=viewer
                )

            if movability_test:
                movability_check(
                    m, d, self.robot_properties,
                    viewer=viewer,
                    config_dict=config_dict,
                    stable_pos_list=self.robot_properties.get("stable_pos"),
                    stable_quat_list=self.robot_properties.get("stable_quat")
                )

            self._log_properties()

        finally:
            if viewer:
                viewer.__exit__()

    def _log_properties(self) -> None:
        """Log the robot's computed properties."""
        properties = {
            "ave_speed": "Average speed",
            "self_collision_rate": "Self collision",
            "stable_height": "Stable height"
        }
        for key, desc in properties.items():
            if key in self.robot_properties:
                print(f"{desc}: {self.robot_properties[key]}")

    def wear_socks(
        self,
        node_ids: List[int],
        color: Tuple[float, float, float] = (1, 1, 1),
        radius: float = 0.04,
        length: float = 0.08,
        stick_length: float = 0.235,
        thickness: float = 0.01
    ) -> None:
        """
        Add sock-like covers to specified nodes.

        Args:
            node_ids: List of node IDs to add socks to
            color: RGB color values for socks
            radius: Radius of sock
            length: Length of sock
            stick_length: Length of stick part
            thickness: Thickness of sock material
        """
        for node_id in node_ids:
            self.builder.add_sock(
                node_id,
                radius=radius,
                length=length,
                stick_length=stick_length,
                color=color,
                thickness=thickness
            )

    def change_color(self, colors: Dict[int, Union[int, List[float]]]) -> None:
        """
        Change colors of specified nodes.

        Args:
            colors: Dictionary mapping node IDs to color values
        """
        for node_id, color in colors.items():
            self.builder.change_color(node_id, color=color)

    def save(
        self,
        save_dir: str,
    ) -> Tuple[str, str]:
        """
        Save the robot design to files and optionally render it.

        Args:
            save_dir: Directory to save files in
            fix: Whether to fix the robot in place
            pos: Initial position [x, y, z]
            quat: Initial orientation quaternion [w, x, y, z]
            joint_pos: Initial joint positions
            render: Whether to render the robot

        Returns:
            xml_path for saved files
        """
        f = self.builder.save(save_dir)

        return f
    
    def render(self, fix=False, pos=(0, 0, 0.4), quat=[1, 0, 0, 0], joint_pos=None):
        view(self.get_xml(), fix, pos=pos, quat=quat, vis_contact=True, joint_pos=joint_pos)

    def get_string(self):
        return self.builder.get_xml()

    @property
    def leg_nodes(self) -> List[Any]:
        """Get the leg nodes of the robot."""
        return self.builder.leg_nodes
    

def test_general_usage():
    """Test general usage of the RobotDesigner class."""
    robot_designer = RobotDesigner()
    robot_designer.reset()
    
    # Test basic pipeline steps
    pipelines = [
        [1, 0, 2, 10],  # Add stick to root
        [1, 0, 1, 7],   # Add another stick
        [1, 2, 2, 2]    # Add third stick
    ]
    
    for pipeline in pipelines:
        module = pipeline[0]
        parent = pipeline[1]
        
        # Get available positions
        pos_list = robot_designer.get_pos_id_list(module, parent)
        if not pos_list:
            print("No valid positions available")
            break
            
        pos = pipeline[2]
        orientation_list = robot_designer.get_rotation_id_list(module, parent, pos)
        orientation = pipeline[3]
        
        # Execute step
        robot_designer.step(pipeline)
        print(f"Added module: type={module}, parent={parent}, pos={pos}, rot={orientation}")

    # Test compilation and saving
    robot_designer.compile(render=False)
    
    save_dir = os.path.join(METAMACHINE_ROOT_DIR, "assets", "robots", "test_general")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save with stable pose if available
    pos = robot_designer.robot_properties.get("stable_pos", [[0, 0, 0.2]])[0]
    quat = robot_designer.robot_properties.get("stable_quat", [[1, 0, 0, 0]])[0]
    robot_designer.save(save_dir, fix=False, pos=pos, quat=quat)

def test_debug():
    """Test specific robot configurations for debugging."""
    robot_designer = RobotDesigner(
        robot_cfg=ROBOT_CFG_AIR1S,
        mesh_dict=MESH_DICT_FINE,
        allow_overlapping=True
    )
    robot_designer.reset()

    # Add initial sticks
    robot_designer._add_a_stick(0, 0, 0, 0, 0)
    robot_designer._add_a_stick(1, 0, 0, 0, 0)

    # Add branching sticks
    for i in range(1, 7):
        for j in [0]:
            robot_designer._add_a_stick(3, 3, i, j, 0)

    # Save for visualization
    save_dir = os.path.join(METAMACHINE_ROOT_DIR, "assets", "robots", "debug")
    robot_designer.save(save_dir)
    robot_designer.render(fix=True, pos=(0, 0, 1), quat=[1, 0, 0, 0])

if __name__ == "__main__":
    test_debug()