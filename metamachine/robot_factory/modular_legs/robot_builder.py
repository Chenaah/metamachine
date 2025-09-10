"""
MuJoCo XML Robot Builder
This module provides a flexible and extensible way to programmatically create MuJoCo XML robot models.
It supports building modular robots with various components like joints, actuators, and sensors.
Key Features:
- Modular robot construction with configurable components
- Support for various terrain types and environments
- Mesh import and validation
- Sensor and actuator management
- Type-safe interfaces with proper validation
- Extensible XML structure generation
Example:
    builder = RobotBuilder(terrain="flat")
    builder.add_module()  # Add first module
    builder.add_stairs(start_distance=3)  # Add stairs to environment
    builder.add_walls()  # Add boundary walls
    builder.save("robot_model")  # Save the model

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

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
from lxml import etree

from ..core.xml_builder import XMLBuilder

from ..core.terrain_builder import TerrainBuilder, TerrainType

from ... import METAMACHINE_ROOT_DIR
from ...utils.math_utils import (construct_quaternion, quaternion_from_vectors,
                                          quaternion_multiply_alt)
from ...utils.types import Color, Quaternion, Vector3
from ...utils.visual_utils import (create_color_spectrum, fix_model_file_path,
                                            lighten_color, vec2string)
from ...utils.validation import is_list_like
from ...utils.rendering import view



class RobotBuilderError(Exception):
    """Base exception class for RobotBuilder errors."""
    pass

class ConfigurationError(RobotBuilderError):
    """Raised when there's an error in the robot configuration parameters."""
    pass

class MeshError(RobotBuilderError):
    """Raised when there's an error with mesh files or mesh configuration."""
    pass




@dataclass
class RobotConfig:
    """
    Configuration parameters for the robot model.
    
    This class defines the physical and geometric parameters that determine
    the robot's structure and behavior.
    
    Attributes:
        theta (float): Angular position in radians (default: 0.610865)
        l (float): Length parameter in meters (default: 0.3)
        initial_pos (Vector3): Initial position vector [x, y, z] (default: [0, 0, 0])
        R (float): Major radius parameter in meters (default: 0.06)
        r (float): Minor radius parameter in meters (default: 0.03)
        d (float): Offset distance in meters (default: 0)
        stick_mass (float): Mass of connecting sticks in kg (default: 0.26)
        top_hemi_mass (float): Mass of top hemisphere in kg (default: 0.7)
        bottom_hemi_mass (float): Mass of bottom hemisphere in kg (default: 0.287)
        battery_mass (float): Mass of battery component in kg (default: 0)
        motor_mass (float): Mass of motor component in kg (default: 0.3)
        pcb_mass (float): Mass of PCB component in kg (default: 0)
        l_ (float): Additional length parameter (default: -1)
        delta_l (float): Length difference parameter (default: -1)
        stick_ball_l (float): Stick-to-ball length ratio (default: -1)
        a (float): General purpose parameter (default: -1)
    """
    theta: float = 0.610865
    l: float = 0.3
    initial_pos: Vector3 = field(default_factory=lambda: np.array([0, 0, 0]))
    R: float = 0.06
    r: float = 0.03
    d: float = 0
    stick_mass: float = 0.26
    top_hemi_mass: float = 0.7
    bottom_hemi_mass: float = 0.287
    battery_mass: float = 0
    motor_mass: float = 0.3
    pcb_mass: float = 0
    l_: float = -1
    delta_l: float = -1
    stick_ball_l: float = -1
    a: float = -1

    def __post_init__(self):
        """Convert lists to numpy arrays and validate configuration."""
        if isinstance(self.initial_pos, list):
            self.initial_pos = np.array(self.initial_pos)
        self._validate_config()

    def _validate_config(self):
        """
        Validate the configuration parameters.
        
        Raises:
            ConfigurationError: If any parameter is invalid
        """
        if not 0 <= self.theta <= np.pi:
            raise ConfigurationError(f"Theta must be between 0 and pi, got {self.theta}")
        if self.l <= 0:
            raise ConfigurationError(f"Length must be positive, got {self.l}")
        if self.R <= 0:
            raise ConfigurationError(f"Radius R must be positive, got {self.R}")
        if self.r <= 0:
            raise ConfigurationError(f"Radius r must be positive, got {self.r}")
        if any(m < 0 for m in [self.stick_mass, self.top_hemi_mass, self.bottom_hemi_mass, 
                              self.battery_mass, self.motor_mass, self.pcb_mass]):
            raise ConfigurationError("All masses must be non-negative")

@dataclass
class Port:
    """Represents a connection point on the robot."""
    pos: Vector3
    body_pos: Optional[Vector3] = None
    leg_node: Optional[etree.Element] = None
    leg_vec: Optional[Vector3] = None
    module_idx: int = 0

    def __post_init__(self):
        """Convert lists to numpy arrays."""
        if isinstance(self.pos, list):
            self.pos = np.array(self.pos)
        if isinstance(self.body_pos, list):
            self.body_pos = np.array(self.body_pos)
        if isinstance(self.leg_vec, list):
            self.leg_vec = np.array(self.leg_vec)

class MeshManager:
    """Manages mesh-related operations and validation."""
    
    SUPPORTED_MESH_TYPES = {".obj", ".stl"}
    SPECIAL_MESH_TYPES = {"SPHERE", "CYLINDER", "CAPSULE", "NONE", "VIRTUAL"}
    REQUIRED_MESHES = {"up", "bottom"}
    
    @classmethod
    def validate_mesh_dict(cls, mesh_dict: Optional[Dict[str, str]]) -> None:
        """Validate the mesh dictionary."""
        if mesh_dict is None:
            raise MeshError("mesh_dict must be provided")
            
        missing_keys = cls.REQUIRED_MESHES - set(mesh_dict.keys())
        if missing_keys:
            raise MeshError(f"Missing required mesh types: {', '.join(missing_keys)}")
            
        for key, path in mesh_dict.items():
            if not cls._is_valid_mesh_spec(path):
                raise MeshError(f"Invalid mesh specification for {key}: {path}")
                
    @classmethod
    def _is_valid_mesh_spec(cls, path: str) -> bool:
        """Check if a mesh specification is valid."""
        if path in cls.SPECIAL_MESH_TYPES or path.startswith("VIRTUAL_"):
            return True
        if not any(path.endswith(ext) for ext in cls.SUPPORTED_MESH_TYPES):
            return False
        if path.endswith(tuple(cls.SUPPORTED_MESH_TYPES)):
            mesh_path = os.path.join(METAMACHINE_ROOT_DIR, "assets", "parts", path)
            if not os.path.exists(mesh_path):
                raise MeshError(f"Mesh file not found: {mesh_path}")
        return True

class RobotBuilder:
    """
    A builder class for creating MuJoCo XML robot models programmatically.
    
    This class provides methods to construct complex robot models by adding various
    components like modules, joints, actuators, and sensors. It handles the XML
    structure creation and manipulation internally.

    The builder supports:
    - Modular robot construction with configurable joints and actuators
    - Various terrain types and environmental features
    - Mesh import and validation for robot components
    - Sensor and actuator configuration
    - Dynamic color and material management
    - Comprehensive error checking and validation

    Attributes:
        terrain (TerrainType): Type of terrain for the robot environment
        mesh_dict (Dict[str, str]): Dictionary mapping mesh names to their file paths
        robot_cfg (RobotConfig): Robot configuration parameters
        root (etree.Element): Root XML element of the robot model
        worldbody (etree.Element): World body XML element
        actuators (etree.Element): Actuators XML element
        contact (etree.Element): Contact XML element
        sensors (etree.Element): Sensors XML element
    """

    def __init__(
        self,
        mesh_dict: Optional[Dict[str, str]] = None,
        robot_cfg: Optional[Union[Dict, RobotConfig]] = None,
        terrain: Optional[Union[str, TerrainType]] = None,
        sim_cfg: Optional[Dict] = None
    ):
        """
        Initialize the RobotBuilder.

        Args:
            terrain: Type of terrain for the robot environment
            mesh_dict: Dictionary mapping mesh names to their file paths
            robot_cfg: Robot configuration parameters

        Raises:
            ConfigurationError: If the robot configuration is invalid
            MeshError: If the mesh dictionary is invalid or missing required meshes
            TerrainError: If the terrain type is invalid
        """
        # Validate mesh dictionary
        MeshManager.validate_mesh_dict(mesh_dict)
        self.mesh_dict = mesh_dict
        self.terrain = TerrainType(terrain) if isinstance(terrain, str) else terrain
        
        # Initialize robot configuration
        if isinstance(robot_cfg, dict):
            self.robot_cfg = RobotConfig(**robot_cfg)
        elif isinstance(robot_cfg, RobotConfig):
            self.robot_cfg = robot_cfg
        else:
            self.robot_cfg = RobotConfig()
        self.sim_cfg = sim_cfg if sim_cfg is not None else {"mj_dt": 0.025}

        # Create base XML structure
        try:
            self._create_xml_structure()
            self._init_robot_parameters()
            self._setup_initial_state()
        except Exception as e:
            raise RobotBuilderError(f"Failed to initialize robot: {str(e)}")

    def _create_xml_structure(self) -> None:
        """
        Create the basic XML structure for the robot model.
        
        Initializes the root element and all major sections of the XML document:
        - Compiler settings
        - Option settings
        - Size parameters
        - Visual settings
        - Default values
        - Asset definitions
        - World body
        - Actuators
        - Contacts
        - Sensors
        """
        # Create root element
        self.root = XMLBuilder.create_element(None, "mujoco", model="modular_legs")
        
        # Add main sections
        self._add_compiler_settings()
        self._add_option_settings()
        self._add_size_settings()
        self._add_visual_settings()
        self._add_default_settings()
        self._add_asset_section()
        
        # Create main body sections
        self.worldbody = XMLBuilder.create_element(self.root, "worldbody")
        self.actuators = XMLBuilder.create_element(self.root, "actuator")
        self.contact = XMLBuilder.create_element(self.root, "contact")
        self.sensors = XMLBuilder.create_element(self.root, "sensor")
        
        # Create tree object
        self.tree = etree.ElementTree(self.root)

    def _add_compiler_settings(self) -> None:
        """
        Add compiler settings to the XML structure.
        
        Configures basic compiler parameters:
        - angle: Units for angle measurements (degrees)
        - coordinate: Coordinate system type (local)
        - inertiafromgeom: Compute inertia from geometry (true)
        """
        XMLBuilder.create_element(
            self.root, 
            "compiler",
            angle="degree",
            coordinate="local",
            inertiafromgeom="true"
        )

    def _add_option_settings(self) -> None:
        """
        Add option settings to the XML structure.
        
        Configures simulation options:
        - integrator: Numerical integration method (RK4)
        - timestep: Simulation timestep (0.01)
        - flags: Enable contact, energy, and gravity
        """
        option = XMLBuilder.create_element(
            self.root,
            "option",
            integrator="RK4",
            timestep=f"{self.sim_cfg['mj_dt']}"
        )
        XMLBuilder.create_element(
            option,
            "flag",
            contact="enable",
            # energy="enable",
            gravity="enable"
        )

    def _add_size_settings(self) -> None:
        """
        Add size settings to the XML structure.
        
        Configures size-related parameters:
        - njmax: Maximum number of joints (500)
        - nconmax: Maximum number of contacts (100)
        """
        XMLBuilder.create_element(
            self.root,
            "size",
            njmax="500",
            nconmax="100"
        )

    def _add_visual_settings(self) -> None:
        """
        Add visual settings to the XML structure.
        
        Configures visualization parameters:
        - znear: Near clipping plane for visualization (0.001)
        """
        visual = XMLBuilder.create_element(self.root, "visual")
        XMLBuilder.create_element(visual, "map", znear="0.01", shadowclip="0.5")
        XMLBuilder.create_element(visual, "headlight", ambient="0.6 0.6 0.6", diffuse="0.3 0.3 0.3", specular="0 0 0")
        XMLBuilder.create_element(visual, "quality", shadowsize="26384")

    def _add_default_settings(self) -> None:
        """Add default settings to the XML structure."""
        default = XMLBuilder.create_element(self.root, "default")
        
        # Joint defaults
        XMLBuilder.create_element(
            default,
            "joint",
            armature="1",
            damping="1",
            limited="true"
        )
        
        # Geom defaults
        XMLBuilder.create_element(
            default,
            "geom",
            conaffinity="0",
            condim="3",
            density="5.0",
            friction="1 0.5 0.5",
            margin="0.01"
        )

    def _add_asset_section(self) -> None:
        """Add asset section with materials and textures."""
        self.assets = XMLBuilder.create_element(self.root, "asset")
        self._add_materials()
        self._add_textures()

    def _add_materials(self) -> None:
        """Add default materials to the asset section."""
        materials = [
            {
                "name": "matplane",
                "specular": "0",
                "shininess": "0.01",
                "reflectance": "0.1",
                "texture": "texplane",
                "texrepeat": "1 1",
                "texuniform": "true",
            },
            {
                "name": "hfield",
                "texture": "texplane",
                "texrepeat": "1 1",
                "texuniform": "true",
                "reflectance": "0.1"
            },
            {
                "name": "boundary",
                "texture": "boundary",
                "texrepeat": "1 1",
                "texuniform": "true",
                "reflectance": ".5",
                "rgba": "1 1 1 1"
            },
            {
                "name": "metallic",
                "specular": "1",
                "shininess": "0.8",
                "reflectance": "0.9",
                "emission": "0.1",
                "rgba": "0.2 0.2 0.2 1"
            }
        ]

        for material in materials:
            XMLBuilder.create_element(self.assets, "material", **material)


    def _add_textures(self) -> None:
        """Add default textures to the asset section."""
        textures = [
            {
                "type": "skybox",
                "builtin": "gradient",
                "rgb1": "0.8 0.8 0.8",
                "rgb2": "0 0 0",
                "width": "512",
                "height": "512"
            },
            {
                "name": "texplane",
                "type": "2d",
                "builtin": "checker",
                "rgb1": "0.6 0.6 0.6",
                "rgb2": "0.5 0.5 0.5",
                "width": "512",
                "height": "512",
                "mark": "cross",
                "markrgb": ".8 .8 .8",
            },
            {
                "name": "boundary",
                "type": "2d",
                "builtin": "flat",
                "rgb1": "0.6 0.6 0.7",
                "rgb2": "0.6 0.6 0.8",
                "width": "300",
                "height": "300",
            },
            {
                "name": "hfield",
                "type": "2d",
                "builtin": "checker",
                "rgb1": "0.4 0.4 0.4",
                "rgb2": "0.4 0.4 0.4",
                "width": "300",
                "height": "300",
                "mark": "edge",
                "markrgb": "0.2 0.2 0.2"
            }
        ]

        for texture in textures:
            XMLBuilder.create_element(self.assets, "texture", **texture)


    def _init_robot_parameters(self) -> None:
        """Initialize robot parameters from configuration."""
        # Create mass dictionary
        self.mass = {
            "stick": self.robot_cfg.stick_mass,
            "top_hemi": self.robot_cfg.top_hemi_mass,
            "bottom_hemi": self.robot_cfg.bottom_hemi_mass,
            "motor": self.robot_cfg.motor_mass,
            "pcb": self.robot_cfg.pcb_mass
        }

        # Define leg vectors
        self.lleg_vec = np.array([
            0,
            self.robot_cfg.l * np.cos(self.robot_cfg.theta),
            self.robot_cfg.l * np.sin(self.robot_cfg.theta)
        ])
        self.rleg_vec = -self.lleg_vec

        # Create color spectrum
        self.colors = create_color_spectrum(num_colors=6)

    def _setup_initial_state(self) -> None:
        """Setup initial state tracking variables."""
        self.n_joint = 0
        self.ports: List[Port] = []
        self.leg_nodes: List[etree.Element] = []
        self.idx_counter = 0
        self.passive_idx_counter = 0
        self.sock_idx_counter = 0
        self.obstacle_idx_counter = 0
        self.imported_mesh: List[str] = []
        self.torsos: List[etree.Element] = []

        # Add floor and import meshes
        self.terrain_builder = TerrainBuilder(self.terrain, self.worldbody)
        self.terrain_builder.build()

        self._add_light()
        self._import_mesh()

    def _add_light(self) -> None:
        self.light = XMLBuilder.create_element(
            self.worldbody,
            "light",
            mode="targetbodycom",
            target="torso0",
            pos="3 0 4",
            cutoff="100",
            diffuse="1 1 1",
            specular=".05 .05 .05"

        )


    

    def _import_mesh(self) -> None:
        """Import mesh files into the asset section."""
        for mesh_name, file in self.mesh_dict.items():
            if file.endswith(".obj") or file.endswith(".stl"):
                XMLBuilder.create_element(
                    self.assets,
                    "mesh",
                    file=self._get_mesh_file(file),
                    name=mesh_name,
                    scale="1 1 1"
                )
                self.imported_mesh.append(file)

    def _get_mesh_file(self, mesh_name: str) -> str:
        """Get the file path for a mesh."""
        file = os.path.join(METAMACHINE_ROOT_DIR, "assets", "parts", mesh_name)
        if not os.path.exists(file):
            raise ValueError(f"Mesh {mesh_name} not found in {file}")
        return file
    
    def _convert_quat(self, quat: Union[List[float], List[List[float]]]) -> np.ndarray:
        """
        Convert various quaternion formats to a numpy array.
        
        Args:
            quat: Quaternion in various formats
            
        Returns:
            np.ndarray: Converted quaternion
            
        Raises:
            ValueError: If quaternion format is invalid
        """
        if not is_list_like(quat):
            raise ValueError("Quaternion must be a list or array")
            
        if not all(is_list_like(i) for i in quat):
            if len(quat) == 4:
                return np.array(quat)
            elif len(quat) == 3:
                return quaternion_from_vectors(
                    [0, np.cos(self.robot_cfg.theta), np.sin(self.robot_cfg.theta)],
                    quat
                )
            else:
                raise ValueError("Invalid quaternion length")
        else:
            result = np.array([1, 0, 0, 0])
            for q in quat:
                if len(q) == 3:
                    q_step = quaternion_from_vectors(
                        [0, np.cos(self.robot_cfg.theta), np.sin(self.robot_cfg.theta)],
                        q
                    )
                elif len(q) == 4:
                    q_step = q
                else:
                    raise ValueError("Invalid quaternion component length")
                result = quaternion_multiply_alt(result, q_step)
            return result

    def _add_one_module(self, parent_id=None, pos=None, quat=None, range="-90 90", color=None, quat_r=[1,0,0,0], joint_axis=[0,0,1]):
        """
        Add a new module to the robot.
        
        Args:
            parent_id: ID of the parent node
            pos: Position vector [x, y, z]
            quat: Quaternion orientation [w, x, y, z]
            range: Joint range string "min max"
            color: RGB color tuple
            quat_r: Relative quaternion orientation
            joint_axis: Joint axis vector [x, y, z]
            
        Returns:
            List[int]: IDs of the created nodes
        """
        # Validate inputs
        self._validate_parent_id(parent_id) if parent_id is not None else None
        if pos is None and parent_id is not None:
            raise ValueError("Position must be specified when parent_id is provided")
        if quat is None and parent_id is not None:
            raise ValueError("Quaternion must be specified when parent_id is provided")

        # Handle default case for first module
        if parent_id is None:
            if self.idx_counter == 0:
                self._build_torsor()
                parent = self.torsos[-1]
                pos = self.robot_cfg.initial_pos
                quat = [1, 0, 0, 0]
            else:
                raise ValueError("parent_id is required after first module")
        else:
            parent = self.leg_nodes[parent_id]

        # Set default colors
        if color is None:
            color_l = self.colors[self.idx_counter]
            color_r = lighten_color(color_l, 0.5)
        elif isinstance(color, (int, float)):
            color_l = self.colors[color]
            color_r = lighten_color(color_l, 0.5)
        else:
            if len(np.array(color).shape) == 1:
                color_l = color
                color_r = lighten_color(color, 0.5)
            else:
                color_l = color[0]
                color_r = color[1]

        idx = self.idx_counter

        # Create left hemisphere
        joint_params = None  # Left hemisphere has no joint
        quat = self._convert_quat(quat)
        l0 = self._create_module_body(parent, f"l{idx}", pos, quat, color_l, joint_params)

        # Create right hemisphere with joint
        quat = self._convert_quat([quat_r, quat])
        joint_params = {
            "axis": vec2string(joint_axis),
            "name": f"joint{idx}",
            "pos": "0 0 0",
            "type": "hinge",
            "armature": "0.05",
            "damping": "0.2",
            "limited": "auto"
        }
        if range is not None:
            joint_params["range"] = range

        r0 = self._create_module_body(parent, f"r{idx}", pos, quat, color_r, joint_params)

        # Add geometries
        self._add_module_geometries(l0, r0, idx, color_l, color_r)
        
        # Add sensors
        self._add_module_sensors(l0, r0, idx)
        
        # Add contact exclusion
        XMLBuilder.create_element(self.contact, "exclude", body1=f"l{idx}", body2=f"r{idx}")

        # Add actuator if range specified
        self._add_actuator(f"joint{idx}")

        # Update lists and counters
        self.leg_nodes.extend([l0, r0])
        self.n_joint += 1
        self.idx_counter += 1

        return [len(self.leg_nodes)-2, len(self.leg_nodes)-1]

    def _add_module_geometries(self, l0: etree.Element, r0: etree.Element, idx: int, color_l: Color, color_r: Color) -> None:
        """Add geometries to a module's bodies."""
        # Add upper hemisphere
        if self.mesh_dict["up"].endswith((".obj", ".stl")):
            XMLBuilder.create_element(
                l0,
                "geom",
                type="mesh",
                name=f"left{idx}",
                mesh="up",
                rgba=f"{color_l[0]} {color_l[1]} {color_l[2]} 0.5",
                mass=str(self.robot_cfg.top_hemi_mass),
                material="metallic",
                friction="1.0 .0 .0",
                priority="2"
            )
        elif self.mesh_dict["up"] == "SPHERE":
            XMLBuilder.create_element(
                l0,
                "geom",
                type="sphere",
                name=f"left{idx}",
                size=str(self.robot_cfg.R),
                rgba=f"{color_l[0]} {color_l[1]} {color_l[2]} 1",
                mass=str(self.robot_cfg.top_hemi_mass),
                friction="1.0 .0 .0",
                priority="2"
            )
        else:
            raise ValueError("Upper hemisphere mesh must be .obj/.stl file or SPHERE")

        # Add lower hemisphere
        if self.mesh_dict["bottom"].endswith((".obj", ".stl")):
            XMLBuilder.create_element(
                r0,
                "geom",
                type="mesh",
                name=f"right{idx}",
                mesh="bottom",
                rgba=f"{color_r[0]} {color_r[1]} {color_r[2]} 0.5",
                mass=str(self.robot_cfg.bottom_hemi_mass),
                material="metallic",
                euler="180 0 60",
                friction="1.0 .0 .0",
                priority="2"
            )
        elif self.mesh_dict["bottom"] == "SPHERE":
            radius = self.robot_cfg.R
            XMLBuilder.create_element(
                r0,
                "geom",
                type="sphere",
                name=f"right{idx}",
                size=str(self.robot_cfg.R),
                rgba=f"{color_r[0]} {color_r[1]} {color_r[2]} 1",
                mass=str(self.robot_cfg.bottom_hemi_mass),
                friction="1.0 .0 .0",
                priority="2"
            )
        else:
            raise ValueError("Lower hemisphere mesh must be .obj/.stl file or SPHERE")

        # Add optional components
        self._add_optional_components(l0, idx, color_l)

    def _add_optional_components(self, body: etree.Element, idx: int, color: Color) -> None:
        """Add optional components (battery, PCB, motor) to a body."""
        components = {
            "battery": (self.robot_cfg.battery_mass, color),
            "pcb": (self.robot_cfg.pcb_mass, [0, 0, 0]),
            "motor": (self.robot_cfg.motor_mass, [1, 0, 0])
        }

        for comp_name, (mass, comp_color) in components.items():
            if comp_name not in self.mesh_dict:
                continue

            mesh_spec = self.mesh_dict[comp_name]
            if mesh_spec.endswith((".obj", ".stl")):
                XMLBuilder.create_element(
                    body,
                    "geom",
                    type="mesh",
                    name=f"{comp_name}{idx}",
                    mesh=comp_name,
                    rgba=f"{comp_color[0]} {comp_color[1]} {comp_color[2]} 0.5",
                    mass=str(mass),
                    material="metallic",
                    contype="10",
                    conaffinity="0"
                )
            elif mesh_spec == "CYLINDER" and comp_name == "motor":
                XMLBuilder.create_element(
                    body,
                    "geom",
                    type="cylinder",
                    name=f"{comp_name}{idx}",
                    pos=vec2string([0, 0, -0.015]),
                    quat=vec2string([1, 0, 0, 0]),
                    size=f"0.05 0.015",
                    rgba=f"{comp_color[0]} {comp_color[1]} {comp_color[2]} 1",
                    mass=str(mass),
                    contype="10",
                    conaffinity="0"
                )
            elif mesh_spec.startswith("VIRTUAL_") and comp_name == "motor":
                dist = float(mesh_spec.split("_")[1])
                XMLBuilder.create_element(
                    body,
                    "geom",
                    type="sphere",
                    name=f"{comp_name}{idx}",
                    size="0.01",
                    rgba=f"{comp_color[0]} {comp_color[1]} {comp_color[2]} 1",
                    mass=str(mass),
                    contype="10",
                    conaffinity="0",
                    pos=vec2string([0, 0, -dist])
                )
            elif mesh_spec != "NONE":
                raise ValueError(f"Invalid {comp_name} specification: {mesh_spec}")


    def _add_passive_ball(self, parent_id=None, pos=None, quat=None, range="-90 90", color=None, quat_r=[1,0,0,0], joint_axis=[0,0,-1]):
        """
        Add a passive ball module to the robot.
        
        This is a lower-level API for testing a realistic robot docking mechanism.
        If parent_id is None, a new torso is created.

        Args:
            parent_id: ID of the parent node
            pos: Position vector [x, y, z]
            quat: Quaternion orientation [w, x, y, z]
            range: Joint range string "min max"
            color: RGB color tuple
            quat_r: Relative quaternion orientation
            joint_axis: Joint axis vector [x, y, z]
            
        Returns:
            List[int]: IDs of the created nodes
        """
        # Handle parent and initial position/orientation
        if parent_id is None:
            if self.idx_counter == 0:
                self._build_torsor()
                parent = self.torsos[-1]
                pos = self.robot_cfg.initial_pos
                quat = [1, 0, 0, 0]
            else:
                raise ValueError("parent_id is required after first module")
        else:
            assert parent_id < len(self.leg_nodes), "The port_id is out of range"
            assert pos is not None, "The pos is required for the module"
            assert quat is not None, "The quat is required for the module"
            parent = self.leg_nodes[parent_id]

        # Set colors
        if color is None:
            color_l = self.colors[self.idx_counter]
            color_r = lighten_color(color_l, 0.5)
        elif isinstance(color, (int, float)):
            color_l = self.colors[color]
            color_r = lighten_color(color_l, 0.5)
        else:
            if len(np.array(color).shape) == 1:
                color_l = color
                color_r = lighten_color(color, 0.5)
            else:
                color_l = color[0]
                color_r = color[1]

        idx = self.idx_counter
        quat = self._convert_quat(quat)

        # Create left hemisphere
        joint_params = None
        l0 = self._create_module_body(parent, f"l{idx}", pos, quat, color_l, joint_params)

        # Create right hemisphere with joint
        quat = self._convert_quat([quat_r, quat])
        joint_params = {
            "axis": vec2string(joint_axis),
            "name": f"joint{idx}",
            "pos": "0 0 0",
            "type": "hinge",
            "armature": "0.05",
            "damping": "0.2",
            "limited": "auto"
        }
        if range is not None:
            joint_params["range"] = range

        r0 = self._create_module_body(parent, f"r{idx}", pos, quat, color_r, joint_params)

        # Add geometries based on mesh type
        if self.mesh_dict["up"].endswith((".obj", ".stl")):
            self._add_geom_to_body(l0, "mesh", f"left{idx}", color_l, self.robot_cfg.top_hemi_mass,
                                 mesh="up", material="metallic", friction="1.0 .0 .0", priority="2")
        elif self.mesh_dict["up"] == "SPHERE":
            self._add_geom_to_body(l0, "sphere", f"left{idx}", color_l, self.robot_cfg.top_hemi_mass,
                                 size=str(self.robot_cfg.R), friction="1.0 .0 .0", priority="2")
        else:
            raise ValueError("The mesh should be either a .obj file or a SPHERE")

        if self.mesh_dict["bottom"].endswith((".obj", ".stl")):
            self._add_geom_to_body(r0, "mesh", f"right{idx}", color_r, self.robot_cfg.bottom_hemi_mass,
                                 mesh="bottom", material="metallic", euler="180 0 60",
                                 friction="1.0 .0 .0", priority="2")
        elif self.mesh_dict["bottom"] == "SPHERE":
            self._add_geom_to_body(r0, "sphere", f"right{idx}", color_r, self.robot_cfg.bottom_hemi_mass,
                                 size=str(self.robot_cfg.R), friction="1.0 .0 .0", priority="2")
        else:
            raise ValueError("The mesh should be either a .obj file or a SPHERE")

        # Update lists and counters
        self.leg_nodes.extend([l0, r0])
        self.n_joint += 1
        self.idx_counter += 1

        return [len(self.leg_nodes)-2, len(self.leg_nodes)-1]

    def add_stick(self, parent_id, radius=0.03, length=0.25, pos=[0,0,0], quat=[1,0,0,0], color=None):
        assert parent_id < len(self.leg_nodes), "The port_id is out of range"
        quat = self._convert_quat(quat)
        if color is None:
            color = self.colors[3]
        elif isinstance(color, (int, float)):
            color = self.colors[color]
        if not isinstance(pos, str):
            pos = vec2string(pos)
        if not isinstance(quat, str):
            quat = vec2string(quat)
        parent = self.leg_nodes[parent_id]
        XMLBuilder.create_element(parent, "geom", type="cylinder", pos=pos, quat=quat, size=f"{radius} {length/2}", rgba=f"{color[0]} {color[1]} {color[2]} 1")


    def add_independent_stick(self, parent_id, pos=[0,0,0], quat=[1,0,0,0], color=None, broken=0., pos_offset=0):
        # stick in a new body element
        assert parent_id < len(self.leg_nodes), "The port_id is out of range"
        quat = self._convert_quat(quat)
        if color is None:
            color = self.colors[3]
        elif isinstance(color, (int, float)):
            color = self.colors[color]

        parent = self.leg_nodes[parent_id]
        # etree.SubElement(parent, "geom", type="cylinder", pos=vec2string(pos), quat=vec2string(quat), size=f"{radius} {length/2}", rgba=f"{color[0]} {color[1]} {color[2]} 1")

        idx = self.passive_idx_counter
        stick = XMLBuilder.create_element(parent, "body", name=f"passive{idx}", pos=vec2string(pos), quat=vec2string(quat))

        radius = self.robot_cfg.r
        length = self.robot_cfg.l_ 
        # assert type in ["cylinder", "capsule", "mesh"], "The type should be either 'cylinder', 'capsule' or 'mesh'"
        if self.mesh_dict["stick"].endswith(".obj") or self.mesh_dict["stick"].endswith(".stl"):
            if broken != 0:
                assert "cut_stick" in self.mesh_dict, "The cut_stick mesh should be provided"
            XMLBuilder.create_element(stick, "geom", name=f"stick{idx}", type="mesh", pos=vec2string(np.array([0,0,0])+pos_offset), quat="1 0 0 0", mesh="stick" if broken==0 else "cut_stick", rgba=f"{color[0]} {color[1]} {color[2]} 1", mass=f"{self.mass['stick']*(1-broken)}", friction="1.0 .0 .0", priority="2")
        elif self.mesh_dict["stick"] == "CYLINDER":
            # assert broken == 0, "The CYLINDER mesh does not support broken stick yet" 
            XMLBuilder.create_element(stick, "geom", name=f"stick{idx}", type="cylinder", pos=vec2string(np.array([0,0,length/2 - length*(1-broken) /2])+pos_offset), quat="1 0 0 0", size=f"{radius} {length*(1-broken) /2 }", rgba=f"{color[0]} {color[1]} {color[2]} 1", mass=f"{self.mass['stick']*(1-broken)}", friction="1.0 .0 .0", priority="2")
        elif self.mesh_dict["stick"] == "CAPSULE":
            XMLBuilder.create_element(stick, "geom", name=f"stick{idx}", type="capsule", pos=vec2string(np.array([0,0,length/2 - length*(1-broken) /2])+pos_offset), quat="1 0 0 0", size=f"{radius} {length/2 *(1-broken)}", rgba=f"{color[0]} {color[1]} {color[2]} 1", mass=f"{self.mass['stick']*(1-broken)}", friction="1.0 .0 .0", priority="2")
        else:
            raise ValueError("The mesh should be either a .obj file or CYLINDER or CAPSULE")


        self.passive_idx_counter += 1
        self.leg_nodes.append(stick)
        node_id = len(self.leg_nodes)-1

        return node_id, stick
    

    def add_sock(self, parent_id, radius=0.04, length=0.08, stick_length=0.235, color=None, thickness=0.01):

        if color is None:
            color = self.colors[3]
        elif isinstance(color, (int, float)):
            color = self.colors[color]
        
        XMLBuilder.create_element(self.leg_nodes[parent_id], "geom", type="cylinder", name=f"sock{self.sock_idx_counter}", pos=f"0 0 {-stick_length/2+length/2-thickness} ", quat="1 0 0 0", size=f"{radius} {length/2}", rgba=f"{color[0]} {color[1]} {color[2]} 1", mass="0")
        self.sock_idx_counter += 1

    def add_simple_ball(self, parent_id=None, pos=None, quat=None, range="-90 90", color=None, passive=False):
        """
        Build a perfect sphere module with two hemispheres.
        
        This is a simplified version of add_module that creates a spherical module
        with proper orientation and joint configuration.

        Args:
            parent_id (Optional[int]): ID of the parent node
            pos (Optional[Vector3]): Position vector [x, y, z]
            quat (Optional[Quaternion]): Orientation quaternion [w, x, y, z]
            range (str): Joint range in degrees, format: "min max"
            color (Optional[Color]): RGB color tuple for the module
            passive (bool): Whether to create a passive (non-actuated) ball

        Returns:
            List[int]: IDs of the upper and lower hemispheres
        """
        quat_r = construct_quaternion([1., 0, 0], np.pi)
        if not passive:
            self._add_one_module(parent_id, pos, quat, range, color, quat_r=quat_r, joint_axis=[0, 0, -1])
        else:
            self._add_passive_ball(parent_id, pos, quat, range, color, quat_r=quat_r, joint_axis=[0, 0, -1])
        return [len(self.leg_nodes)-2, len(self.leg_nodes)-1]
    

    def change_color(self, node_id: int, color: Color) -> None:
        """
        Change the color of a node's geometry.
        
        Args:
            node_id: ID of the node to change color
            color: New RGB color
        """
        for geom in self.tree.iter('geom'):
            if 'rgba' in geom.attrib:
                if geom.attrib.get('name', '').endswith(str(node_id)):
                    geom.set('rgba', f"{color[0]} {color[1]} {color[2]} 1")

    def change_color_name(self, name: str, color: Color) -> None:
        """
        Change the color of all geometries containing a name.
        
        Args:
            name: Name substring to match
            color: New RGB color
        """
        for geom in self.tree.iter('geom'):
            if 'rgba' in geom.attrib and 'name' in geom.attrib and name in geom.attrib['name']:
                geom.set('rgba', f"{color[0]} {color[1]} {color[2]} 1")

    def remove_contact_name(self, name: str) -> None:
        """
        Remove contact properties from geometries containing a name.
        
        Args:
            name: Name substring to match
        """
        for geom in self.tree.iter('geom'):
            if 'name' in geom.attrib and name in geom.attrib['name']:
                geom.set('contype', "2")
                geom.set('conaffinity', "0")

    def delete_body(self, body_name: str, keep_tag: Optional[str] = None) -> None:
        """
        Delete a body and its related elements.
        
        Args:
            body_name: Name of the body to delete
            keep_tag: Optional tag type to preserve
        """
        for body in self.tree.iter('body'):
            if 'name' in body.attrib and body_name in body.attrib['name']:
                if keep_tag is None:
                    parent = body.getparent()
                    if parent is not None:
                        parent.remove(body)
                    # Remove related contact exclusions
                    for contact in self.root.findall('contact'):
                        for exclude in contact.findall('exclude'):
                            if exclude.attrib.get('body1') == body_name or exclude.attrib.get('body2') == body_name:
                                contact.remove(exclude)
                else:
                    # Remove all child elements except those with the specified tag
                    to_remove = [child for child in body if child.tag != keep_tag]
                    for child in to_remove:
                        body.remove(child)

    def delete_sensor(self, sensor_name: str) -> None:
        """
        Delete sensors containing a name.
        
        Args:
            sensor_name: Name substring to match
        """
        for sensor in self.tree.iter('sensor'):
            for child in sensor:
                if 'name' in child.attrib and sensor_name in child.attrib['name']:
                    sensor.remove(child)

    def delete_joint(self, joint_name: str) -> None:
        """
        Delete a joint and its actuator.
        
        Args:
            joint_name: Name of the joint to delete
        """
        # Delete joint
        for joint in self.tree.iter('joint'):
            if 'name' in joint.attrib and joint_name in joint.attrib['name']:
                parent = joint.getparent()
                if parent is not None:
                    parent.remove(joint)
        
        # Delete associated actuator
        for actuators in self.tree.iter('actuator'):
            for actuator in actuators:
                if 'joint' in actuator.attrib and joint_name in actuator.attrib['joint']:
                    parent = actuator.getparent()
                    if parent is not None:
                        parent.remove(actuator)

    def hind_imu(self, name: str) -> None:
        """
        Hide an IMU site by making it transparent.
        
        Args:
            name: Name of the IMU site
        """
        for site in self.tree.iter('site'):
            if 'name' in site.attrib and name in site.attrib['name']:
                site.set('rgba', "0 0 0 0")

    def save(self, filename: str = "m1") -> str:
        """
        Save the robot model to an XML file.
        
        Args:
            filename (str): Name of the file (without extension)
            
        Returns:
            str: Full path to the saved file
            
        Raises:
            RobotBuilderError: If saving fails
        """
        if not filename.endswith(".xml"):
            if os.path.isdir(filename):
                filename = os.path.join(filename, "robot.xml")
            else:
                filename = os.path.join(METAMACHINE_ROOT_DIR, "assets", "robots", f'{filename}.xml')

        try:
            self.tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')
            print(f"Saved the robot to {filename}")
            return filename
        except Exception as e:
            raise RobotBuilderError(f"Failed to save robot model: {str(e)}")

    def get_xml(self, fix_file_path: bool = False) -> str:
        """
        Get the XML string representation of the robot model.
        
        Args:
            fix_file_path: Whether to fix file paths in the XML
            
        Returns:
            str: XML string
        """
        root = fix_model_file_path(self.root) if fix_file_path else self.root
        return etree.tostring(root, pretty_print=True, xml_declaration=False, encoding='utf-8').decode()

    @property
    def init_quat(self) -> np.ndarray:
        """Get the initial quaternion orientation."""
        return quaternion_from_vectors(self.lleg_vec, np.array([1, 0, 0]))

    def _build_torsor(self) -> None:
        """Build a torso element with a free joint."""
        torso = XMLBuilder.create_element(
            self.worldbody,
            "body",
            name=f"torso{len(self.torsos)}",
            pos=vec2string(self.robot_cfg.initial_pos)
        )
        XMLBuilder.create_element(torso, "freejoint", name=f"root{len(self.torsos)}")
        XMLBuilder.create_element(torso, "camera", name=f"follow_camera{len(self.torsos)}", mode="track", pos="0 -2 1", xyaxes="1 0 0 0 1 2")
        self.torsos.append(torso)

    def _create_module_body(
        self,
        parent: etree.Element,
        name: str,
        pos: Union[str, Vector3],
        quat: Union[str, Quaternion],
        color: Color,
        joint_params: Optional[Dict[str, str]] = None
    ) -> etree.Element:
        """Create a module body with optional joint."""
        if not isinstance(pos, str):
            pos = vec2string(pos)
        if not isinstance(quat, str):
            quat = vec2string(quat)
            
        body = XMLBuilder.create_element(
            parent,
            "body",
            name=name,
            pos=pos,
            quat=quat
        )
        
        if joint_params is not None:
            XMLBuilder.create_element(body, "joint", **joint_params)
            
        return body

    def _add_geom_to_body(
        self,
        body: etree.Element,
        geom_type: str,
        name: str,
        color: Color,
        mass: float,
        **kwargs
    ) -> None:
        """Add a geometry element to a body."""
        attrs = {
            "type": geom_type,
            "name": name,
            "rgba": f"{color[0]} {color[1]} {color[2]} 1",
            "mass": str(mass),
            **kwargs
        }
        XMLBuilder.create_element(body, "geom", **attrs)

    def _add_module_sensors(
        self,
        l_body: etree.Element,
        r_body: etree.Element,
        idx: int
    ) -> None:
        """Add IMU and other sensors to a module."""
        # Left body sensors
        XMLBuilder.create_element(
            self.sensors,
            "framequat",
            name=f"imu_quat{idx}",
            objtype="xbody",
            objname=f"l{idx}"
        )
        XMLBuilder.create_element(
            l_body,
            "site",
            name=f"imu_site{idx}",
            pos="0 0 0",
            size="0.01",
            rgba="0 0 1 1"
        )
        
        sensor_types = ["gyro", "velocimeter", "accelerometer"]
        sensor_names = ["imu_gyro", "imu_vel", "imu_acc"]
        for sensor_type, name in zip(sensor_types, sensor_names):
            XMLBuilder.create_element(
                self.sensors,
                sensor_type,
                name=f"{name}{idx}",
                site=f"imu_site{idx}"
            )

        sensor_types = ["framelinvel"]
        sensor_names = ["imu_globvel"]
        for sensor_type, name in zip(sensor_types, sensor_names):
            XMLBuilder.create_element(
                self.sensors,
                sensor_type,
                name=f"{name}{idx}",
                objtype="xbody",
                objname=f"l{idx}"
            )
            
        # Right body sensors
        XMLBuilder.create_element(
            self.sensors,
            "framequat",
            name=f"back_imu_quat{idx}",
            objtype="xbody",
            objname=f"r{idx}"
        )
        XMLBuilder.create_element(
            r_body,
            "site",
            name=f"back_imu_site{idx}",
            pos="0 0 0",
            size="0.01",
            rgba="0 0 1 1"
        )
        XMLBuilder.create_element(
            self.sensors,
            "gyro",
            name=f"back_imu_gyro{idx}",
            site=f"back_imu_site{idx}"
        )
        XMLBuilder.create_element(
            self.sensors,
            "velocimeter",
            name=f"back_imu_vel{idx}",
            site=f"back_imu_site{idx}"
        )

    def _add_actuator(self, joint_name: str, **kwargs) -> None:
        """Add an actuator for a joint."""
        # default_params = {
        #     "ctrlrange": "-3.14 3.14",
        #     "kp": "20",
        #     "kv": "0.5",
        #     "forcerange": "-12 12"
        # }
        # params = {**default_params, **kwargs, "joint": joint_name}
        # XMLBuilder.create_element(self.actuators, "position", **params)
        XMLBuilder.create_element(self.actuators, "motor", ctrlrange="-12 12", joint=joint_name)

    def _validate_port_id(self, port_id: Optional[int]) -> None:
        """Validate a port ID."""
        if port_id is not None and port_id >= len(self.ports):
            raise ValueError(f"Port ID {port_id} is out of range")

    def _validate_parent_id(self, parent_id: int) -> None:
        """Validate a parent ID."""
        if parent_id >= len(self.leg_nodes):
            raise ValueError(f"Parent ID {parent_id} is out of range")

    def _get_mesh_geom_params(
        self,
        mesh_type: str,
        color: Color,
        mass: float,
        **kwargs
    ) -> Dict[str, str]:
        """Get geometry parameters for a mesh type."""
        if mesh_type.endswith((".obj", ".stl")):
            return {
                "type": "mesh",
                "mesh": mesh_type,
                "rgba": f"{color[0]} {color[1]} {color[2]} 0.5",
                "mass": str(mass),
                "material": "metallic",
                "friction": "1.0 .0 .0",
                "priority": "2",
                **kwargs
            }
        elif mesh_type == "SPHERE":
            return {
                "type": "sphere",
                "size": str(self.robot_cfg.R),
                "rgba": f"{color[0]} {color[1]} {color[2]} 1",
                "mass": str(mass),
                "friction": "1.0 .0 .0",
                "priority": "2",
                **kwargs
            }
        else:
            raise ValueError(f"Unsupported mesh type: {mesh_type}")


if __name__ == "__main__":
    robot_cfg_air = RobotConfig(
        theta=0.4625123,
        R=0.07,
        r=0.03,
        l_=0.236,
        delta_l=0,
        stick_ball_l=0.005,
        a=0.236/4,
        stick_mass=0.1734,
        top_hemi_mass=0.1153,
        battery_mass=0.122,
        motor_mass=0.317,
        bottom_hemi_mass=0.1623,
        pcb_mass=0.1
    )

    mesh_dict_fine = {
        "up": "top_lid.obj",
        "bottom": "bottom_lid.obj",
        "stick": "leg4.4.obj",
        "battery": "battery.obj",
        "pcb": "pcb.obj",
        "motor": "motor.obj",
        "cut_stick": "legcut.obj"
    }
    
    builder = RobotBuilder(mesh_dict=mesh_dict_fine, robot_cfg=robot_cfg_air, terrain="flat")
    # builder.add_stairs(start_distance = 3)
    # builder.add_walls()
    builder.add_simple_ball()
    builder.add_independent_stick(0, pos=[0, 0, 0.1], quat=[1, 0, 0, 0], color=[1, 0, 0])
    f = builder.save("test001")
    view(f, False)