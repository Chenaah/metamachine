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

from enum import Enum
import os
from typing import Optional
from lxml import etree
import numpy as np

from metamachine import METAMACHINE_ROOT_DIR
from metamachine.robot_factory.core.xml_builder import XMLBuilder
from metamachine.utils.types import Vector3
from metamachine.utils.validation import is_list_like

class TerrainError(Exception):
    """Raised when there's an error with terrain configuration or generation."""
    pass


class TerrainType(Enum):
    """
    Supported terrain types for the robot environment.
    
    Each terrain type provides different characteristics for testing robot behavior:
    - FLAT: Basic flat surface
    - RUGGED: Uneven terrain with random height variations
    - STAIRS: Stepped terrain for climbing tests
    - BUMPY: Terrain with regular bumps
    - WALLS: Environment with boundary walls
    - RANDOM_BUMPY: Terrain with randomly placed bumps
    - DUNE: Sand dune-like terrain
    - SLOPE: Inclined surface
    - HFIELD: Height field based terrain
    """
    FLAT = "flat"
    RUGGED = "rugged"
    STAIRS = "stairs"
    BUMPY = "bumpy"
    WALLS = "walls"
    RANDOM_BUMPY = "random_bumpy"
    DUNE = "dune"
    SLOPE = "slope"
    HFIELD = "hfield"

class TerrainBuilder:
    def __init__(self, terrain_type: TerrainType, worldbody: etree.Element):
        if terrain_type is None:
            terrain_type = TerrainType.FLAT  # Default to flat terrain if not specified
        self.terrain_type = terrain_type
        self.worldbody = worldbody

    def build(self):
        if self.terrain_type == TerrainType.FLAT:
            self._add_flat()
        elif self.terrain_type == TerrainType.STAIRS:
            self._add_stairs()  # Stairs terrain implementation
        elif self.terrain_type == TerrainType.WALLS:
            self._add_walls()  # Walls terrain implementation
        else:
            raise NotImplementedError(f"Terrain type {self.terrain_type} is not supported yet.")
        
    def _add_flat(self) -> None:
        """Add floor to the world body."""
            
        self.floor_elem = XMLBuilder.create_element(
            self.worldbody,
            "geom",
            name="floor",
            pos="0 0 0",
            size="40 40 40",
            type="plane",
            material="matplane",
            conaffinity="1",
            condim="6",
            friction="1.0 .0 .0",
            priority="1"
        )

    def set_terrain(self, terrains, terrain_params=None):
        # Handle terrain configuration
        if not is_list_like(terrains):
            terrains = [terrains]
        if "bumpy" in terrains:
            self.add_bumps(h_func=lambda x : 0.04 + 0.02*x if 0.04 + 0.02*x < 0.1 else 0.1)
        if "walls" in terrains:
            transparent = terrain_params["transparent"] if terrain_params is not None and "transparent" in terrain_params else True
            angle = terrain_params["angle"] if terrain_params is not None and "angle" in terrain_params else False
            self.add_walls(transparent=transparent, angle=angle)
        if "random_bumpy" in terrains:
            terrain_params = terrain_params if terrain_params is not None else {"num_bumps": 200, "height_range": (0.01, 0.05), "width_range": (0.1, 0.1)}
            for _ in range(terrain_params["num_bumps"]):
                pos = np.random.uniform(-5, 5, 2)
                angle = np.random.uniform(0, 360)
                height = np.random.uniform(*terrain_params["height_range"])
                width = np.random.uniform(*terrain_params["width_range"])
                self.add_minibump(pos=pos, angle=angle, height=height, width=width, length=1)
        if "stairs" in terrains:
            num_steps = 20 if terrain_params is None or "num_steps" not in terrain_params else terrain_params["num_steps"]
            step_height = 0.1 if terrain_params is None or "step_height" not in terrain_params else terrain_params["step_height"]
            width = 0.5 if terrain_params is None or "width" not in terrain_params else terrain_params["width"]
            reserve_area = 2 if terrain_params is None or "reserve_area" not in terrain_params else terrain_params["reserve_area"]
            for i in range(reserve_area, num_steps+reserve_area):
                wall_center = reserve_area+width*(i-reserve_area)+width/2
                pos = [[wall_center, 0], [-wall_center, 0], [0, wall_center], [0, -wall_center]]
                for p in pos:
                    height = step_height*(i-reserve_area+1)
                    length = wall_center*2
                    angle = 0
                    if p[0] == 0:
                        length += width
                        angle = 90
                    else:
                        length -= width
                    assert length > 0, f"Length: {length}"
                    assert width > 0, f"Width: {width}"
                    assert height > 0, f"Height: {height}"
                    self.add_minibump(pos=p, angle=angle, height=height, width=width, length=length)

        if "ministairs" in terrains:
            num_steps = 20
            step_height = 0.1
            width = 0.5
            reserve_area = 2
            num_substeps = 2
            for i in range(num_steps):
                wall_center = reserve_area+width*i+width/2
                flag = (i+1)%(num_substeps*2+1)
                if flag > num_substeps:
                    flag = num_substeps*2-flag
                height = round(step_height*flag,2)
                # pdb.set_trace()
                print(f"Height: {height}")
                if height == 0:
                    continue
                pos = [[wall_center, 0], [-wall_center, 0], [0, wall_center], [0, -wall_center]]
                for p in pos:
                    length = wall_center*2
                    angle = 0
                    if p[0] == 0:
                        length += width
                        angle = 90
                    else:
                        length -= width
                    assert length > 0, f"Length: {length}"
                    assert width > 0, f"Width: {width}"
                    assert height > 0, f"Height: {height}"
                    self.add_minibump(pos=p, angle=angle, height=height, width=width, length=length)

        if "dune" in terrains:
            radius_x = 20 if terrain_params is None or "radius_x" not in terrain_params else terrain_params["radius_x"]
            radius_y = 20 if terrain_params is None or "radius_y" not in terrain_params else terrain_params["radius_y"]
            elevation_z = 0.5 if terrain_params is None or "elevation_z" not in terrain_params else terrain_params["elevation_z"]
            base_z = 0.01 if terrain_params is None or "base_z" not in terrain_params else terrain_params["base_z"]
            self.set_hfield(file="dune.png",radius_x=radius_x, radius_y=radius_y, elevation_z=elevation_z, base_z=base_z)

        if "slope" in terrains:
            radius_x = 20 if terrain_params is None or "radius_x" not in terrain_params else terrain_params["radius_x"]
            radius_y = 20 if terrain_params is None or "radius_y" not in terrain_params else terrain_params["radius_y"]
            elevation_z = 0.5 if terrain_params is None or "elevation_z" not in terrain_params else terrain_params["elevation_z"]
            base_z = 0.01 if terrain_params is None or "base_z" not in terrain_params else terrain_params["base_z"]
            self.set_hfield(file="wave20.png",radius_x=radius_x, radius_y=radius_y, elevation_z=elevation_z, base_z=base_z)

        if "hfield" in terrains:
            radius_x = 20 if terrain_params is None or "radius_x" not in terrain_params else terrain_params["radius_x"]
            radius_y = 20 if terrain_params is None or "radius_y" not in terrain_params else terrain_params["radius_y"]
            elevation_z = 0.5 if terrain_params is None or "elevation_z" not in terrain_params else terrain_params["elevation_z"]
            base_z = 0.01 if terrain_params is None or "base_z" not in terrain_params else terrain_params["base_z"]
            hfield = terrain_params["hfield"]
            self.set_hfield(file=hfield,radius_x=radius_x, radius_y=radius_y, elevation_z=elevation_z, base_z=base_z)


    def add_walls(self, transparent: bool = False, angle: float = 0) -> None:
        """
        Add boundary walls to the environment.
        
        Creates vertical walls to contain the robot within a specified area.
        The walls can be made transparent for visualization purposes.

        Args:
            transparent (bool): Whether the walls should be transparent
            angle (float): Rotation angle of the walls in degrees

        Returns:
            None
        """
        wall = XMLBuilder.create_element(
            self.worldbody,
            "body",
            name="boundary",
            pos="0 0 0",
            axisangle=f"0 0 1 {angle}"
        )
        
        wall_attrs = {
            "type": "box",
            "material": "boundary",
            "size": "25 0.1 0.5"
        }
        if transparent:
            wall_attrs["rgba"] = "0.1 0.1 0.1 0.0"
            
        XMLBuilder.create_element(
            wall,
            "geom",
            name="boundary/right",
            pos="0 1 0.25",
            **wall_attrs
        )
        XMLBuilder.create_element(
            wall,
            "geom",
            name="boundary/left",
            pos="0 -1 0.25",
            **wall_attrs
        )

    def add_bumps(self, h: float = 0.1, h_func: Optional[callable] = None) -> None:
        """
        Add bumps to the terrain for testing robot traversal capabilities.
        
        Creates a series of box-shaped bumps with configurable heights. The heights can be
        either uniform or determined by a custom function.

        Args:
            h (float): Uniform height of bumps if h_func is not provided
            h_func (Optional[callable]): Function that takes bump index and returns height.
                                       If provided, overrides the h parameter.

        Returns:
            None
        """
        for i in range(20):
            height = h_func(i) if h_func is not None else h
            XMLBuilder.create_element(
                self.worldbody,
                "geom",
                name=f"bump{i}",
                pos=f"{i+1} 0 {height/2}",
                type="box",
                material="boundary",
                size=f"0.1 25 {height}"
            )

    def add_minibump(
        self,
        pos: Vector3,
        angle: float,
        height: float = 0.1,
        width: float = 0.1,
        length: float = 1
    ) -> None:
        """
        Add a small bump obstacle to the terrain.
        
        Creates a single box-shaped bump with configurable dimensions and orientation.
        Useful for creating specific obstacle patterns or testing scenarios.

        Args:
            pos (Vector3): Position [x, y] of the bump center
            angle (float): Rotation angle in degrees around vertical axis
            height (float): Height of the bump in meters
            width (float): Width of the bump in meters
            length (float): Length of the bump in meters

        Returns:
            None
        """
        XMLBuilder.create_element(
            self.worldbody,
            "geom",
            name=f"obstacle{self.obstacle_idx_counter}",
            pos=f"{pos[0]} {pos[1]} {height/2}",
            axisangle=f"0 0 1 {angle}",
            type="box",
            material="boundary",
            size=f"{width/2} {length/2} {height/2}"
        )
        self.obstacle_idx_counter += 1

    def add_stairs(
        self,
        start_distance: float = 0,
        num_steps: int = 10,
        step_width: float = 2,
        step_height: float = 0.1,
        step_depth: float = 1,
        direction: str = 'x'
    ) -> None:
        """
        Add stairs to the terrain for testing climbing capabilities.
        
        Creates a staircase with configurable dimensions. The stairs can be oriented
        along either the x or y axis.

        Args:
            start_distance (float): Distance from origin to first step in meters
            num_steps (int): Number of steps in the staircase
            step_width (float): Width of each step in meters
            step_height (float): Height of each step in meters
            step_depth (float): Depth of each step in meters
            direction (str): Direction of stairs ('x' or 'y')
            
        Raises:
            TerrainError: If direction is not 'x' or 'y'
        """
        if direction not in ['x', 'y']:
            raise TerrainError(f"Invalid stairs direction: {direction}")
            
        for i in range(num_steps):
            if direction == 'x':
                x, y, z = i * step_depth + start_distance, 0, i * step_height
                self._create_box_xml(
                    f"step{i}",
                    x, y, z,
                    step_depth,
                    step_width,
                    step_height
                )
            else:  # direction == 'y'
                x, y, z = 0, i * step_depth + start_distance, i * step_height
                self._create_box_xml(
                    f"step{i}",
                    x, y, z,
                    step_width,
                    step_depth,
                    step_height
                )

    def set_hfield(
        self,
        file: str = "rugged.png",
        radius_x: float = 20,
        radius_y: float = 20,
        elevation_z: float = 0.3,
        base_z: float = 0.1
    ) -> None:
        """
        Set a height field for creating complex terrain from an image.
        
        The height field allows creating detailed terrain based on a grayscale image,
        where pixel intensity determines terrain height.

        Args:
            file (str): Path to height field image file (relative to assets/hfields)
            radius_x (float): X radius of the field in meters
            radius_y (float): Y radius of the field in meters
            elevation_z (float): Maximum elevation in meters
            base_z (float): Base elevation in meters
            
        Raises:
            TerrainError: If height field file doesn't exist
        """
        hfield_path = os.path.join(METAMACHINE_ROOT_DIR, "sim", "assets", "hfields", file)
        if not os.path.exists(hfield_path):
            raise TerrainError(f"Height field file not found: {hfield_path}")
            
        self.worldbody.remove(self.floor_elem)
        
        XMLBuilder.create_element(
            self.worldbody,
            "geom",
            name="floor",
            pos="0 0 0",
            type="hfield",
            material="hfield",
            conaffinity="1",
            condim="6",
            friction="1.0 .0 .0",
            hfield="rugged"
        )
        XMLBuilder.create_element(
            self.assets,
            "hfield",
            name="rugged",
            size=f"{radius_x} {radius_y} {elevation_z} {base_z}",
            file=file
        )

    def _create_box_xml(
        self,
        name: str,
        x: float,
        y: float,
        z: float,
        width: float,
        height: float,
        depth: float
    ) -> None:
        """
        Create a box geometry element.
        
        Args:
            name: Name of the box
            x, y, z: Position coordinates
            width: Width of the box
            height: Height of the box
            depth: Depth of the box
        """
        XMLBuilder.create_element(
            self.worldbody,
            'geom',
            name=name,
            type='box',
            pos=f"{x} {y} {z}",
            size=f"{width/2} {height/2} {depth/2}",
            material="boundary"
        )