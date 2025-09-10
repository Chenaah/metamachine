"""Visual utilities for robot simulation and rendering.

This module provides utilities for:
- MuJoCo model manipulation and XML processing
- Color spectrum generation and manipulation
- Geometric calculations for robot visualization
- File path resolution and XML compilation

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
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Any

import numpy as np
from numpy import cos, sin
from lxml import etree
from matplotlib.pyplot import get_cmap

from metamachine import METAMACHINE_ROOT_DIR

def get_joint_pos_addr(model) -> np.ndarray:
    """Get joint position addresses in the qpos array.
    
    Assumes joint names follow the pattern 'joint{i}' where i is the joint index.
    Alternative: use d.joint('name').qpos for named access.
    
    Args:
        model: MuJoCo model object
        
    Returns:
        Array of joint position addresses in qpos
    """
    joint_idx = [model.joint(f'joint{i}').id for i in range(model.nu)]
    return model.jnt_qposadr[joint_idx]





def create_color_spectrum(colormap_name: str = 'plasma', 
                         num_colors: int = 100) -> List[Tuple[float, float, float, float]]:
    """Create a color spectrum using Matplotlib colormap.

    Args:
        colormap_name: Name of the Matplotlib colormap
        num_colors: Number of colors in the spectrum

    Returns:
        List of RGBA tuples representing the color spectrum
        
    Raises:
        ValueError: If colormap_name is invalid
    """
    try:
        colormap = get_cmap(colormap_name)
    except ValueError as e:
        raise ValueError(f"Invalid colormap name '{colormap_name}': {e}")
        
    colors = [colormap(i / num_colors) for i in range(num_colors)]
    return colors


def vec2string(arr: Union[str, List[float], np.ndarray]) -> str:
    """Convert array or list to space-separated string.
    
    Args:
        arr: Array, list, or string to convert
        
    Returns:
        Space-separated string representation
    """
    if isinstance(arr, str):
        return arr
    
    arr = np.asarray(arr)
    return ' '.join(arr.astype(str))



def euler_to_rotation_matrix(euler_angles: Union[List[float], np.ndarray]) -> np.ndarray:
    """Convert Euler angles to 3D rotation matrix.
    
    Args:
        euler_angles: Euler angles [roll, pitch, yaw] in degrees
        
    Returns:
        3x3 rotation matrix
        
    Note:
        Input angles are assumed to be in degrees and are converted to radians.
        Uses ZYX intrinsic rotation order (yaw-pitch-roll).
    """
    euler_angles = np.asarray(euler_angles)
    roll, pitch, yaw = euler_angles
    
    # Convert to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # Trigonometric values
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    # ZYX rotation matrix
    rotation_matrix = np.array([
        [cp*cy, cp*sy, -sp],
        [sr*sp*cy - cr*sy, sr*sp*sy + cr*cy, sr*cp],
        [cr*sp*cy + sr*sy, cr*sp*sy - sr*cy, cr*cp]
    ])
    
    return rotation_matrix

def random_point_at_distance(origin: Union[List[float], np.ndarray], 
                            distance: float) -> np.ndarray:
    """Generate a random point at specified distance from origin.
    
    Args:
        origin: Origin point [x, y, z]
        distance: Distance from origin
        
    Returns:
        Random point at specified distance from origin
    """
    origin = np.asarray(origin)
    
    # Generate random unit direction vector
    direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)

    # Scale by distance and add to origin
    return origin + distance * direction


def lighten_color(rgb: Union[List[float], Tuple[float, float, float]], 
                 factor: float) -> List[float]:
    """Lighten an RGB color by blending with white.
    
    Args:
        rgb: RGB color tuple/list (r, g, b) with values in [0, 1]
        factor: Blend factor in [0, 1]. 0 = original color, 1 = white
        
    Returns:
        Lightened RGB color as list [r, g, b]
        
    Raises:
        ValueError: If factor is not in [0, 1]
    """
    if not (0 <= factor <= 1):
        raise ValueError("Factor must be between 0 and 1")

    r, g, b = rgb[0], rgb[1], rgb[2]
    
    # Blend with white
    new_r = r + (1 - r) * factor
    new_g = g + (1 - g) * factor
    new_b = b + (1 - b) * factor

    # Clamp to valid range
    new_r = np.clip(new_r, 0, 1)
    new_g = np.clip(new_g, 0, 1)
    new_b = np.clip(new_b, 0, 1)

    return [new_r, new_g, new_b]


def get_jing_vector(alpha: float, theta: float) -> np.ndarray:
    """Calculate the Jing vector for robot geometry.
    
    Args:
        alpha: Alpha angle parameter
        theta: Theta angle parameter
        
    Returns:
        Normalized Jing vector
        
    Note:
        This function handles the special case when alpha is near zero
        by applying a small epsilon to avoid division by zero.
    """
    epsilon = 1e-5
    
    if abs(alpha) < epsilon:
        # Special case: alpha ≈ 0
        jing_vec = np.array([0, cos(theta), sin(theta)])
    else:
        # Apply epsilon if alpha is very small but non-zero
        if 0 < alpha < epsilon:
            alpha = epsilon
        elif -epsilon < alpha < 0:
            alpha = -epsilon
            
        jing_vec = np.array([
            (cos(alpha) - 1) / sin(theta),
            1,
            ((1 - cos(alpha))**2 + sin(theta) * sin(alpha)) / (cos(theta) * sin(alpha))
        ])
    
    return jing_vec / np.linalg.norm(jing_vec)


def get_local_zvec(alpha: float, theta: float) -> np.ndarray:
    """Calculate the local Z vector for robot geometry.
    
    Args:
        alpha: Alpha angle parameter
        theta: Theta angle parameter
        
    Returns:
        Normalized local Z vector
    """
    jing_vec = get_jing_vector(alpha, theta)
    mid_vec = np.array([cos(theta) * sin(theta), cos(theta) * (1 - cos(alpha)), 0])
    local_zvec = np.cross(jing_vec, mid_vec)
    
    # Handle case where cross product might be zero
    norm = np.linalg.norm(local_zvec)
    if norm < 1e-8:
        return np.array([0, 0, 1])  # Default to z-axis
        
    return local_zvec / norm


def fix_model_file_path(root: etree._Element) -> etree._Element:
    """Fix mesh file paths in XML to use absolute paths.
    
    Args:
        root: XML root element
        
    Returns:
        Modified XML root element with fixed file paths
    """
    parts_dir = os.path.join(METAMACHINE_ROOT_DIR, "sim", "assets", "parts")
    
    if not os.path.exists(parts_dir):
        print(f"Warning: Parts directory not found: {parts_dir}")
        return root
    
    # Build mapping of filename to full path
    mesh_files = {}
    for file_path in Path(parts_dir).rglob('*'):
        if file_path.is_file():
            mesh_files[file_path.name] = str(file_path.resolve())
    
    # Fix mesh and hfield file paths
    for element in root.findall('.//mesh') + root.findall('.//hfield'):
        file_attr = element.get('file')
        if file_attr and file_attr in mesh_files:
            element.set('file', mesh_files[file_attr])
    
    return root

def position_to_torque_control(root: etree._Element) -> etree._Element:
    """Convert position control actuators to torque control.
    
    Args:
        root: XML root element
        
    Returns:
        Modified XML root element with torque control
    """
    for position in root.findall('.//position'):
        position.tag = 'motor'
        
        # Remove position control parameters
        if 'kp' in position.attrib:
            del position.attrib['kp']
        if 'kv' in position.attrib:
            del position.attrib['kv']
            
        # Convert force range to control range
        if 'forcerange' in position.attrib:
            position.attrib['ctrlrange'] = position.attrib['forcerange']
            del position.attrib['forcerange']
    
    return root

def torque_to_position_control(root: etree._Element, 
                              kp: float = 20, 
                              kd: float = 0.5) -> etree._Element:
    """Convert torque control actuators to position control.
    
    Args:
        root: XML root element
        kp: Position control proportional gain
        kd: Position control derivative gain
        
    Returns:
        Modified XML root element with position control
    """
    for motor in root.findall('.//motor'):
        motor.tag = 'position'
        motor.attrib['kp'] = f"{kp}"
        motor.attrib['kv'] = f"{kd}"
        
        # Convert control range to force range
        if 'ctrlrange' in motor.attrib:
            motor.attrib['forcerange'] = motor.attrib['ctrlrange']
            del motor.attrib['ctrlrange']
    
    return root

def update_xml_timestep(root: etree._Element, timestep: float) -> etree._Element:
    """Update simulation timestep in XML.
    
    Args:
        root: XML root element
        timestep: New timestep value
        
    Returns:
        Modified XML root element
        
    Raises:
        IndexError: If no RK4 integrator option is found
    """
    options = root.xpath('//option[@integrator="RK4"]')
    if not options:
        raise IndexError("No RK4 integrator option found in XML")
        
    option = options[0]
    option.attrib['timestep'] = f"{timestep}"
    return root

def compile_xml(xml_file: Union[str, os.PathLike], 
               torque_control: bool = False, 
               timestep: Optional[float] = None) -> str:
    """Compile and process XML file for MuJoCo simulation.
    
    Loads an XML file and applies various transformations:
    - Fixes mesh file paths to absolute paths
    - Optionally converts to torque control
    - Optionally updates simulation timestep
    
    Args:
        xml_file: Path to XML file
        torque_control: Whether to convert to torque control
        timestep: Optional new timestep value
        
    Returns:
        Processed XML string
        
    Raises:
        FileNotFoundError: If XML file doesn't exist
        etree.XMLSyntaxError: If XML is malformed
    """
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"XML file not found: {xml_file}")
    
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(xml_file, parser)
        root = tree.getroot()
    except etree.XMLSyntaxError as e:
        raise etree.XMLSyntaxError(f"Invalid XML in {xml_file}: {e}")

    # Apply transformations
    root = fix_model_file_path(root)
    
    if torque_control:
        root = position_to_torque_control(root)
        
    if timestep is not None:
        try:
            root = update_xml_timestep(root, timestep)
        except IndexError:
            print(f"Warning: Could not update timestep in {xml_file} - no RK4 integrator found")

    # Convert to string
    xml_string = etree.tostring(
        root, 
        pretty_print=True, 
        xml_declaration=False, 
        encoding='utf-8'
    ).decode()
    
    return xml_string




def is_headless() -> bool:
    """Check if running in headless environment (no display).
    
    Returns:
        True if no display is available, False otherwise
    """
    return not os.getenv('DISPLAY')



def numpy_to_native(obj: Any) -> Any:
    """Recursively convert numpy objects to native Python types.
    
    Args:
        obj: Object that may contain numpy arrays or scalars
        
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: numpy_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(numpy_to_native(value) for value in obj)
    else:
        return obj