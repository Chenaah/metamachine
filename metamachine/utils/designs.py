"""Robot design validation and testing utilities.

This module provides functions for validating robot designs through:
- Self-collision detection
- Stability analysis in different orientations
- Movability testing with random control inputs

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

import time
from typing import Dict, Optional, List, Union, Any

import numpy as np
import mujoco

from .visual_utils import get_joint_pos_addr
from .math_utils import quaternion_from_vectors
from .validation import is_list_like, is_number


def self_collision_check(m: mujoco.MjModel, 
                         d: mujoco.MjData, 
                         robot_properties: Dict[str, Any], 
                         viewer: Optional[Any] = None) -> None:
    """Check robot for self-collisions across random joint configurations.
    
    Tests the robot in various joint configurations to detect self-collisions.
    Updates robot_properties with collision statistics.
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        robot_properties: Dictionary to store results
        viewer: Optional MuJoCo viewer for visualization
    """
    n_sim_steps = 10
    n_random_joints = 100
    
    # Generate test configurations: zero position + random positions
    random_joints = [np.zeros(m.nu)] + [
        np.random.uniform(0, 2*np.pi, m.nu) 
        for _ in range(n_random_joints - 1)
    ]

    # Initialize robot state
    qpos = np.zeros(m.nq)
    qpos[2] = 1  # Set height
    qvel = np.zeros(m.nv)
    
    n_self_collision_list = []
    
    for joints in random_joints:
        qpos[get_joint_pos_addr(m)] = joints
        n_self_collision = 0
        
        for _ in range(n_sim_steps):
            step_start = time.time()

            d.qpos[:] = qpos
            d.qvel[:] = qvel
            mujoco.mj_step(m, d)

            # Count self-collisions (exclude floor contacts and valid module connections)
            for contact in d.contact:
                if contact.geom1 != 0 and contact.geom2 != 0:
                    try:
                        b1 = m.body(m.geom(contact.geom1).bodyid).name
                        b2 = m.body(m.geom(contact.geom2).bodyid).name
                        
                        # Allow contacts between left and right parts of same module
                        valid_contact = (
                            ((b1[0] == "l" and b2[0] == "r") or (b1[0] == "r" and b2[0] == "l"))
                            and (b1[1] == b2[1])
                        )
                        if not valid_contact:
                            n_self_collision += 1
                    except (IndexError, AttributeError):
                        # Skip malformed body names
                        continue

            # Optional visualization
            if viewer is not None:
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
                viewer.sync()
                
                # Basic time keeping
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        n_self_collision_list.append(n_self_collision)

    # Store results
    robot_properties["init_self_collision"] = n_self_collision_list[0] > 0
    robot_properties["self_collision_rate"] = (
        np.sum(np.array(n_self_collision_list) > 0) / n_random_joints
    )




def stable_state_check(m: mujoco.MjModel, 
                       d: mujoco.MjData, 
                       robot_properties: Dict[str, Any], 
                       theta: float, 
                       viewer: Optional[Any] = None) -> None:
    """Test robot stability in different initial orientations.
    
    Simulates the robot from various initial orientations to find stable states.
    Updates robot_properties with stability information.
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        robot_properties: Dictionary to store results
        theta: Robot design parameter for orientation calculation
        viewer: Optional MuJoCo viewer for visualization
    """
    qvel = np.zeros(m.nv)
    n_sim_steps = 100
    
    # Test orientations: identity + rotations to align with coordinate axes
    forward_vec = [0, np.cos(theta), np.sin(theta)]
    target_vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    init_quat_list = [[1, 0, 0, 0]]  # Identity quaternion
    for target_vec in target_vectors:
        try:
            quat = quaternion_from_vectors(forward_vec, target_vec)
            init_quat_list.append(quat)
        except ValueError:
            # Skip if vectors are problematic
            continue
    
    stable_quat_list = []
    stable_pos_list = []
    height_list = []

    for quat in init_quat_list:
        # Initialize robot state
        qpos = np.zeros(m.nq)
        qpos[2] = 1  # Set initial height
        qpos[3:7] = quat
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        
        # Simulate to find stable state
        for _ in range(n_sim_steps):
            step_start = time.time()
            mujoco.mj_step(m, d)

            # Optional visualization
            if viewer is not None:
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
                viewer.sync()
                
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        # Record final stable state
        stable_quat_list.append(d.qpos[3:7].copy())
        stable_pos_list.append(d.qpos[0:3].copy())

        # Calculate average height of all body positions
        xposes = np.array(d.xipos)
        average_height = np.mean(xposes[:, 2])
        height_list.append(average_height)

    # Store results
    robot_properties["stable_quat"] = stable_quat_list
    robot_properties["stable_pos"] = stable_pos_list
    robot_properties["stable_height"] = height_list



def movability_check(m: mujoco.MjModel, 
                    d: mujoco.MjData, 
                    robot_properties: Dict[str, Any], 
                    viewer: Optional[Any] = None, 
                    config_dict: Optional[Dict[str, Any]] = None, 
                    stable_pos_list: Optional[List[np.ndarray]] = None, 
                    stable_quat_list: Optional[List[np.ndarray]] = None) -> None:
    """Test robot movability with random control inputs.
    
    Applies random control inputs and measures the robot's ability to move.
    Updates robot_properties with average speed information.
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        robot_properties: Dictionary to store results
        viewer: Optional MuJoCo viewer for visualization
        config_dict: Configuration specifying initial position and orientation
        stable_pos_list: List of stable positions from stability test
        stable_quat_list: List of stable quaternions from stability test
        
    Raises:
        ValueError: If required configuration is missing
    """
    n_sim_steps = 100

    if config_dict is None:
        raise ValueError("config_dict must be provided for movability test")
    if "init_pos" not in config_dict:
        raise ValueError("init_pos must be provided in config_dict")
    if "init_quat" not in config_dict:
        raise ValueError("init_quat must be provided in config_dict")

    # Resolve initial position
    if is_list_like(config_dict["init_pos"]):
        init_pos = np.array(config_dict["init_pos"])
    elif is_number(config_dict["init_pos"]):
        if stable_pos_list is None:
            raise ValueError("stable_pos_list required when init_pos is an index")
        idx = int(config_dict["init_pos"])
        init_pos = stable_pos_list[idx]
    else:
        raise ValueError("init_pos must be array-like or integer index")
        
    # Resolve initial quaternion
    if is_list_like(config_dict["init_quat"]):
        init_quat = np.array(config_dict["init_quat"])
    elif is_number(config_dict["init_quat"]):
        if stable_quat_list is None:
            raise ValueError("stable_quat_list required when init_quat is an index")
        idx = int(config_dict["init_quat"])
        init_quat = stable_quat_list[idx]
    else:
        raise ValueError("init_quat must be array-like or integer index")

    # Initialize robot state
    qpos = np.zeros(m.nq)
    qpos[:3] = init_pos
    qpos[3:7] = init_quat
    qvel = np.zeros(m.nv)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    
    acc_speed = 0.0
    
    for _ in range(n_sim_steps):
        step_start = time.time()

        # Apply random control inputs (position control assumed)
        current_joints = d.qpos[get_joint_pos_addr(m)]
        random_offset = np.random.uniform(-1, 1, m.nu)
        d.ctrl[:] = current_joints + random_offset

        mujoco.mj_step(m, d)

        # Measure planar speed (x-y velocity)
        speed = np.linalg.norm(d.qvel[:2])
        acc_speed += speed

        # Optional visualization
        if viewer is not None:
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
            viewer.sync()
            
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Store average speed
    robot_properties["ave_speed"] = acc_speed / n_sim_steps