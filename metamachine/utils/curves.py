"""Reward curve functions for robotics control and reinforcement learning.

This module provides various reward shaping functions commonly used in
robotics applications, including forward velocity rewards, plateau functions,
and exponential tracking rewards.

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

from typing import Union

import numpy as np


def forward_reward_curve2_1(vel: float, cmd: float, cmd_scale: float) -> float:
    """Calculate forward velocity reward with piecewise linear curve.
    
    This function implements a reward curve that:
    - Gives maximum reward (1.0) when velocity matches command
    - Linearly decreases reward for deviations from command
    - Returns zero reward for excessive velocities
    
    Args:
        vel: Current velocity
        cmd: Commanded velocity  
        cmd_scale: Scale factor used when cmd=0
        
    Returns:
        Reward value between 0 and 1
    """
    if cmd > 0:
        if cmd <= vel <= 2 * cmd:
            return 1.0
        elif vel <= 0 or vel >= 4 * cmd:
            return 0.0
        elif 0 < vel < cmd:
            return vel / cmd
        elif 2 * cmd < vel < 4 * cmd:
            return -vel / (2 * cmd) + 2.0
    elif cmd < 0:
        if 2 * cmd <= vel <= cmd:
            return 1.0
        elif vel >= 0 or vel <= 4 * cmd:
            return 0.0
        elif cmd < vel < 0:
            return vel / cmd
        elif 4 * cmd < vel < 2 * cmd:
            return -vel / (2 * cmd) + 2.0
    else:  # cmd == 0
        if -cmd_scale/2 <= vel <= cmd_scale/2:
            return 1.0
        elif vel >= 3*cmd_scale/2 or vel <= -3*cmd_scale/2:
            return 0.0
        elif cmd_scale/2 < vel < 3*cmd_scale/2:
            return (vel - 3*cmd_scale/2) / (1 - 3*cmd_scale/2)
        elif -3*cmd_scale/2 < vel < -cmd_scale/2:
            return (vel + 3*cmd_scale/2) / (-1 + 3*cmd_scale/2)
    
    return 0.0  # Default fallback


def plateau(vel: float, max_desired_vel: float) -> float:
    """Calculate plateau-style reward for velocity.
    
    Creates a reward that increases linearly up to max_desired_vel,
    then plateaus at maximum reward (1.0) for higher velocities.
    
    Args:
        vel: Current velocity
        max_desired_vel: Velocity threshold for plateau
        
    Returns:
        Reward value between 0 and 1
    """
    if max_desired_vel > 0:
        if 0 < vel <= max_desired_vel:
            return vel / max_desired_vel
        elif vel > max_desired_vel:
            return 1.0
        else:
            return 0.0
    elif max_desired_vel < 0:
        if max_desired_vel <= vel < 0:
            return vel / max_desired_vel
        elif vel < max_desired_vel:
            return 1.0
        else:
            return 0.0
    else:
        return 0.0

def isaac_reward(desired_value: Union[float, np.ndarray], 
                current_value: Union[float, np.ndarray], 
                tracking_sigma: float = 0.25) -> float:
    """Calculate exponential tracking reward (Isaac Gym style).
    
    Computes an exponential reward based on the squared error between
    desired and current values. Commonly used in Isaac Gym environments.
    
    Args:
        desired_value: Target value(s)
        current_value: Current value(s)
        tracking_sigma: Exponential decay parameter (smaller = steeper decay)
        
    Returns:
        Reward value between 0 and 1 (approaches 1 for perfect tracking)
    """
    desired_value = np.asarray(desired_value)
    current_value = np.asarray(current_value)
    
    error = np.sum(np.square(desired_value - current_value))
    reward = np.exp(-error / tracking_sigma)
    
    return float(reward)