"""Mathematical utilities for robotics applications.

This module provides:
- Quaternion operations and conversions
- Rotation matrix computations
- Coordinate frame transformations
- Angular velocity calculations
- Filtering and signal processing utilities

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

from typing import Union, List, Tuple, Optional

import numpy as np
import torch

# Optional JAX support
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None


def quat_rotate_inverse_jax(q, v):
    """JAX implementation of inverse quaternion rotation (batch).
    
    Args:
        q: Quaternion array of shape (N, 4) in [x, y, z, w] format
        v: Vector array of shape (N, 3)
        
    Returns:
        Rotated vectors of shape (N, 3)
        
    Raises:
        ImportError: If JAX is not available
    """
    if not HAS_JAX:
        raise ImportError("JAX is required for this function. Install with: pip install jax")
        
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).reshape(-1, 1)
    b = jnp.cross(q_vec, v, axis=-1) * q_w.reshape(-1, 1) * 2.0
    c = q_vec * jnp.einsum('bi,bi->b', q_vec, v).reshape(-1, 1) * 2.0
    return a - b + c

def quat_rotate_inverse_jax_wxyz(q, v):
    """JAX implementation of inverse quaternion rotation with WXYZ format (batch).
    
    Args:
        q: Quaternion array of shape (N, 4) in [w, x, y, z] format
        v: Vector array of shape (N, 3)
        
    Returns:
        Rotated vectors of shape (N, 3)
        
    Raises:
        ImportError: If JAX is not available
    """
    if not HAS_JAX:
        raise ImportError("JAX is required for this function. Install with: pip install jax")
        
    q_w = q[:, 0]  # Extract w component
    q_vec = q[:, 1:]  # Extract x, y, z components
    a = v * (2.0 * q_w ** 2 - 1.0).reshape(-1, 1)
    b = jnp.cross(q_vec, v, axis=-1) * q_w.reshape(-1, 1) * 2.0
    c = q_vec * jnp.einsum('bi,bi->b', q_vec, v).reshape(-1, 1) * 2.0
    return a - b + c


def quat_rotate_inverse_batch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """PyTorch batch implementation of inverse quaternion rotation.
    
    Args:
        q: Quaternion tensor of shape (N, 4) in [x, y, z, w] format
        v: Vector tensor of shape (N, 3)
        
    Returns:
        Rotated vectors of shape (N, 3)
    """
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(
        q_vec.view(shape[0], 1, 3), 
        v.view(shape[0], 3, 1)
    ).squeeze(-1) * 2.0
    return a - b + c


def quat_rotate_inverse(q: Union[List[float], np.ndarray], 
                        v: Union[List[float], np.ndarray]) -> np.ndarray:
    """Rotate a vector in the inverse direction of a given quaternion.

    Args:
        q: Quaternion [x, y, z, w] or array-like
        v: Vector [x, y, z] or array-like

    Returns:
        Rotated vector as numpy array
    """
    q = np.asarray(q)
    v = np.asarray(v)
    
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def quaternion_to_rotation_matrix(q: Union[List[float], np.ndarray]) -> np.ndarray:
    """Convert a quaternion to a 3x3 rotation matrix.
    
    Args:
        q: Quaternion [x, y, z, w] or array-like
        
    Returns:
        3x3 rotation matrix as numpy array
    """
    q = np.asarray(q)
    x, y, z, w = q[0], q[1], q[2], q[3]
    
    rotation_matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return rotation_matrix

def calculate_angular_velocity(initial_quaternion: Union[List[float], np.ndarray], 
                              final_quaternion: Union[List[float], np.ndarray], 
                              dt: float) -> np.ndarray:
    """Calculate angular velocity from two quaternions and time difference.
    
    Args:
        initial_quaternion: Initial quaternion [x, y, z, w]
        final_quaternion: Final quaternion [x, y, z, w]
        dt: Time difference in seconds
        
    Returns:
        Angular velocity vector [wx, wy, wz]
        
    Raises:
        ValueError: If dt is zero or negative
    """
    if dt <= 0:
        raise ValueError("Time difference must be positive")
        
    # Convert quaternions to rotation matrices
    R1 = quaternion_to_rotation_matrix(initial_quaternion)
    R2 = quaternion_to_rotation_matrix(final_quaternion)

    # Compute the rotation matrix derivative
    dR = (R2 - R1) / dt

    # Extract the skew-symmetric part to get the angular velocity vector
    angular_velocity = np.array([dR[2, 1], dR[0, 2], dR[1, 0]])

    return angular_velocity

class AverageFilter:
    """Moving average filter for smoothing sensor data.
    
    Attributes:
        window_size: Number of samples to average over
        buffer: Internal storage for recent values
    """
    
    def __init__(self, window_size: int):
        """Initialize the average filter.
        
        Args:
            window_size: Number of samples to average over
            
        Raises:
            ValueError: If window_size is not positive
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive")
            
        self.window_size = window_size
        self.buffer = []

    def reset_window_size(self, window_size: int) -> None:
        """Reset the window size and clear buffer.
        
        Args:
            window_size: New window size
        """
        if window_size <= 0:
            raise ValueError("Window size must be positive")
            
        self.window_size = window_size
        self.buffer = []
        
    def __call__(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Add new value and return moving average.
        
        Args:
            value: New value to add (scalar or array)
            
        Returns:
            Moving average of recent values
        """
        self.buffer.append(value)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        return np.mean(self.buffer, axis=0)
    
    def reset(self) -> None:
        """Clear the internal buffer."""
        self.buffer = []


def velocity_transform_yaw(V_world: Union[List[float], np.ndarray], 
                          quaternion: Union[List[float], np.ndarray]) -> np.ndarray:
    """Transform velocity from world frame to body frame using yaw angle only.
    
    Args:
        V_world: World frame velocity [vx, vy] (2D) or [vx, vy, vz] (3D)
        quaternion: Orientation quaternion [x, y, z, w] or [w, x, y, z]
        
    Returns:
        Body frame velocity (2D) considering only yaw rotation
    """
    V_world = np.asarray(V_world)
    quaternion = np.asarray(quaternion)
    
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(quaternion)
    
    # Extract yaw rotation (2x2 submatrix)
    R_yaw = R[:2, :2]
    
    # Transform velocity (use only x, y components)
    V_world_2d = V_world[:2] if len(V_world) >= 2 else V_world
    V_body = np.dot(R_yaw, V_world_2d.reshape(2, 1))
    
    return V_body.flatten()


def world_to_body_velocity_yaw(world_vel: Union[List[float], np.ndarray], 
                               quat: Union[List[float], np.ndarray]) -> np.ndarray:
    """Transform velocity from world frame to body frame using yaw angle only.
    
    Args:
        world_vel: 3D velocity vector in world frame [vx, vy, vz]
        quat: Quaternion [qw, qx, qy, qz] representing body orientation
        
    Returns:
        3D velocity vector in body frame, rotated only around z-axis (yaw)
    """
    world_vel = np.asarray(world_vel)
    quat = np.asarray(quat)
    
    # Normalize quaternion
    quat_norm = quat / np.linalg.norm(quat)
    
    # Extract yaw angle from quaternion (assuming WXYZ format)
    yaw = np.arctan2(
        2 * (quat_norm[0] * quat_norm[3] + quat_norm[1] * quat_norm[2]),
        1 - 2 * (quat_norm[2]**2 + quat_norm[3]**2)
    )
    
    # Create 3D rotation matrix for yaw-only rotation
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rotation_mat = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1]
    ])
    
    # Apply rotation to world velocity
    body_vel = np.dot(rotation_mat, world_vel)
    
    return body_vel

def world_velocity_to_forward_body_velocity(quaternion: Union[List[float], np.ndarray], 
                                           v_world: Union[List[float], np.ndarray]) -> float:
    """Convert world velocity to forward body velocity component.
    
    Args:
        quaternion: Quaternion representing orientation [x, y, z, w] or [w, x, y, z]
        v_world: World velocity vector [vx, vy, vz]
        
    Returns:
        Forward (x-component) body velocity as scalar
    """
    R = quaternion_to_rotation_matrix(quaternion)
    v_world = np.asarray(v_world)
    v_body = np.dot(R, v_world)
    return float(v_body[0])  # Return x-component as scalar

def quaternion_multiply(q1: Union[List[float], np.ndarray], 
                       q2: Union[List[float], np.ndarray]) -> np.ndarray:
    """Multiply two quaternions in WXYZ format.
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
        
    Returns:
        Product quaternion [w, x, y, z]
    """
    q1, q2 = np.asarray(q1), np.asarray(q2)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def quaternion_multiply_alt(q1: Union[List[float], np.ndarray], 
                            q2: Union[List[float], np.ndarray]) -> np.ndarray:
    """Alternative quaternion multiplication with different order (WXYZ format).
    
    Args:
        q1: First quaternion [w, x, y, z]
        q2: Second quaternion [w, x, y, z]
        
    Returns:
        Product quaternion [w, x, y, z] with q2 * q1 order
        
    Note:
        This function computes q2 * q1 instead of q1 * q2
    """
    q1, q2 = np.asarray(q1), np.asarray(q2)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    # Calculate q2 * q1
    w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
    x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
    y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
    z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
    
    return np.array([w, x, y, z])

def quat_apply_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation to vectors (PyTorch batch version).
    
    Args:
        a: Quaternion tensor of shape (N, 4) in [x, y, z, w] format
        b: Vector tensor of shape (N, 3) or (..., 3)
        
    Returns:
        Rotated vectors with same shape as input b
    """
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

def quat_apply(a: Union[List[float], np.ndarray], 
               b: Union[List[float], np.ndarray]) -> np.ndarray:
    """Apply quaternion rotation to a vector (NumPy version).
    
    Args:
        a: Quaternion [x, y, z, w]
        b: Vector [x, y, z]
        
    Returns:
        Rotated vector
    """
    a, b = np.asarray(a), np.asarray(b)
    xyz = a[:3]
    t = np.cross(xyz, b) * 2
    return b + a[3] * t + np.cross(xyz, t)


def construct_quaternion(axis: Union[List[float], np.ndarray], 
                         angle: float, 
                         order: str = "wxyz") -> np.ndarray:
    """Construct a quaternion from axis-angle representation.
    
    Args:
        axis: Rotation axis vector [x, y, z]
        angle: Rotation angle in radians
        order: Output format, either "wxyz" or "xyzw"
        
    Returns:
        Quaternion in specified format
        
    Raises:
        ValueError: If axis is zero vector or order is invalid
    """
    axis = np.asarray(axis, dtype=np.float64)
    
    # Check for zero axis
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        raise ValueError("Axis vector cannot be zero")
        
    axis = axis / axis_norm  # Normalize axis

    # Calculate quaternion components
    half_angle = angle / 2
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)
    
    w = cos_half
    x = axis[0] * sin_half
    y = axis[1] * sin_half
    z = axis[2] * sin_half
    
    # Return in specified order
    if order == "wxyz":
        return np.array([w, x, y, z])
    elif order == "xyzw":
        return np.array([x, y, z, w])
    else:
        raise ValueError(f"Invalid order '{order}'. Use 'wxyz' or 'xyzw'")

def euler_to_quaternion(euler_angles: Union[List[float], np.ndarray]) -> np.ndarray:
    """Convert Euler angles to quaternion (ZYX intrinsic rotation order).
    
    Args:
        euler_angles: Euler angles [roll, pitch, yaw] in radians
        
    Returns:
        Quaternion [w, x, y, z]
    """
    euler_angles = np.asarray(euler_angles)
    roll, pitch, yaw = euler_angles
    
    # Half angles
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Quaternion components
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])


def quaternion_to_euler(q: Union[List[float], np.ndarray]) -> np.ndarray:
    """Convert quaternion to Euler angles (ZYX intrinsic rotation order).
    
    Args:
        q: Quaternion [w, x, y, z]
    
    Returns:
        Euler angles [yaw, pitch, roll] in radians
        
    Note:
        This function returns [yaw, pitch, roll] order, which differs from
        the input order [roll, pitch, yaw] in euler_to_quaternion.
    """
    q = np.asarray(q)
    w, x, y, z = q

    # Yaw (Z-axis rotation)
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t0, t1)

    # Pitch (Y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)  # Clamp to avoid numerical issues
    pitch = np.arcsin(t2)

    # Roll (X-axis rotation)
    t3 = 2.0 * (w * x + y * z)
    t4 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t3, t4)

    return np.array([yaw, pitch, roll])

def quaternion_to_euler2(quaternion):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw).
    Quaternion should be in the form [w, x, y, z].
    """
    w, x, y, z = quaternion

    # Convert quaternion to rotation matrix
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)

    # Extract Euler angles
    roll = np.arctan2(r21, r22)
    pitch = np.arcsin(-r20)
    yaw = np.arctan2(r10, r00)

    return roll, pitch, yaw

def quat_rotate_batch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation to vectors (PyTorch batch version).
    
    Args:
        q: Quaternion tensor of shape (N, 4) in [x, y, z, w] format
        v: Vector tensor of shape (N, 3)
        
    Returns:
        Rotated vectors of shape (N, 3)
    """
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(
        q_vec.view(shape[0], 1, 3), 
        v.view(shape[0], 3, 1)
    ).squeeze(-1) * 2.0
    return a + b + c


def quat_rotate(q: Union[List[float], np.ndarray], 
                v: Union[List[float], np.ndarray]) -> np.ndarray:
    """Apply quaternion rotation to a vector (NumPy version).
    
    Args:
        q: Quaternion [x, y, z, w]
        v: Vector [x, y, z]
        
    Returns:
        Rotated vector
    """
    q, v = np.asarray(q), np.asarray(v)
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a + b + c


def ang_vel_to_ang_forward(ang_vel, projected_gravity, theta, alpha, onedir=True, return_y_vec=False):
    # this is for the new module model
    l = 1
    v0 = np.array([0,l*np.cos(theta),l*np.sin(theta)])
    v1 = np.array([l*np.sin(alpha),l*np.cos(alpha),-l*np.sin(theta)])
    # v0 /= np.linalg.norm(v0)
    # v1 /= np.linalg.norm(v1)
    forward_vec = (v0+v1) / np.linalg.norm((v0+v1))
    virtual_y = np.cross(forward_vec, projected_gravity)
    if virtual_y[1] < 0 and onedir:
        virtual_y *= -1

    ang_vel_forward = np.dot(virtual_y, ang_vel)
    if not return_y_vec:
        return ang_vel_forward
    else:
        return ang_vel_forward, virtual_y

def rotate_vector2D(v: Union[List[float], np.ndarray], theta: float) -> np.ndarray:
    """Rotate a 2D vector by angle theta.
    
    Args:
        v: 2D vector [x, y]
        theta: Rotation angle in radians
        
    Returns:
        Rotated 2D vector
        
    Raises:
        ValueError: If input vector is not 2D
    """
    v = np.asarray(v)
    if len(v) != 2:
        raise ValueError("Input vector must be 2D")
    
    # 2D rotation matrix
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                               [sin_theta,  cos_theta]])
    
    return np.dot(rotation_matrix, v)

def xyzw_to_wxyz(q: Union[List[float], np.ndarray]) -> np.ndarray:
    """Convert quaternion from XYZW to WXYZ format.
    
    Args:
        q: Quaternion [x, y, z, w]
        
    Returns:
        Quaternion [w, x, y, z]
    """
    q = np.asarray(q)
    return np.array([q[3], q[0], q[1], q[2]])

def wxyz_to_xyzw(q: Union[List[float], np.ndarray]) -> np.ndarray:
    """Convert quaternion from WXYZ to XYZW format.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Quaternion [x, y, z, w]
    """
    q = np.asarray(q)
    return np.array([q[1], q[2], q[3], q[0]])


def calculate_transformation_matrix(point_A, orientation_A, point_B, orientation_B):
    """
    Calculates the transformation matrix that aligns point_A in A's frame with point_B in B's frame,
    including their orientations.
    Args:
        point_A (numpy.ndarray): Connection point in A's frame (3x1).
        orientation_A (numpy.ndarray): Orientation matrix in A's frame (3x3).
        point_B (numpy.ndarray): Connection point in B's frame (3x1).
        orientation_B (numpy.ndarray): Orientation matrix in B's frame (3x3).

    Returns:
        numpy.ndarray: Transformation matrix (4x4).
    """
    # Transformation matrix for A's frame
    T_A = np.eye(4)
    T_A[:3, :3] = orientation_A
    T_A[:3, 3] = np.array(point_A)

    # Transformation matrix for B's frame
    T_B = np.eye(4)
    T_B[:3, :3] = orientation_B
    T_B[:3, 3] = np.array(point_B)

    # Invert the transformation matrix of part B to get the transformation from B to the connection point
    T_B_inv = np.linalg.inv(T_B)

    # The transformation matrix from A's frame to B's frame
    T_A_B = np.dot(T_A, T_B_inv)
    return T_A_B

def transform_point(T, point):
    """
    Transforms a point using a given transformation matrix.
    Args:
        T (numpy.ndarray): Transformation matrix (4x4).
        point (numpy.ndarray): Point to be transformed (3x1).

    Returns:
        numpy.ndarray: Transformed point (3x1).
    """
    point_homogeneous = np.append(point, 1)  # Convert to homogeneous coordinates
    transformed_point_homogeneous = np.dot(T, point_homogeneous)
    return transformed_point_homogeneous[:3]  # Convert back to Cartesian coordinates

def rotation_matrix(axis, angle):
    # Normalize the axis vector
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    
    u_x, u_y, u_z = axis
    
    # Compute the components of the rotation matrix
    cos_alpha = np.cos(angle)
    sin_alpha = np.sin(angle)
    one_minus_cos = 1 - cos_alpha
    
    R = np.array([
        [
            cos_alpha + u_x**2 * one_minus_cos,
            u_x * u_y * one_minus_cos - u_z * sin_alpha,
            u_x * u_z * one_minus_cos + u_y * sin_alpha
        ],
        [
            u_y * u_x * one_minus_cos + u_z * sin_alpha,
            cos_alpha + u_y**2 * one_minus_cos,
            u_y * u_z * one_minus_cos - u_x * sin_alpha
        ],
        [
            u_z * u_x * one_minus_cos - u_y * sin_alpha,
            u_z * u_y * one_minus_cos + u_x * sin_alpha,
            cos_alpha + u_z**2 * one_minus_cos
        ]
    ])
    
    return R

def rotation_matrix_multiply2(R1, R2):
    return np.dot(R2, R1)

def rotation_matrix_sequence(r_list):
    R = np.eye(3)
    for r in r_list:
        R = rotation_matrix_multiply2(R, r)
    return R


def rotation_matrix_to_quaternion(R):
    """
    Converts a rotation matrix to a quaternion.
    Args:
        R (numpy.ndarray): Rotation matrix (3x3).

    Returns:
        numpy.ndarray: Quaternion (4,).
    """
    q = np.empty((4,))
    t = np.trace(R)
    if t > 0:
        t = np.sqrt(t + 1.0)
        q[0] = 0.5 * t
        t = 0.5 / t
        q[1] = (R[2, 1] - R[1, 2]) * t
        q[2] = (R[0, 2] - R[2, 0]) * t
        q[3] = (R[1, 0] - R[0, 1]) * t
    else:
        i = np.argmax(np.diagonal(R))
        j = (i + 1) % 3
        k = (i + 2) % 3
        t = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0)
        q[i + 1] = 0.5 * t
        t = 0.5 / t
        q[0] = (R[k, j] - R[j, k]) * t
        q[j + 1] = (R[j, i] + R[i, j]) * t
        q[k + 1] = (R[k, i] + R[i, k]) * t
    return q

def matrix_to_pos_quat(T_A_B):
    origin_B_in_A = transform_point(T_A_B, np.zeros(3))
    R_A_B = T_A_B[:3, :3]
    quaternion_A_B = rotation_matrix_to_quaternion(R_A_B) # [1,0,0,0] # 
    return origin_B_in_A, quaternion_A_B

def normalize_angle(angle: float) -> float:
    """Normalize angle to [0, 2π) range.
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle in [0, 2π) range
    """
    return angle % (2 * np.pi)


def quaternion_from_vectors(v1: Union[List[float], np.ndarray], 
                           v2: Union[List[float], np.ndarray]) -> np.ndarray:
    """Calculate quaternion that rotates v1 to align with v2.
    
    Args:
        v1: Source vector [x, y, z]
        v2: Target vector [x, y, z]
        
    Returns:
        Quaternion [w, x, y, z] representing the rotation
        
    Raises:
        ValueError: If vectors are zero or parallel in opposite directions
    """
    v1, v2 = np.asarray(v1), np.asarray(v2)
    
    # Normalize input vectors
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm < 1e-8 or v2_norm < 1e-8:
        raise ValueError("Input vectors cannot be zero")
        
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm

    # Check for parallel vectors
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    
    if abs(dot_product - 1.0) < 1e-8:  # Same direction
        return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    elif abs(dot_product + 1.0) < 1e-8:  # Opposite directions
        # Find a perpendicular axis
        if abs(v1[0]) < 0.9:
            axis = np.cross(v1, [1, 0, 0])
        else:
            axis = np.cross(v1, [0, 1, 0])
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])  # 180° rotation

    # Calculate rotation axis and angle
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])  # Identity for parallel vectors
        
    axis = axis / axis_norm
    angle = np.arccos(dot_product)

    # Construct quaternion
    half_angle = angle / 2
    w = np.cos(half_angle)
    xyz = np.sin(half_angle) * axis

    return np.array([w, xyz[0], xyz[1], xyz[2]])




