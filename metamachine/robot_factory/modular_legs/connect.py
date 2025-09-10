"""
Author: Jing Gu, Chen Yu

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

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union
import numpy as np
from numpy import cos, sin, pi
from scipy.spatial.transform import Rotation
from metamachine.utils.types import Vector3
from ...utils.math_utils import rotation_matrix, rotation_matrix_sequence



class ConnectionType(Enum):
    """Enum for different types of connections between modules."""
    UP_BALL_TO_UP_BALL = "00"  # Ball module connecting to ball module
    UP_BALL_TO_STICK = "02"  # Ball module connecting to stick module
    BOTTOM_BALL_TO_UP_BALL = "10"  # Bottom ball connecting to ball
    BOTTOM_BALL_TO_STICK = "12"  # Bottom ball connecting to stick
    STICK_TO_BALL = "20"  # Stick connecting to ball
    STICK_TO_STICK = "22"  # Stick connecting to stick
    

@dataclass
class DockingPoint:
    """Represents a docking point between two modules."""
    pos_a: Vector3  # Local position in parent module
    pos_b: Vector3  # Local position in child module
    rotate_a: np.ndarray  # Rotation matrix in parent module
    rotate_b: np.ndarray  # Rotation matrix in child module
    position_a: int  # Position index in parent module
    position_b: int  # Position index in child module



class ConnectAsym(object):

    def __init__(self, conn_type: Union[str, ConnectionType], robot_cfg: Dict, lite: bool = False):

        self.robot_cfg = copy.deepcopy(robot_cfg)
        R = robot_cfg["R"]
        delta_l = robot_cfg["delta_l"]
        theta = robot_cfg["theta"]
        l = robot_cfg["l"]
        stick_ball_l = robot_cfg["stick_ball_l"]
        r = robot_cfg["r"] - stick_ball_l
        a = robot_cfg["a"]

        self.lite = lite
        self.j_list = [1,3]
        self.j_c_list = [0,2]
        
        self.side_pos_list = [a, 0, -a]
        self.side_j_list = [self.j_list, self.j_c_list, self.j_list]

        screw_thetas = [0, 2*pi/3, 4*pi/3]

        self.dock_list: List[DockingPoint] = []
        # self.dock_list = []
        
        dock_pos = [0, 2*pi/3, 4*pi/3] # docking positions along the hemisphere

        # Convert string connection type to enum if needed
        if isinstance(conn_type, str):
            conn_type = next(ct for ct in ConnectionType if ct.value == conn_type)
            
        
        # conn_type == "00" or conn_type == "10" or conn_type == "02" or conn_type == "12"
        
        # if conn_type[0] in ["0", "1"]:
        if conn_type in [ConnectionType.UP_BALL_TO_UP_BALL, ConnectionType.BOTTOM_BALL_TO_UP_BALL,
                        ConnectionType.UP_BALL_TO_STICK, ConnectionType.BOTTOM_BALL_TO_STICK]:
            # ball <- ball or ball <- stick
            # screw_thetas = [0, 2*pi/3, 4*pi/3, pi/3, pi, 5*pi/3] # screw the ball at each docking position
            self._generate_ball_connections(conn_type, R, r, delta_l, theta, l, stick_ball_l, dock_pos, screw_thetas)
        
        # elif conn_type == "20":
        elif conn_type in [ConnectionType.STICK_TO_BALL]:
            # stick <- ball
            # screw_thetas = [0, 2*pi/3, 4*pi/3, pi/3, pi, 5*pi/3] # screw the ball at each docking position
            self._generate_stick_to_ball_connections(conn_type, R, r, delta_l, theta, l, stick_ball_l, dock_pos, screw_thetas)
            
        # elif conn_type == "22":
        elif conn_type in [ConnectionType.STICK_TO_STICK]:
            # stick <- stick
            # screw_thetas = [0, 2*pi/3, 4*pi/3, pi/3, pi, 5*pi/3] # screw the ball at each docking position
            self._generate_stick_to_stick_connections(conn_type, R, r, delta_l, theta, l, stick_ball_l, screw_thetas)
            
        
    def _generate_ball_connections(self, conn_type, R, r, delta_l, theta, l, stick_ball_l,
                                         dock_pos, screw_thetas):
        for i, dock_theta_a in enumerate(dock_pos):
            for screw_theta in screw_thetas:
                pos_a = [(R-delta_l)*cos(theta)*sin(dock_theta_a), 
                            (R-delta_l)*cos(theta)*cos(dock_theta_a), 
                            (R-delta_l)*sin(theta)]
                rotate_a = rotation_matrix_sequence([
                    rotation_matrix([0,0,1],dock_theta_a),
                    rotation_matrix(pos_a,-(pi/3+screw_theta))
                ])
                # if conn_type == "00" or conn_type == "10":
                if conn_type in [ConnectionType.UP_BALL_TO_UP_BALL, ConnectionType.BOTTOM_BALL_TO_UP_BALL]:
                    for j, dock_theta_b in enumerate(dock_pos):
                        pos_b = [(R-delta_l)*cos(theta)*sin(dock_theta_b), 
                                    (R-delta_l)*cos(theta)*cos(dock_theta_b), 
                                    (R-delta_l)*sin(theta)]
                        rotate_b = rotation_matrix_sequence([
                            rotation_matrix([0,0,1],pi-dock_theta_a),
                            rotation_matrix([1,0,0],2*theta),
                            rotation_matrix([0,0,1],-dock_theta_b)
                            
                        ])
                        self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, i, j))

                elif conn_type in [ConnectionType.UP_BALL_TO_STICK, ConnectionType.BOTTOM_BALL_TO_STICK]:
                    pos_counter_b = 0
                    pos_b = [0, 0, (l-stick_ball_l)/2]
                    rotate_b = rotation_matrix_sequence([
                        rotation_matrix([cos(dock_theta_a),sin(dock_theta_a),0],-pi/2-theta),
                        rotation_matrix([0,0,1],pi/6)
                    ])
                    self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, i, pos_counter_b))
                    # self.dock_list.append(self.Dock(pos_a, pos_b, rotate_a, rotate_b, i, pos_counter_b))
                    pos_counter_b += 1

                    for m in range(3):
                        for j in self.side_j_list[m]:
                            pos_b = [r*cos(j*pi/2+pi/12), r*sin(j*pi/2+pi/12), self.side_pos_list[m]]
                            rotate_b = rotation_matrix_sequence([
                                rotation_matrix([0,0,1],pi/2+j*pi/2-dock_theta_a),
                                rotation_matrix([-sin(j*pi/2),cos(j*pi/2),0],-theta),
                                rotation_matrix([1,0,0],pi*((m+1)%2)),
                                rotation_matrix([0,0,1],pi/12+pi*((m+1)%2))
                            ])
                            self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, i, pos_counter_b))
                            
                            pos_counter_b += 1

                    pos_b = [0, 0, -(l-stick_ball_l)/2]
                    rotate_b = rotation_matrix_sequence([
                        rotation_matrix([cos(dock_theta_a),sin(dock_theta_a),0],-pi/2-theta),
                        rotation_matrix([0,0,1],pi/3),
                        rotation_matrix([1,0,0],pi),
                    ])
                    self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, i, pos_counter_b))
                    pos_counter_b += 1
                
    def _generate_stick_to_ball_connections(self, conn_type, R, r, delta_l, theta, l, stick_ball_l, dock_pos, screw_thetas):

        pos_counter = 0

        # tip of stick a
        pos_a = [0, 0, (l-stick_ball_l)/2]
        for screw_theta in screw_thetas:
            rotate_a = rotation_matrix_sequence([
                rotation_matrix([1,0,0],-pi/2-theta),
                rotation_matrix([0,0,1],-pi/6+screw_theta)
            ])
            
            for i, dock_theta_b in enumerate(dock_pos):
                pos_b = [(R-delta_l)*cos(theta)*sin(dock_theta_b), 
                                    (R-delta_l)*cos(theta)*cos(dock_theta_b), 
                                    (R-delta_l)*sin(theta)]
                rotate_b = rotation_matrix_sequence([
                            rotation_matrix([0,0,1],-dock_theta_b)
                        ])
                self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, pos_counter, i))
            
        pos_counter += 1

        # side of stick a
        for m in range(3):
            for j_a in self.side_j_list[m]:
                pos_a = [r*cos(j_a*pi/2+pi/12), r*sin(j_a*pi/2+pi/12), self.side_pos_list[m]]
                rotate_a = rotation_matrix_sequence([
                    rotation_matrix([1,0,0],pi*((m+1)%2)),
                    rotation_matrix([0,0,1],pi/12)
                ])
                for screw_theta in screw_thetas:
                    for i, dock_theta_b in enumerate(dock_pos):
                        pos_b = [(R-delta_l)*cos(theta)*sin(dock_theta_b), 
                                    (R-delta_l)*cos(theta)*cos(dock_theta_b), 
                                    (R-delta_l)*sin(theta)]
                        rotate_b = rotation_matrix_sequence([
                            rotation_matrix([0,0,1],pi/2-j_a*pi/2+pi*(m%2)),
                            rotation_matrix([1,0,0],theta),
                            rotation_matrix([0,0,1],-dock_theta_b),
                            rotation_matrix(pos_b,screw_theta+pi),
                        ])
                        self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, pos_counter, i))
                    
                pos_counter += 1
        
        # tip of stick a
        pos_a = [0, 0, -(l-stick_ball_l)/2]
        for screw_theta in screw_thetas:
            rotate_a = rotation_matrix_sequence([
                rotation_matrix([1,0,0],pi/2-theta),
                rotation_matrix([0,0,1],screw_theta+pi-pi/3)
            ])
            for i, dock_theta_b in enumerate(dock_pos):
                pos_b = [(R-delta_l)*cos(theta)*sin(dock_theta_b), 
                            (R-delta_l)*cos(theta)*cos(dock_theta_b), 
                            (R-delta_l)*sin(theta)]
                rotate_b = rotation_matrix_sequence([
                            rotation_matrix([0,0,1],-dock_theta_b)
                        ])
                self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, pos_counter, i))

        pos_counter += 1


    def _generate_stick_to_stick_connections(self, conn_type, R, r, delta_l, theta, l, stick_ball_l, screw_thetas):

        pos_counter = 0

        # tip of stick a
        pos_a = [0, 0, (l-stick_ball_l)/2]
        for screw_theta in screw_thetas:
            rotate_a = rotation_matrix_sequence([
                rotation_matrix([0,0,1],screw_theta)
            ])

            pos_counter_b = 0
            pos_b = [0, 0, (l-stick_ball_l)/2]
            rotate_b = Rotation.from_euler('xz', [pi, pi+pi/6]).as_matrix()
            self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
            pos_counter_b += 1

            for m in range(3):
                for j_b in self.side_j_list[m]:
                    pos_b = [r*cos(j_b*pi/2+pi/12), r*sin(j_b*pi/2+pi/12), self.side_pos_list[m]]
                    rotate_b = Rotation.from_euler('yz', [-pi/2, j_b*pi/2+pi/12]).as_matrix()
                    # rotate_b = np.eye(3)
                    self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
                    pos_counter_b += 1
            
            pos_b = [0, 0, -(l-stick_ball_l)/2]
            rotate_b = rotation_matrix_sequence([
                rotation_matrix([0,0,1],-pi/6)
            ])
            self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))

        pos_counter += 1

        # side of stick a
        for m in range(3):
            for j_a in self.side_j_list[m]:
                pos_a = [r*cos(j_a*pi/2+pi/12), r*sin(j_a*pi/2+pi/12), self.side_pos_list[m]]
                rotate_a = rotation_matrix_sequence([
                    rotation_matrix([1,0,0],pi*((m+1)%2)),
                    rotation_matrix([0,0,1],pi/12+pi*(m%2))
                ])
                for screw_theta in screw_thetas:
                    pos_counter_b = 0
                    pos_b = [0, 0, (l-stick_ball_l)/2]
                    rotate_b = rotation_matrix_sequence([
                        rotation_matrix([-sin(j_a*pi/2),cos(j_a*pi/2),0],-pi/2),
                        rotation_matrix([0,0,1],-j_a*pi/2+screw_theta),
                        rotation_matrix([0,0,1],pi/6)
                    ])
                    self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
                    pos_counter_b += 1
                    
                    if screw_theta != 0:
                        for n in range(3):
                            for j_b in self.side_j_list[n]:
                                pos_b = [r*cos(j_b*pi/2+pi/12), r*sin(j_b*pi/2+pi/12), self.side_pos_list[n]]
                                rotate_b = rotation_matrix_sequence([
                                    rotation_matrix([0,1,0],pi*(n%2)),
                                    rotation_matrix([0,0,1],j_b*pi/2-j_a*pi/2),
                                    rotation_matrix([cos(j_b*pi/2), sin(j_b*pi/2), 0], screw_theta),
                                    rotation_matrix([0,0,1],pi/12+pi*(n%2)*(m%2))
                                ])
                                self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
                                pos_counter_b += 1
                    else:
                        pos_counter_b += 6
                    
                    pos_b = [0, 0, -(l-stick_ball_l)/2]
                    rotate_b = rotation_matrix_sequence([
                        rotation_matrix([-sin(j_a*pi/2),cos(j_a*pi/2),0],-pi/2),
                        rotation_matrix([0,0,1],-j_a*pi/2+screw_theta+pi/6),
                        rotation_matrix([1,0,0],pi)
                    ])
                    self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
                    
                pos_counter += 1
        
        
        # tip of stick a
        pos_a = [0, 0, -(l-stick_ball_l)/2]
        for screw_theta in screw_thetas:
            rotate_a = rotation_matrix_sequence([
                rotation_matrix([0,0,1],screw_theta-pi/6)
            ])
            pos_counter_b = 0
            pos_b = [0, 0, (l-stick_ball_l)/2]
            rotate_b = np.eye(3)
            self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
            pos_counter_b += 1

            for m in range(3):
                for j_b in self.side_j_list[m]:
                    pos_b = [r*cos(j_b*pi/2+pi/12), r*sin(j_b*pi/2+pi/12), self.side_pos_list[m]]
                    rotate_b = rotation_matrix_sequence([
                        rotation_matrix([0,1,0],pi/2+pi*((m+1)%2)),
                        rotation_matrix([0,0,1],j_b*pi/2+pi/12+pi*((m+1)%2)),
                    ])
                    # rotate_b = np.eye(3)
                    self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))
                    pos_counter_b += 1
            
            pos_b = [0, 0, -(l-stick_ball_l)/2]
            rotate_b = rotation_matrix_sequence([
                rotation_matrix([1,0,0],pi)
            ])
            self.dock_list.append(DockingPoint(pos_a, pos_b, rotate_a, rotate_b, pos_counter, pos_counter_b))

        pos_counter += 1
    
    def __getitem__(self, index):
        # Return the element at the specified index
        return self.dock_list[index]
    


############## # Preparing for new docking management system

@dataclass
class Dock:
    """Represents a docking point between two modules."""
    position: Vector3  # Local position in parent module
    quaternion: np.ndarray  # Local position in child module
    name: str = None  # Name of the docking point



def generate_stick_docks(robot_cfg):

    l = robot_cfg["l"]
    stick_ball_l = robot_cfg["stick_ball_l"]
    screw_thetas = [0, 2*pi/3, 4*pi/3]  # screw the ball at each docking position
    r = robot_cfg["r"] - stick_ball_l
    a = robot_cfg["a"]  # l/6 stick center to the dock center on the side
    side_pos_list = [a, 0, -a]
    j_list = [1,3]
    j_c_list = [0,2]
    side_j_list = [j_list, j_c_list, j_list]

    dock_list = []

    # parent docking
    # tip of stick a
    pos_a = [0, 0, (l-stick_ball_l)/2]
    for i, screw_theta in enumerate(screw_thetas):
        rotate_a = Rotation.from_euler('z', screw_theta).as_quat()
        dock_list.append(Dock(pos_a, rotate_a, f"head_{i}"))

    # side of stick a
    for m in range(3):
        for i, j_a in enumerate(side_j_list[m]):
            pos_a = [r*cos(j_a*pi/2+pi/12), r*sin(j_a*pi/2+pi/12), side_pos_list[m]]
            rotate_a = Rotation.from_euler('xz', [pi*((m+1)%2), pi/12+pi*(m%2)]).as_quat()
            dock_list.append(Dock(pos_a, rotate_a, f"side_{m}_{i}"))

    # tip of stick a
    pos_a = [0, 0, -(l-stick_ball_l)/2]
    for i, screw_theta in enumerate(screw_thetas):
        rotate_a = Rotation.from_euler('z', screw_theta-pi/6).as_quat()
        dock_list.append(Dock(pos_a, rotate_a, f"tail_{i}"))

    return dock_list




def generate_stick_to_stick_child_connections(robot_cfg):


    l = robot_cfg["l"]
    stick_ball_l = robot_cfg["stick_ball_l"]
    screw_thetas = [0, 2*pi/3, 4*pi/3]  # screw the ball at each docking position
    r = robot_cfg["r"] - stick_ball_l
    a = robot_cfg["a"]  # l/6 stick center to the dock center on the side
    side_pos_list = [a, 0, -a]
    j_list = [1,3]
    j_c_list = [0,2]
    side_j_list = [j_list, j_c_list, j_list]

    dock_list = []

    # tip of stick a
    pos_b = [0, 0, (l-stick_ball_l)/2]
    rotate_b = Rotation.from_euler('xz', [pi, pi+pi/6]).as_matrix()
    
    dock_list.append(DockingPoint(None, pos_b, None, rotate_b, None, None))

    for m in range(3):
        for j_b in side_j_list[m]:
            pos_b = [r*cos(j_b*pi/2+pi/12), r*sin(j_b*pi/2+pi/12), side_pos_list[m]]
            rotate_b = Rotation.from_euler('yz', [-pi/2, j_b*pi/2+pi/12]).as_matrix()
            dock_list.append(DockingPoint(None, pos_b, None, rotate_b, None, None))
    
    pos_b = [0, 0, -(l-stick_ball_l)/2]
    rotate_b = Rotation.from_euler('z', -pi/6).as_matrix()
    dock_list.append(DockingPoint(None, pos_b, None, rotate_b, None, None))


    # side of stick a
    for m in range(3):
        for j_a in side_j_list[m]:
            for screw_theta in screw_thetas:
                pos_b = [0, 0, (l-stick_ball_l)/2]
                first_rot = Rotation.from_rotvec(np.array([-sin(j_a*pi/2),cos(j_a*pi/2),0])*-pi/2)
                rotate_b = (Rotation.from_euler('z', -j_a*pi/2+screw_theta+pi/6) * first_rot).as_matrix()
                dock_list.append(DockingPoint(None, pos_b, None, rotate_b, None, None))
                
                pos_b = [0, 0, -(l-stick_ball_l)/2]
                rot0 = Rotation.from_rotvec(np.array([-sin(j_a*pi/2),cos(j_a*pi/2),0])*-pi/2)
                rot1 = Rotation.from_euler('zx', [-j_a*pi/2+screw_theta+pi/6, pi])
                rotate_b = (rot1 * rot0).as_matrix()
                dock_list.append(DockingPoint(None, pos_b, None, rotate_b, None, None))

    for m in range(3):
        for j_a in side_j_list[m]:
            for screw_theta in screw_thetas:
                if screw_theta != 0:
                    for n in range(3):
                        for j_b in side_j_list[n]:
                            pos_b = [r*cos(j_b*pi/2+pi/12), r*sin(j_b*pi/2+pi/12), side_pos_list[n]]
                            rot0 = Rotation.from_euler('yz', [pi*(n%2), j_b*pi/2-j_a*pi/2])
                            rot1 = Rotation.from_rotvec(np.array([cos(j_b*pi/2), sin(j_b*pi/2), 0]) * screw_theta)
                            rot2 = Rotation.from_euler('z', pi/12+pi*(n%2)*(m%2))
                            rotate_b = (rot2*rot1*rot0).as_matrix()
                            dock_list.append(DockingPoint(None, pos_b, None, rotate_b, None, None))

    
    # tip of stick a
    for screw_theta in screw_thetas:
        pos_b = [0, 0, (l-stick_ball_l)/2]
        rotate_b = np.eye(3)
        dock_list.append(DockingPoint(None, pos_b, None, rotate_b, None, None))

        for m in range(3):
            for j_b in side_j_list[m]:
                pos_b = [r*cos(j_b*pi/2+pi/12), r*sin(j_b*pi/2+pi/12), side_pos_list[m]]
                rotate_b = Rotation.from_euler('yz', [pi/2+pi*((m+1)%2), j_b*pi/2+pi/12+pi*((m+1)%2)]).as_matrix()
                # rotate_b = np.eye(3)
                dock_list.append(DockingPoint(None, pos_b, None, rotate_b, None, None))
        
        pos_b = [0, 0, -(l-stick_ball_l)/2]
        rotate_b = Rotation.from_euler('x', pi).as_matrix()
        dock_list.append(DockingPoint(None, pos_b, None, rotate_b, None, None))
