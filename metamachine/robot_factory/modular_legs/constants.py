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

ROBOT_CFG_AIR1S = {
        "theta": 0.4625123,
        "R": 0.07,
        "r": 0.03,
        "l_": 0.236,
        "delta_l": 0, # 0,
        "stick_ball_l": 0.005,# 0, #-0.1, # ,
        "a": 0.236/4, # 0.0380409255338946, # l/6 stick center to the dock center on the side
        "stick_mass": 0.1734, #0.154,
        "top_hemi_mass": 0.1153, 
        "battery_mass": 0.122,
        "motor_mass": 0.317,
        "bottom_hemi_mass": 0.1623, #0.097, 
        "pcb_mass": 0.1
    }
MESH_DICT_FINE = {
            "up": "top_lid.obj",
            "bottom": "bottom_lid.obj",
            "stick": "leg4.4.obj",
            "battery": "battery.obj",
            "pcb": "pcb.obj",
            "motor": "motor.obj"
        }

MESH_DICT_DRAFT = {
                "up": "SPHERE",
                "bottom": "SPHERE",
                "stick": "CAPSULE",
                "battery": "NONE",
                "pcb": "NONE",
                "motor": "NONE"
            }