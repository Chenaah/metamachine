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

from rich.table import Table
from rich.live import Live
from omegaconf import OmegaConf
import numpy as np

class Visualizer:
    """Handles visualization and logging of environment state."""
    
    def __init__(self, cfg: OmegaConf):
        """Initialize visualizer.
        
        Args:
            cfg: Configuration object
        """
        self.cfg = cfg
        self.info_dict = {}
        self.status_dict = {}
        
        # Initialize live display
        self.live = Live(self._generate_table(), refresh_per_second=20)
        self.live.__enter__()
        
    def update(self, state):
        """Update visualization with current state.
        
        Args:
            state: Current environment state
        """
        self._log_info(state)
        self._diagnose()
        self.live.update(self._generate_table())
        
    def _log_info(self, state):
        """Log current state information."""
        self.info_dict = {
            "robot": f"{self.cfg.robot.mode} - {state.num_act} modules",
            "last_action": state.last_action,
            "commands": state.commands
        }
        
        # Add all observable data
        for k, v in state.observable_data.items():
            self.info_dict[k] = v
            
    def _diagnose(self):
        """Diagnose system state and update status."""
        # Set default status for all info items
        self.status_dict = dict.fromkeys(self.info_dict, "[green]NORMAL")
        
        # Add specific diagnostics here
        # TODO: Add hardware-specific diagnostics
        
    def _generate_table(self) -> Table:
        """Generate rich table for display.
        
        Returns:
            Table: Rich table object
        """
        table = Table()
        table.add_column("Parameter")
        table.add_column("Value")
        table.add_column("Status")
        
        # Define which keys should end a section
        end_keys = ["Enable", "Motor Torque"]
        
        # Add rows to table
        for key, value in self.info_dict.items():
            if isinstance(value, float):
                value = f"{value:3.3f}"
            elif isinstance(value, np.ndarray):
                with np.printoptions(precision=3, suppress=True):
                    value = str(value)
                    
            table.add_row(
                key,
                str(value),
                self.status_dict[key],
                end_section=(key in end_keys)
            )
            
        return table
    
    def close(self):
        """Clean up visualization resources."""
        self.live.__exit__(None, None, None) 