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
import numpy as np
from omegaconf import OmegaConf, ListConfig, DictConfig
from typing import Optional, Union, List, Dict, Any
from .action_filter import ActionFilterButter, ActionFilterMelter

"""
Enhanced Action Processing with Flexible Frozen Joints

This module provides a modern action processing system with advanced frozen joints support.

Key Features:
1. Automatic action space reduction: For N total joints with M frozen joints, 
   the policy only needs to output N-M actions.
2. Flexible frozen joint patterns: zero, constant, sine wave, random, noise
3. Proper action space management and bounds handling
4. Backward compatibility with legacy frozen joints configuration

Configuration Examples:

# Legacy format (simple frozen joints set to zero)
frozen_joints: [1, 3, 5]

# Advanced format with patterns
frozen_joints:
  - joint: 1
    type: sine
    amplitude: 0.5
    frequency: 2.0
    phase: 0.0
  - joint: 3
    type: constant
    value: 0.8
  - joint: 5
    type: random
    min: -0.3
    max: 0.3
    interval: 1.0  # seconds

Supported Pattern Types:
- zero: Always outputs 0.0
- constant: Outputs a fixed value
- sine: Sine wave with configurable amplitude, frequency, and phase
- random: Piecewise constant random values with configurable range and change interval
- noise: Gaussian noise with configurable amplitude
"""


class FrozenJointsProcessor:
    """Handles frozen joints with flexible behavior patterns.
    
    This class manages joints that are not controlled by the policy. Frozen joints
    can be set to various patterns: zero, constant values, sine waves, random, etc.
    It also handles action space remapping to account for the reduced degrees of freedom.
    """
    
    def __init__(self, cfg: OmegaConf, total_joints: int):
        """Initialize frozen joints processor.
        
        Args:
            cfg: Configuration object
            total_joints: Total number of joints in the robot
        """
        self.total_joints = total_joints
        self.cfg = cfg
        
        # Parse frozen joints configuration
        self._setup_frozen_joints(cfg)
        
        # Calculate effective action space size
        self.active_joints = [i for i in range(total_joints) if i not in self.frozen_indices]
        self.num_active_joints = len(self.active_joints)
        
        # Initialize pattern generators
        self._setup_pattern_generators()
        
        # State tracking
        self.step_count = 0
        self.time = 0.0
        self.dt = 0.02  # Default timestep, will be set properly in reset()
    
    def _setup_frozen_joints(self, cfg):
        """Parse and validate frozen joints configuration."""
        frozen_config = cfg.environment.get('frozen_joints', {})
        
        if frozen_config is None or (isinstance(frozen_config, (list, tuple, ListConfig)) and len(frozen_config) == 0):
            # No frozen joints
            self.frozen_indices = []
            self.frozen_patterns = {}
            return
        
        # Handle legacy format (just a list of indices)
        if isinstance(frozen_config, (list, tuple, ListConfig)) and len(frozen_config) > 0 and all(isinstance(x, int) for x in frozen_config):
            self.frozen_indices = list(frozen_config)
            self.frozen_patterns = {idx: {'type': 'zero'} for idx in self.frozen_indices}
        elif isinstance(frozen_config, (list, tuple, ListConfig)):
            # New format with patterns
            self.frozen_indices = []
            self.frozen_patterns = {}
            
            for joint_spec in frozen_config:
                if isinstance(joint_spec, int):
                    # Simple index, default to zero
                    idx = joint_spec
                    self.frozen_indices.append(idx)
                    self.frozen_patterns[idx] = {'type': 'zero'}
                elif isinstance(joint_spec, (dict, DictConfig)):
                    # Full specification with pattern
                    idx = joint_spec['joint']
                    self.frozen_indices.append(idx)
                    self.frozen_patterns[idx] = dict(joint_spec)  # Convert to regular dict
                    del self.frozen_patterns[idx]['joint']  # Remove joint index from pattern
        else:
            # Other formats or empty
            self.frozen_indices = []
            self.frozen_patterns = {}
        
        # Validate indices
        if self.frozen_indices and max(self.frozen_indices) >= self.total_joints:
            raise ValueError(f"Frozen joint index {max(self.frozen_indices)} >= total joints {self.total_joints}")
        
        if len(set(self.frozen_indices)) != len(self.frozen_indices):
            raise ValueError("Duplicate frozen joint indices found")
    
    def _setup_pattern_generators(self):
        """Setup pattern generators for each frozen joint."""
        self.pattern_state = {}
        
        for idx in self.frozen_indices:
            pattern = self.frozen_patterns[idx]
            pattern_type = pattern.get('type', 'zero')
            
            if pattern_type == 'random':
                # Initialize random state
                self.pattern_state[idx] = {
                    'last_value': 0.0,
                    'next_change_time': 0.0
                }
            elif pattern_type == 'sine':
                # Initialize sine wave state
                self.pattern_state[idx] = {
                    'phase': pattern.get('phase', 0.0)
                }
    
    def reset(self, dt: float):
        """Reset frozen joints processor.
        
        Args:
            dt: Environment timestep
        """
        self.step_count = 0
        self.time = 0.0
        self.dt = dt
        
        # Reset pattern states
        for idx in self.frozen_indices:
            pattern = self.frozen_patterns[idx]
            pattern_type = pattern.get('type', 'zero')
            
            if pattern_type == 'random':
                self.pattern_state[idx]['last_value'] = 0.0
                self.pattern_state[idx]['next_change_time'] = 0.0
    
    def map_action_to_full_space(self, reduced_action: np.ndarray) -> np.ndarray:
        """Map reduced action space to full joint space.
        
        Args:
            reduced_action: Action with size (num_active_joints,) or (batch, num_active_joints)
            
        Returns:
            Full action with size (total_joints,) or (batch, total_joints)
        """
        if len(self.frozen_indices) == 0:
            return reduced_action
        
        # Handle batch dimension
        if reduced_action.ndim == 1:
            full_action = np.zeros(self.total_joints, dtype=reduced_action.dtype)
            full_action[self.active_joints] = reduced_action
        else:
            batch_size = reduced_action.shape[0]
            full_action = np.zeros((batch_size, self.total_joints), dtype=reduced_action.dtype)
            full_action[:, self.active_joints] = reduced_action
        
        return full_action
    
    def apply_frozen_patterns(self, action: np.ndarray) -> np.ndarray:
        """Apply frozen joint patterns to the action.
        
        Args:
            action: Full action array
            
        Returns:
            Action with frozen joints set according to their patterns
        """
        if len(self.frozen_indices) == 0:
            return action
        
        action = action.copy()
        
        for idx in self.frozen_indices:
            pattern = self.frozen_patterns[idx]
            pattern_type = pattern.get('type', 'zero')
            
            value = self._generate_pattern_value(idx, pattern)
            
            if action.ndim == 1:
                action[idx] = value
            else:
                action[:, idx] = value
        
        return action
    
    def _generate_pattern_value(self, joint_idx: int, pattern: Dict[str, Any]) -> float:
        """Generate value for a specific pattern.
        
        Args:
            joint_idx: Index of the frozen joint
            pattern: Pattern configuration
            
        Returns:
            Generated value for this timestep
        """
        pattern_type = pattern.get('type', 'zero')
        
        if pattern_type == 'zero':
            return 0.0
        
        elif pattern_type == 'constant':
            return pattern.get('value', 0.0)
        
        elif pattern_type == 'sine':
            amplitude = pattern.get('amplitude', 1.0)
            frequency = pattern.get('frequency', 1.0)  # Hz
            phase = self.pattern_state[joint_idx]['phase']
            return amplitude * np.sin(2 * np.pi * frequency * self.time + phase)
        
        elif pattern_type == 'random':
            state = self.pattern_state[joint_idx]
            
            if self.time >= state['next_change_time']:
                # Generate new random value
                min_val = pattern.get('min', -1.0)
                max_val = pattern.get('max', 1.0)
                state['last_value'] = np.random.uniform(min_val, max_val)
                
                # Schedule next change
                interval = pattern.get('interval', 1.0)  # seconds
                state['next_change_time'] = self.time + interval
            
            return state['last_value']
        
        elif pattern_type == 'noise':
            amplitude = pattern.get('amplitude', 0.1)
            return np.random.normal(0, amplitude)
        
        else:
            raise ValueError(f"Unknown frozen joint pattern type: {pattern_type}")
    
    def step(self):
        """Update internal state for next timestep."""
        self.step_count += 1
        self.time = self.step_count * self.dt


class ActionProcessor:
    """Handles action processing including scaling, clipping, filtering and melting.
    
    This class processes raw actions from the policy through several stages:
    1. Action space mapping - Maps reduced action space to full joint space
    2. Scaling - Applies action scaling factor
    3. Clipping - Enforces action bounds
    4. Frozen joints - Applies patterns to frozen joints
    5. Filtering - Optional smoothing via Butterworth filter
    6. Melting - Optional action melting for specific axes
    """
    
    def __init__(self, cfg: OmegaConf):
        """Initialize action processor.
        
        Args:
            cfg: Configuration object containing environment and control parameters
        """
        self.cfg = cfg  # Store for remapping config access
        
        # Core parameters
        self.total_joints = cfg.control.num_actions  # Total robot joints
        self.num_envs = cfg.environment.num_envs
        
        # Initialize frozen joints processor
        self.frozen_processor = FrozenJointsProcessor(cfg, self.total_joints)
        
        # Effective action space size (excluding frozen joints)
        self.num_actions = self.frozen_processor.num_active_joints
        self._total_dofs = self.total_joints * self.num_envs  # Total DOFs for filtering
        
        # Action bounds and scaling
        self.scale = cfg.control.action_scale
        self.action_bounds = ActionBounds(
            limit=cfg.control.symmetric_limit,
            custom_limits=cfg.control.custom_limits
        )
        
        # Control settings
        self.control_mode = cfg.control.control_mode
        self.default_dof_pos = self._get_default_dof(cfg)
        
        # Initialize filters
        self._setup_filters(cfg)
        
        # State tracking
        self.last_action = np.zeros((self.num_envs, self.num_actions))  # Reduced action space
        self.last_action_full = np.zeros((self.num_envs, self.total_joints))  # Full joint space
        self.last_action_flat = np.zeros(self._total_dofs)
    
    def _get_default_dof(self, cfg) -> np.ndarray:
        """Get default joint positions from config."""
        default = cfg.control.default_dof_pos
        if isinstance(default, (list, np.ndarray)):
            return np.array(default)
        return default
    
    def _setup_filters(self, cfg):
        """Initialize action filters if enabled."""
        # Butterworth filter for smoothing
        self.use_filter = cfg.control.filter.enabled
        if self.use_filter:
            self.smoother = ActionFilterButter(
                sampling_rate=1/cfg.control.dt,
                num_joints=self._total_dofs,
                highcut=[cfg.control.filter.cutoff_freq]
            )
            
        # Action melting filter
        self.use_melter = cfg.control.melter.enabled
        if self.use_melter:
            self.melter = ActionFilterMelter(
                axis=cfg.control.melter.axis
            )
        else:
            self.melter = None
            
    def reset(self, state=None):
        """Reset processor state including action history and filters."""
        self.last_action.fill(0)
        self.last_action_full.fill(0)
        self.last_action_flat.fill(0)
        
        # Reset frozen joints processor
        self.frozen_processor.reset(self.cfg.control.dt)
        
        if self.use_filter:
            self.smoother.reset(init_hist=state.dof_pos if state is not None else None)
        if self.use_melter:
            self.melter.reset()
            
    def process(self, action: np.ndarray) -> np.ndarray:
        """Process raw action through the processing pipeline.
        
        Args:
            action: Raw action from policy with shape (num_active_joints,) or (batch, num_active_joints)
            
        Returns:
            Processed action ready for execution with shape (total_joints,) or (batch, total_joints)
        """
        # Store reduced action for observation
        self.last_action = action.copy()
        
        # Map to full joint space
        action_full = self.frozen_processor.map_action_to_full_space(action)
        
        # Scale and clip (only affects active joints)
        action_full = self._preprocess_action(action_full)
        
        # Apply frozen joint patterns
        action_full = self.frozen_processor.apply_frozen_patterns(action_full)
        
        # Store full action
        self.last_action_full = action_full.copy()
        self.last_action_flat = action_full.flatten()
        
        # Apply processing pipeline
        processed = self._apply_processing_pipeline(action_full)
        
        # Update frozen joints processor
        self.frozen_processor.step()
        
        return processed
    
    @property
    def effective_action_size(self) -> int:
        """Get the effective action space size (excluding frozen joints)."""
        return self.num_actions
    
    @property
    def has_frozen_joints(self) -> bool:
        """Check if there are any frozen joints."""
        return len(self.frozen_processor.frozen_indices) > 0
    
    @property
    def frozen_joint_indices(self) -> List[int]:
        """Get list of frozen joint indices."""
        return self.frozen_processor.frozen_indices.copy()
    
    @property
    def active_joint_indices(self) -> List[int]:
        """Get list of active (controllable) joint indices."""
        return self.frozen_processor.active_joints.copy()
    
    def get_action_bounds_for_active_joints(self):
        """Get action bounds adjusted for active joints only."""
        if self.action_bounds.custom_limits is not None:
            # Filter custom limits to active joints only
            active_limits = self.action_bounds.custom_limits[self.frozen_processor.active_joints]
            return ActionBounds(
                limit=self.action_bounds.symmetric_limit,
                custom_limits=active_limits
            )
        else:
            # Use symmetric limit for all active joints
            return ActionBounds(limit=self.action_bounds.symmetric_limit)

    def _preprocess_action(self, action: np.ndarray) -> np.ndarray:
        """Apply scaling and clipping to raw action."""
        # print(f"!!!Scale: {self.scale}", file=sys.stderr)
        action = action * self.scale
        return self.action_bounds.clip(action)
    
    def _apply_processing_pipeline(self, action: np.ndarray) -> np.ndarray:
        """Apply the full action processing pipeline based on control mode."""
        # Apply remapping if configured
        if self.cfg.control.remapping.enabled:
            action = self._remap_action(action)
            
        # Process based on control mode
        if self.control_mode == "position":
            # print(f"!!!action: {action}", file=sys.stderr)
            action = action + self.default_dof_pos
            # print(f"!!!action + self.default_dof_pos: {action}", file=sys.stderr)
            if self.use_filter:
                # Flatten for filtering and reshape back
                original_shape = action.shape
                action_flat = action.flatten()
                # print(f"!!!action_flat: {action_flat}", file=sys.stderr)
                action_flat = self.smoother.filter(action_flat)
                # print(f"!!!action_flat after filter: {action_flat}", file=sys.stderr)
                action = action_flat.reshape(original_shape)
                # print(f"!!!action after reshape: {action}", file=sys.stderr)
            if self.use_melter:
                action = self.melter.filter(action)
                
        elif self.control_mode == "incremental":
            if self.use_filter:
                # Flatten for filtering and reshape back
                original_shape = action.shape
                action_flat = action.flatten()
                action_flat = self.smoother.filter(action_flat)
                action = action_flat.reshape(original_shape)
                
        elif self.control_mode == "velocity":
            pass  # No additional processing needed
            
        elif self.control_mode == "advanced":
            pass  # Additional params handled in environment
            
        else:
            raise ValueError(f"Unknown control mode: {self.control_mode}")
            
        return action
    
    def _remap_action(self, action: np.ndarray) -> np.ndarray:
        """Apply action remapping if configured."""
        remap_type = self.cfg.control.remapping.type
        if remap_type is None:
            return action
        else:
            raise NotImplementedError(f"Action remapping type '{remap_type}' not implemented")


class ActionBounds:
    """Handles action bounds and clipping logic.
    
    This class supports two types of bounds:
    1. Symmetric limit: A single value applied to all actions symmetrically
       Example: limit=1.0 clips all actions to [-1.0, 1.0]
       
    2. Custom limits: Different limits for each action dimension
       Example: limits=[1.0, 0.5, 2.0] clips actions to:
       - Action 0: [-1.0, 1.0]
       - Action 1: [-0.5, 0.5]
       - Action 2: [-2.0, 2.0]
    """
    
    def __init__(self, limit: float, custom_limits: Optional[Union[List[float], np.ndarray]] = None):
        """Initialize action bounds.
        
        Args:
            limit: Global symmetric limit for all actions (used if custom_limits is None)
            custom_limits: Optional per-dimension limits. If provided, overrides the symmetric limit
                         and applies different bounds to each action dimension.
        """
        self.symmetric_limit = limit
        if custom_limits is not None:
            self.custom_limits = np.array(custom_limits, dtype=np.float32)
            if len(self.custom_limits.shape) != 1:
                raise ValueError("custom_limits must be a 1D array/list of limits")
        else:
            self.custom_limits = None
        
    def clip(self, action: np.ndarray) -> np.ndarray:
        """Clip action to bounds.
        
        Args:
            action: Action to clip, shape (num_actions,) or (batch_size, num_actions)
            
        Returns:
            Clipped action with same shape as input
            
        Examples:
            >>> bounds = ActionBounds(limit=1.0)
            >>> bounds.clip(np.array([2.0, -1.5, 0.5]))
            array([ 1.0, -1.0,  0.5])
            
            >>> bounds = ActionBounds(custom_limits=[1.0, 0.5, 2.0])
            >>> bounds.clip(np.array([1.5, 0.8, 1.5]))
            array([ 1.0,  0.5,  1.5])
        """
        if self.custom_limits is None:
            # Apply same symmetric bounds to all actions
            return np.clip(action, -self.symmetric_limit, self.symmetric_limit)
        else:
            # Apply different bounds to each action dimension
            return np.clip(action, -self.custom_limits, self.custom_limits) 

def run_example():
    """Example usage of ActionProcessor with enhanced frozen joints."""
    import numpy as np
    from metamachine.environments.configs.config_registry import ConfigRegistry
    
    # Load basic environment config
    cfg = ConfigRegistry.create_from_name("basic_env")
    
    # Modify some settings for the example
    cfg.environment.num_envs = 1
    cfg.control.num_actions = 6  # 6 DOF robot
    cfg.control.filter.enabled = True
    cfg.control.melter.enabled = False
    cfg.control.melter.axis = 0
    cfg.control.default_dof_pos = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]  # Default joint positions
    cfg.control.custom_limits = [1.0, 0.5, 2.0, 1.5, 1.0, 0.8]  # Different limits per joint
    cfg.control.dt = 0.02  # 50 Hz
    
    # Example 1: Simple frozen joints (legacy format)
    print("Example 1: Simple Frozen Joints")
    cfg.environment.frozen_joints = [1, 4]  # Freeze joints 1 and 4
    processor = ActionProcessor(cfg)
    
    print(f"Total joints: {processor.total_joints}")
    print(f"Effective action size: {processor.effective_action_size}")
    print(f"Frozen joints: {processor.frozen_joint_indices}")
    print(f"Active joints: {processor.active_joint_indices}")
    
    # Raw action from policy (only for active joints)
    raw_action = np.array([0.5, 0.3, -0.2, 0.1])  # 4 active joints
    processed = processor.process(raw_action)
    print(f"Raw action (active joints): {raw_action}")
    print(f"Processed action (all joints): {processed}")
    print()
    
    # Example 2: Advanced frozen joints with patterns
    print("Example 2: Advanced Frozen Joints with Patterns")
    cfg.environment.frozen_joints = [
        {'joint': 1, 'type': 'sine', 'amplitude': 0.5, 'frequency': 2.0, 'phase': 0.0},
        {'joint': 3, 'type': 'constant', 'value': 0.8},
        {'joint': 5, 'type': 'random', 'min': -0.3, 'max': 0.3, 'interval': 1.0}
    ]
    processor2 = ActionProcessor(cfg)
    
    print(f"Total joints: {processor2.total_joints}")
    print(f"Effective action size: {processor2.effective_action_size}")
    print(f"Frozen joints: {processor2.frozen_joint_indices}")
    print(f"Active joints: {processor2.active_joint_indices}")
    
    # Simulate multiple timesteps to show patterns
    raw_action = np.array([0.2, 0.1, -0.3])  # 3 active joints (0, 2, 4)
    print(f"\nRaw action (active joints): {raw_action}")
    print("Time evolution of frozen joint patterns:")
    print("Step | Joint 1 (sine) | Joint 3 (const) | Joint 5 (random)")
    
    for step in range(10):
        processed = processor2.process(raw_action)
        print(f"{step:4d} | {processed[1]:11.3f} | {processed[3]:12.3f} | {processed[5]:12.3f}")
    print()
    
    # Example 3: Different pattern types
    print("Example 3: All Pattern Types")
    cfg.environment.frozen_joints = [
        {'joint': 0, 'type': 'zero'},
        {'joint': 1, 'type': 'constant', 'value': 1.0},
        {'joint': 2, 'type': 'sine', 'amplitude': 0.8, 'frequency': 1.0},
        {'joint': 3, 'type': 'random', 'min': -0.5, 'max': 0.5, 'interval': 0.5},
        {'joint': 4, 'type': 'noise', 'amplitude': 0.1}
    ]
    processor3 = ActionProcessor(cfg)
    
    print(f"Active joints: {processor3.active_joint_indices}")  # Should be [5]
    print(f"Effective action size: {processor3.effective_action_size}")  # Should be 1
    
    raw_action = np.array([0.5])  # Only 1 active joint
    print(f"\nRaw action: {raw_action}")
    print("Pattern demonstration:")
    print("Step | Zero | Const | Sine  | Random | Noise | Active")
    
    for step in range(15):
        processed = processor3.process(raw_action)
        print(f"{step:4d} | {processed[0]:4.1f} | {processed[1]:5.1f} | {processed[2]:5.2f} | {processed[3]:6.2f} | {processed[4]:5.2f} | {processed[5]:6.2f}")
    
    # Example 4: Action bounds for active joints
    print(f"\nExample 4: Action Bounds")
    bounds = processor3.get_action_bounds_for_active_joints()
    test_action = np.array([2.0])  # Will be clipped
    clipped = bounds.clip(test_action)
    print(f"Original action: {test_action}")
    print(f"Clipped action: {clipped}")

if __name__ == "__main__":
    run_example()