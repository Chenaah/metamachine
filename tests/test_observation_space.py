"""
Tests for observation space functionality.

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

import pytest
import numpy as np
from gymnasium.spaces import Box
from unittest.mock import patch

from metamachine.environments.components.state import State, RawState, DerivedState, AccurateState, ActionHistoryBuffer
from metamachine.environments.env_sim import MetaMachine
from metamachine.environments.configs.config_registry import ConfigRegistry


class TestRawState:
    """Test cases for RawState class."""
    
    def test_initialization(self):
        """Test RawState initialization."""
        state = RawState(num_dof=5)
        
        # Check initial shapes
        assert state.pos_world.shape == (3,)
        assert state.quat.shape == (4,)
        assert state.vel_body.shape == (3,)
        assert state.vel_world.shape == (3,)
        assert state.ang_vel_body.shape == (3,)
        assert state.ang_vel_world.shape == (3,)
        assert state.dof_pos.shape == (5,)
        assert state.dof_vel.shape == (5,)
        
        # Check initial values are zeros
        np.testing.assert_array_equal(state.pos_world, np.zeros(3))
        np.testing.assert_array_equal(state.quat, np.zeros(4))
        np.testing.assert_array_equal(state.dof_pos, np.zeros(5))
        np.testing.assert_array_equal(state.dof_vel, np.zeros(5))
        
    def test_update_valid_data(self):
        """Test updating RawState with valid data."""
        state = RawState(num_dof=3)
        
        update_data = {
            'pos_world': np.array([1.0, 2.0, 3.0]),
            'vel_body': np.array([0.1, 0.2, 0.3]),
            'dof_pos': np.array([0.5, -0.5, 1.0])
        }
        
        state.update(update_data)
        
        np.testing.assert_array_equal(state.pos_world, update_data['pos_world'])
        np.testing.assert_array_equal(state.vel_body, update_data['vel_body'])
        np.testing.assert_array_equal(state.dof_pos, update_data['dof_pos'])
        
    def test_update_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        state = RawState(num_dof=3)
        
        # Wrong shape for pos_world
        with pytest.raises(ValueError, match="Shape mismatch for pos_world"):
            state.update({'pos_world': np.array([1.0, 2.0])})  # Should be (3,)
            
        # Wrong shape for dof_pos
        with pytest.raises(ValueError, match="Shape mismatch for dof_pos"):
            state.update({'dof_pos': np.array([1.0, 2.0])})  # Should be (3,)


class TestActionHistoryBuffer:
    """Test cases for ActionHistoryBuffer class."""
    
    def test_initialization(self):
        """Test ActionHistoryBuffer initialization."""
        buffer = ActionHistoryBuffer(num_actions=5, history_steps=3)
        
        assert buffer.num_actions == 5
        assert buffer.history_steps == 3
        assert buffer.action_history.shape == (3, 5)
        np.testing.assert_array_equal(buffer.action_history, np.zeros((3, 5)))
        
    def test_reset_with_initial_action(self):
        """Test reset with initial action."""
        buffer = ActionHistoryBuffer(num_actions=3, history_steps=2)
        initial_action = np.array([1, 2, 3])
        
        buffer.reset(initial_action)
        
        expected = np.array([[1, 2, 3], [1, 2, 3]])
        np.testing.assert_array_equal(buffer.action_history, expected)
        
    def test_update_action_history(self):
        """Test updating action history."""
        buffer = ActionHistoryBuffer(num_actions=2, history_steps=3)
        buffer.reset()
        
        # Update with first action
        action1 = np.array([1, 2])
        buffer.update(action1)
        expected1 = np.array([[0, 0], [0, 0], [1, 2]])
        np.testing.assert_array_equal(buffer.action_history, expected1)
        
        # Update with second action
        action2 = np.array([3, 4])
        buffer.update(action2)
        expected2 = np.array([[0, 0], [1, 2], [3, 4]])
        np.testing.assert_array_equal(buffer.action_history, expected2)


class TestState:
    """Test cases for State class."""
    
    @pytest.fixture
    def basic_config(self):
        """Create a basic configuration for testing."""
        config = ConfigRegistry.create_from_name("basic_quadruped")
        
        # Override specific values for basic state testing
        config.control.num_actions = 3  # Reduced for simple tests
        config.observation.components = [
            {'name': 'projected_gravity'},
            {'name': 'dof_pos'},
            {'name': 'last_action'}
        ]
        config.observation.include_history_steps = 3
        config.observation.action_history_steps = 3
        
        return config
        
    def test_initialization(self, basic_config):
        """Test State initialization."""
        state = State(basic_config)
        
        assert state.num_act == 3
        assert state.num_envs == 1
        assert state.include_history_steps == 3
        assert state.clip_observations == 100.0
        assert state.dt == 0.05
        
        # Check state containers
        assert isinstance(state.raw, RawState)
        assert isinstance(state.derived, DerivedState)
        assert isinstance(state.accurate, AccurateState)
        assert isinstance(state.action_history, ActionHistoryBuffer)


class TestObservationSpaceIntegration:
    """Integration tests for observation space with full environment."""
    
    @pytest.fixture
    def test_config(self):
        """Create a test configuration."""
        config = ConfigRegistry.create_from_name("basic_quadruped")
        
        # Override specific values for integration testing
        config.observation.components = [
            {'name': 'projected_gravity'},
            {'name': 'ang_vel_body'},
            {'name': 'dof_pos', 'transform': 'cos'},
            {'name': 'dof_vel'},
            {'name': 'last_action'}
        ]
        config.observation.include_history_steps = 3
        
        return config
    
    @patch('metamachine.environments.base.KBHit')
    def test_observation_space_creation(self, mock_kbhit, test_config):
        """Test that observation space is created correctly."""
        env = MetaMachine(test_config)
        
        # Check observation space properties
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.dtype == np.float32
        assert len(env.observation_space.shape) == 1  # Should be 1D
        
        # Check bounds
        expected_low = -np.inf
        expected_high = np.inf
        assert np.all(env.observation_space.low == expected_low)
        assert np.all(env.observation_space.high == expected_high)
        
        env.close()
        
    @patch('metamachine.environments.base.KBHit')
    def test_observation_generation(self, mock_kbhit, test_config):
        """Test that observations are generated correctly."""
        env = MetaMachine(test_config)
        obs, info = env.reset()
        
        # Check observation properties
        assert isinstance(obs, np.ndarray)
        assert obs.dtype in [np.float32, np.float64]  # Accept both float32 and float64
        assert obs.shape == env.observation_space.shape
        
        # Convert to correct dtype for contains check if needed
        obs_check = obs.astype(env.observation_space.dtype) if obs.dtype != env.observation_space.dtype else obs
        assert env.observation_space.contains(obs_check)
        
        # Check observation is finite (no NaN or inf values)
        assert np.all(np.isfinite(obs))
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__])
