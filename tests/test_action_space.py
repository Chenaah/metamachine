"""
Tests for action space functionality.

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

from metamachine.environments.components.action import ActionProcessor, ActionBounds
from metamachine.environments.env_sim import MetaMachine
from metamachine.environments.configs.config_registry import ConfigRegistry


class TestActionBounds:
    """Test cases for ActionBounds class."""
    
    def test_symmetric_limit_initialization(self):
        """Test ActionBounds initialization with symmetric limits."""
        bounds = ActionBounds(limit=1.0)
        assert bounds.symmetric_limit == 1.0
        assert bounds.custom_limits is None
        
    def test_custom_limits_initialization(self):
        """Test ActionBounds initialization with custom limits."""
        custom_limits = [1.0, 0.5, 2.0]
        bounds = ActionBounds(limit=1.0, custom_limits=custom_limits)
        assert bounds.symmetric_limit == 1.0
        np.testing.assert_array_equal(bounds.custom_limits, custom_limits)
        
    def test_invalid_custom_limits(self):
        """Test that invalid custom limits raise ValueError."""
        with pytest.raises(ValueError, match="custom_limits must be a 1D array"):
            ActionBounds(limit=1.0, custom_limits=[[1.0, 0.5], [2.0, 1.5]])
            
    def test_symmetric_clipping(self):
        """Test action clipping with symmetric limits."""
        bounds = ActionBounds(limit=1.0)
        
        # Test 1D array
        action = np.array([1.5, -1.5, 0.5])
        clipped = bounds.clip(action)
        expected = np.array([1.0, -1.0, 0.5])
        np.testing.assert_array_equal(clipped, expected)
        
        # Test 2D array (multiple environments)
        action = np.array([[1.5, -1.5], [0.5, -0.5]])
        clipped = bounds.clip(action)
        expected = np.array([[1.0, -1.0], [0.5, -0.5]])
        np.testing.assert_array_equal(clipped, expected)
        
    def test_custom_limits_clipping(self):
        """Test action clipping with custom limits."""
        custom_limits = [1.0, 0.5, 2.0]
        bounds = ActionBounds(limit=1.0, custom_limits=custom_limits)
        
        # Test 1D array
        action = np.array([1.5, 0.8, -3.0])
        clipped = bounds.clip(action)
        expected = np.array([1.0, 0.5, -2.0])
        np.testing.assert_array_equal(clipped, expected)
        
        # Test shape mismatch
        action = np.array([1.5, 0.8])  # Only 2 elements, but custom_limits has 3
        with pytest.raises(ValueError, match="operands could not be broadcast together"):
            bounds.clip(action)


class TestActionProcessor:
    """Test cases for ActionProcessor class."""
    
    @pytest.fixture
    def basic_config(self):
        """Create a basic configuration for testing using ConfigRegistry."""
        cfg = ConfigRegistry.create_from_name("basic_quadruped")
        # Override some values for testing
        cfg.control.num_actions = 3
        cfg.control.default_dof_pos = [0, -0.5, 0.5]
        cfg.control.filter.enabled = False  # Disable filter for basic tests
        return cfg
        
    @pytest.fixture
    def filter_config(self):
        """Create a configuration with filtering enabled."""
        cfg = ConfigRegistry.create_from_name("basic_quadruped")
        # Override some values for testing
        cfg.control.num_actions = 3
        cfg.control.default_dof_pos = [0, -0.5, 0.5]
        cfg.control.filter.enabled = True
        return cfg
        
    def test_initialization(self, basic_config):
        """Test ActionProcessor initialization."""
        processor = ActionProcessor(basic_config)
        
        assert processor.num_actions == 3  # Overridden in fixture
        assert processor.num_envs == 1
        assert processor.scale == 1.0
        assert processor.control_mode == 'position'
        assert not processor.use_filter  # Disabled in basic_config fixture
        assert not processor.use_melter
        
    def test_action_scaling(self, basic_config):
        """Test action scaling."""
        basic_config.control.action_scale = 2.0
        basic_config.control.symmetric_limit = 1000
        processor = ActionProcessor(basic_config)
        
        action = np.array([0.5, -0.5, 0.0])
        processed = processor.process(action)

        
        # Should be scaled by 2.0 and added to default positions
        expected = np.array([1.0, -1.0, 0.0]) + np.array(basic_config.control.default_dof_pos)
        np.testing.assert_array_equal(processed, expected)
        
    def test_action_clipping(self, basic_config):
        """Test action clipping."""
        basic_config.control.symmetric_limit = 0.5
        processor = ActionProcessor(basic_config)
        
        action = np.array([1.0, -1.0, 0.25])
        processed = processor.process(action)
        
        # Should be clipped to [-0.5, 0.5] then added to default positions
        expected = np.array([0.5, -0.5, 0.25]) + np.array(basic_config.control.default_dof_pos)
        np.testing.assert_array_equal(processed, expected)
        
    def test_position_control_mode(self, basic_config):
        """Test position control mode processing."""
        basic_config.control.default_dof_pos = [1.0, -1.0, 0.5]
        processor = ActionProcessor(basic_config)
        
        action = np.array([0.1, 0.2, -0.1])
        processed = processor.process(action)
        
        # Should add to default positions
        expected = np.array([1.1, -0.8, 0.4])
        np.testing.assert_array_equal(processed, expected)
        
    def test_incremental_control_mode(self, basic_config):
        """Test incremental control mode processing."""
        basic_config.control.control_mode = 'incremental'
        processor = ActionProcessor(basic_config)
        
        action = np.array([0.1, 0.2, -0.1])
        processed = processor.process(action)
        
        # Should not add to default positions in incremental mode
        np.testing.assert_array_equal(processed, action)
        
    def test_velocity_control_mode(self, basic_config):
        """Test velocity control mode processing."""
        basic_config.control.control_mode = 'velocity'
        processor = ActionProcessor(basic_config)
        
        action = np.array([0.1, 0.2, -0.1])
        processed = processor.process(action)
        
        # Should not modify action in velocity mode
        np.testing.assert_array_equal(processed, action)
        
    def test_frozen_joints(self, basic_config):
        """Test frozen joints functionality."""
        basic_config.environment.frozen_joints = [0, 2]  # Freeze first and third joints
        processor = ActionProcessor(basic_config)
        
        # With joints 0 and 2 frozen, only joint 1 is active - so action size should be 1
        assert processor.effective_action_size == 1
        assert processor.active_joint_indices == [1]
        assert processor.frozen_joint_indices == [0, 2]
        
        # Action should only have 1 element (for the active joint 1)
        action = np.array([0.2])  # Only one value for the active joint
        processed = processor.process(action)
        
        # Expected: frozen joints (0,2) set to zero, active joint (1) gets the action value
        # Then add default_dof_pos: [0, -1, 1] from basic_quadruped config
        expected = np.array([0.0, 0.2, 0.0]) + np.array(basic_config.control.default_dof_pos)
        np.testing.assert_array_equal(processed, expected)
        
    def test_reset(self, basic_config):
        """Test processor reset functionality."""
        processor = ActionProcessor(basic_config)
        
        # Process some actions to change internal state
        action = np.array([0.1, 0.2, -0.1])
        processor.process(action)
        
        # Verify state has changed
        assert not np.allclose(processor.last_action, 0)
        assert not np.allclose(processor.last_action_flat, 0)
        
        # Reset and verify state is cleared
        processor.reset()
        # last_action shape depends on the last processed action shape
        np.testing.assert_array_equal(processor.last_action, np.zeros(3))
        np.testing.assert_array_equal(processor.last_action_flat, np.zeros(3))
        
    def test_last_action_tracking(self, basic_config):
        """Test that last action is properly tracked."""
        processor = ActionProcessor(basic_config)
        
        action = np.array([0.1, 0.2, -0.1])
        processor.process(action)
        
        # Check that scaled/clipped action is stored (before position addition)
        np.testing.assert_array_equal(processor.last_action, action)
        np.testing.assert_array_equal(processor.last_action_flat, action)
        
    def test_invalid_control_mode(self, basic_config):
        """Test that invalid control mode raises ValueError."""
        basic_config.control.control_mode = 'invalid_mode'
        processor = ActionProcessor(basic_config)
        
        action = np.array([0.1, 0.2, -0.1])
        with pytest.raises(ValueError, match="Unknown control mode"):
            processor.process(action)
            
    def test_multi_environment_processing(self, basic_config):
        """Test action processing with multiple environments."""
        basic_config.environment.num_envs = 2
        processor = ActionProcessor(basic_config)
        
        # 2D action for multiple environments
        action = np.array([[0.1, 0.2, -0.1], [0.5, -0.5, 0.0]])
        processed = processor.process(action)
        
        # Should add default positions to each environment's actions
        expected = action + np.array(basic_config.control.default_dof_pos)
        np.testing.assert_array_equal(processed, expected)


    def test_processing_pipeline(self, basic_config):
        """Test position control mode processing."""
        basic_config.control.default_dof_pos = [0, -0.5, 0.5]
        basic_config.control.symmetric_limit = 0.8
        basic_config.control.action_scale = 1.0
        basic_config.control.control_mode = 'position'
        basic_config.control.filter.enabled = True
        basic_config.control.filter.cutoff_freq = 3.0

        processor = ActionProcessor(basic_config)
        processor.smoother.reset(init_hist=np.array([0.29616328, -0.62756255, 0.38721781]))
        
        action = np.array([-0.35130492, 0.5923216, 0.10904181])
        processed = processor.process(action)
        
        # Should add to default positions
        expected = np.array([0.21127603, -0.5331811, 0.41630037])
        np.testing.assert_allclose(processed, expected, rtol=1e-7, atol=1e-9)

        # Step2
        action = np.array([0.42552537, 0.20097286, -0.65422064])
        processed = processor.process(action)
        # Should add to default positions
        expected = np.array([0.07987122, -0.32514913, 0.39614447])
        np.testing.assert_allclose(processed, expected, rtol=1e-7, atol=1e-9)


class TestActionSpaceIntegration:
    """Integration tests for action space with full environment."""
    
    @pytest.fixture
    def test_config(self):
        """Create a test configuration using ConfigRegistry."""
        cfg = ConfigRegistry.create_from_name("basic_quadruped")
        # Disable rendering to avoid CUDA issues
        cfg.simulation.render = False
        return cfg
    
    @patch('metamachine.environments.base.KBHit')
    def test_action_space_creation(self, mock_kbhit, test_config):
        """Test that action space is created correctly."""
        env = MetaMachine(test_config)
        
        # Check action space properties
        assert isinstance(env.action_space, Box)
        assert env.action_space.shape == (5,)  # num_actions from basic_quadruped config
        assert env.action_space.dtype == np.float32
        
        # Check bounds
        expected_low = -0.8  # symmetric_limit from basic_quadruped config
        expected_high = 0.8
        np.testing.assert_allclose(env.action_space.low, np.full(5, expected_low, dtype=np.float32))
        np.testing.assert_allclose(env.action_space.high, np.full(5, expected_high, dtype=np.float32))
        
        env.close()
        
    @patch('metamachine.environments.base.KBHit')
    def test_action_space_sampling(self, mock_kbhit, test_config):
        """Test that action space sampling works correctly."""
        env = MetaMachine(test_config)
        
        # Sample multiple actions
        for _ in range(10):
            action = env.action_space.sample()
            
            # Check shape and bounds
            assert action.shape == (5,)
            assert np.all(action >= -0.8)
            assert np.all(action <= 0.8)
            assert action.dtype == np.float32
            
        env.close()
        
    @patch('metamachine.environments.base.KBHit')
    def test_action_space_contains(self, mock_kbhit, test_config):
        """Test action space contains method."""
        env = MetaMachine(test_config)
        
        # Valid actions
        valid_action = np.array([0.0, 0.5, -0.5, 0.8, -0.8], dtype=np.float32)
        assert env.action_space.contains(valid_action)
        
        # Invalid actions (out of bounds)
        invalid_action = np.array([0.0, 0.5, -0.5, 1.0, -0.8])  # 1.0 > 0.8
        assert not env.action_space.contains(invalid_action)
        
        # Invalid shape
        invalid_shape = np.array([0.0, 0.5, -0.5])  # Wrong shape
        assert not env.action_space.contains(invalid_shape)
        
        env.close()
        
    @patch('metamachine.environments.base.KBHit')
    def test_action_processing_integration(self, mock_kbhit, test_config):
        """Test that actions are processed correctly in environment step."""
        env = MetaMachine(test_config)
        env.reset()
        
        # Test action within bounds
        action = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
        obs, reward, done, truncated, info = env.step(action)
        
        # Check that step completed successfully
        assert obs is not None
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(done, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        
        # Check that last action is tracked
        np.testing.assert_array_equal(env.action_processor.last_action, action)
        
        env.close()
        
    @patch('metamachine.environments.base.KBHit')
    def test_action_bounds_enforcement(self, mock_kbhit, test_config):
        """Test that action bounds are enforced during environment step."""
        env = MetaMachine(test_config)
        env.reset()
        
        # Test action outside bounds - should be clipped
        action = np.array([1.5, -1.5, 0.0, 0.0, 0.0])  # Outside [-0.8, 0.8]
        obs, reward, done, truncated, info = env.step(action)
        
        # Check that action was clipped (before adding default positions)
        expected_clipped = np.array([0.8, -0.8, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(env.action_processor.last_action_full, expected_clipped)
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__]) 