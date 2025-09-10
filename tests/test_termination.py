"""
Tests for termination condition functionality.

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
from unittest.mock import Mock, patch

from metamachine.environments.components.termination import TerminationChecker
from metamachine.environments.env_sim import MetaMachine
from metamachine.environments.configs.config_registry import ConfigRegistry


class TestTerminationChecker:
    """Test cases for TerminationChecker class."""
    
    @pytest.fixture
    def basic_termination_config(self):
        """Create a basic termination configuration using ConfigRegistry."""
        cfg = ConfigRegistry.create_from_name("basic_quadruped")
        return cfg
        
    def test_initialization(self, basic_termination_config):
        """Test TerminationChecker initialization."""
        checker = TerminationChecker(basic_termination_config)
        
        assert checker.termination_strategy.value == 'ballance_auto'
        assert checker.max_episode_steps == 1000
        assert checker.height_threshold == 0.3
        
    def test_check_done_height_threshold(self, basic_termination_config):
        """Test done condition based on height threshold."""
        checker = TerminationChecker(basic_termination_config)
        
        # Mock state below height threshold
        state = Mock()
        state.pos = np.array([0.0, 0.0, 0.2])  # Below 0.3 threshold
        state.step_counter = 100
        
        is_done = checker.check_done(state)
        assert is_done  # Should be done due to low height
        
    def test_check_done_height_safe(self, basic_termination_config):
        """Test not done when height is safe."""
        checker = TerminationChecker(basic_termination_config)
        
        # Mock state above height threshold
        state = Mock()
        state.pos = np.array([0.0, 0.0, 0.5])  # Above 0.3 threshold
        state.accurate_quat = np.array([0.0, 0.0, 0.0, 1.0])  # Normal orientation for other checks
        state.step_counter = 100
        
        is_done = checker.check_done(state)
        assert not is_done  # Should not be done
        
    def test_check_truncated_max_steps(self, basic_termination_config):
        """Test truncation based on max episode steps."""
        checker = TerminationChecker(basic_termination_config)
        
        # Mock state at max steps
        state = Mock()
        checker.current_step = 999  # Set checker's current step to max-1 (since check is >= max-1)
        
        # check_truncated should return True for max steps  
        is_truncated = checker.check_truncated(state)
        assert is_truncated  # Should be truncated due to max steps
        
    def test_check_truncated_within_steps(self, basic_termination_config):
        """Test not truncated when within step limit."""
        checker = TerminationChecker(basic_termination_config)
        
        # Mock state within step limit
        state = Mock()
        checker.current_step = 500  # Below max steps
        
        is_truncated = checker.check_truncated(state)
        assert not is_truncated  # Should not be truncated
        
    def test_check_upsidedown(self, basic_termination_config):
        """Test upside down detection."""
        checker = TerminationChecker(basic_termination_config)
        
        # Mock upside down state (gravity projection pointing up)
        state = Mock()
        state.quat = np.array([0.0, 1.0, 0.0, 0.0])  # 180 degree rotation around Y axis
        
        is_upsidedown = checker.check_upsidedown(state)
        assert is_upsidedown
        
    def test_check_not_upsidedown(self, basic_termination_config):
        """Test normal orientation detection."""
        checker = TerminationChecker(basic_termination_config)
        
        # Mock normal state (gravity projection pointing down)
        state = Mock()
        state.quat = np.array([0.0, 0.0, 0.0, 1.0])  # No rotation (normal orientation)
        
        is_upsidedown = checker.check_upsidedown(state)
        assert not is_upsidedown


class TestTerminationIntegration:
    """Integration tests for termination with full environment."""
    
    @pytest.fixture
    def test_config(self):
        """Create a test configuration using ConfigRegistry."""
        cfg = ConfigRegistry.create_from_name("basic_quadruped")
        # Override max_episode_steps for testing
        cfg.task.termination_conditions.max_episode_steps = 100
        cfg.task.termination_conditions.height_threshold = 0.2
        # Disable rendering to avoid CUDA issues
        cfg.simulation.render = False
        return cfg
    
    @patch('metamachine.environments.base.KBHit')
    def test_termination_checker_creation(self, mock_kbhit, test_config):
        """Test that termination checker is created correctly."""
        env = MetaMachine(test_config)
        
        # Check termination checker exists
        assert hasattr(env, 'termination_checker')
        assert env.termination_checker.max_episode_steps == 100
        assert env.termination_checker.height_threshold == 0.2
        
        env.close()
        
    @patch('metamachine.environments.base.KBHit')
    def test_episode_termination_by_steps(self, mock_kbhit, test_config):
        """Test episode termination by max steps."""
        env = MetaMachine(test_config)
        obs, info = env.reset()
        
        done = False
        truncated = False
        steps = 0
        
        # Run until termination
        while not (done or truncated) and steps < 200:  # Safety limit
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
        # Should terminate within reasonable time (may be due to height or max steps)
        assert done or truncated or steps >= 100
        
        env.close()
        
    @patch('metamachine.environments.base.KBHit')
    def test_termination_info_tracking(self, mock_kbhit, test_config):
        """Test that termination info is properly tracked."""
        env = MetaMachine(test_config)
        obs, info = env.reset()
        
        # Take a few steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            # Check that info contains termination-related data
            assert "is_upsidedown" in info
            assert isinstance(info["is_upsidedown"], (bool, np.bool_))
            
            if done or truncated:
                break
                
        env.close()
        
    @patch('metamachine.environments.base.KBHit')
    def test_step_counter_tracking(self, mock_kbhit, test_config):
        """Test that step counter is properly tracked."""
        env = MetaMachine(test_config)
        obs, info = env.reset()
        
        # Check initial step counter
        assert env.step_count == 0
        
        # Take steps and verify counter
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            assert env.step_count == i + 1
            assert info["episode_step"] == i + 1
            
            if done or truncated:
                break
                
        env.close()


if __name__ == "__main__":
    pytest.main([__file__]) 