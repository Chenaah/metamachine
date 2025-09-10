"""
Tests for reward system functionality.

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

from metamachine.environments.components.reward import (
    RewardCalculator, RewardComponent, LinearVelocityTrackingComponent,
    ContactFlightTimeComponent,
    DOFVelocityPenaltyComponent, DOFAccelerationPenaltyComponent,
    ContactPenaltyComponent, ActionRateComponent
)
from metamachine.environments.env_sim import MetaMachine
from metamachine.environments.configs.config_registry import ConfigRegistry


class TestRewardComponent:
    """Test cases for base RewardComponent class."""
    
    def test_initialization(self):
        """Test RewardComponent initialization."""
        # Create a concrete subclass for testing
        class TestComponent(RewardComponent):
            def calculate(self, state, calculator):
                return 1.0
                
        component = TestComponent("test_component", weight=2.0, param1=10.0)
        
        assert component.name == "test_component"
        assert component.weight == 2.0
        assert component.params == {"param1": 10.0}
        
    def test_reset_default(self):
        """Test default reset method does nothing."""
        class TestComponent(RewardComponent):
            def calculate(self, state, calculator):
                return 1.0
                
        component = TestComponent("test", weight=1.0)
        # Should not raise any errors
        component.reset()


class TestLinearVelocityTrackingComponent:
    """Test cases for LinearVelocityTrackingComponent."""
    
    def test_initialization(self):
        """Test component initialization."""
        component = LinearVelocityTrackingComponent(
            "velocity_tracking",
            weight=1.0,
            target_velocity=2.0,
            tracking_sigma=0.1
        )
        
        assert component.name == "velocity_tracking"
        assert component.weight == 1.0
        assert component.params["target_velocity"] == 2.0
        assert component.params["tracking_sigma"] == 0.1
        
    def test_calculate_perfect_tracking(self):
        """Test reward calculation with perfect velocity tracking."""
        component = LinearVelocityTrackingComponent(
            "velocity_tracking",
            weight=1.0,
            target_velocity=1.0,
            tracking_sigma=0.15
        )
        
        # Mock state and calculator
        state = Mock()
        state.accurate_vel_body = np.array([1.0, 0.0, 0.0])
        
        calculator = Mock()
        calculator.projected_forward_vec = np.array([1.0, 0.0, 0.0])
        
        reward = component.calculate(state, calculator)
        
        # Perfect tracking should give reward close to 1.0
        assert reward == pytest.approx(1.0, abs=1e-6)
        
    def test_calculate_poor_tracking(self):
        """Test reward calculation with poor velocity tracking."""
        component = LinearVelocityTrackingComponent(
            "velocity_tracking",
            weight=1.0,
            target_velocity=1.0,
            tracking_sigma=0.15
        )
        
        # Mock state and calculator with very different velocity
        state = Mock()
        state.accurate_vel_body = np.array([5.0, 0.0, 0.0])  # Much higher than target
        
        calculator = Mock()
        calculator.projected_forward_vec = np.array([1.0, 0.0, 0.0])
        
        reward = component.calculate(state, calculator)
        
        # Poor tracking should give very small reward
        assert reward < 0.1


class TestContactFlightTimeComponent:
    """Test cases for ContactFlightTimeComponent."""
    
    def test_initialization(self):
        """Test component initialization."""
        component = ContactFlightTimeComponent(
            "flight_time",
            weight=1.0,
            allowed_num_contacts=1
        )
        
        assert component.name == "flight_time"
        assert component.contact_counter == {}
        
    def test_calculate_no_contacts(self):
        """Test calculation when no contacts exist."""
        component = ContactFlightTimeComponent(
            "flight_time",
            weight=1.0,
            allowed_num_contacts=1
        )
        
        # Initialize contact counter with some history
        component.contact_counter = {0: 10, 1: 5}
        
        state = Mock()
        state.contact_floor_socks = []
        state.contact_floor_balls = []
        
        calculator = Mock()
        calculator.dt = 0.05
        
        reward = component.calculate(state, calculator)
        
        # Should increment counters and return sum of air times
        expected_reward = (11 + 6) * 0.05  # (10+1 + 5+1) * dt
        assert reward == pytest.approx(expected_reward)
        
    def test_reset_functionality(self):
        """Test reset clears contact counter."""
        component = ContactFlightTimeComponent("flight_time", weight=1.0)
        component.contact_counter = {0: 10, 1: 5}
        
        component.reset()
        
        assert component.contact_counter == {}


class TestDOFVelocityPenaltyComponent:
    """Test cases for DOFVelocityPenaltyComponent."""
    
    def test_calculate_within_limits(self):
        """Test calculation when velocities are within limits."""
        component = DOFVelocityPenaltyComponent(
            "velocity_penalty",
            weight=1.0,
            velocity_limit=10.0
        )
        
        state = Mock()
        state.dof_vel = np.array([5.0, -5.0, 0.0])  # All within ±10.0
        
        calculator = Mock()
        
        reward = component.calculate(state, calculator)
        
        # No penalty when within limits
        assert reward == 0.0
        
    def test_calculate_exceeding_limits(self):
        """Test calculation when velocities exceed limits."""
        component = DOFVelocityPenaltyComponent(
            "velocity_penalty",
            weight=1.0,
            velocity_limit=5.0
        )
        
        state = Mock()
        state.dof_vel = np.array([8.0, -10.0, 2.0])  # 8 and 10 exceed limit of 5
        
        calculator = Mock()
        
        reward = component.calculate(state, calculator)
        
        # Should be negative penalty for exceeded velocities
        expected_penalty = -((8.0 - 5.0) + (10.0 - 5.0))  # -(3 + 5)
        assert reward == pytest.approx(expected_penalty)


class TestDOFAccelerationPenaltyComponent:
    """Test cases for DOFAccelerationPenaltyComponent."""
    
    def test_calculate_first_step(self):
        """Test calculation on first step (no previous velocity)."""
        component = DOFAccelerationPenaltyComponent("accel_penalty", weight=1.0)
        
        state = Mock()
        state.dof_vel = np.array([1.0, 2.0, 3.0])
        
        calculator = Mock()
        calculator.dt = 0.05
        
        reward = component.calculate(state, calculator)
        
        # First step should return 0
        assert reward == 0.0
        assert np.array_equal(component.last_dof_vel, np.array([1.0, 2.0, 3.0]))
        
    def test_calculate_with_acceleration(self):
        """Test calculation with actual acceleration."""
        component = DOFAccelerationPenaltyComponent("accel_penalty", weight=1.0)
        component.last_dof_vel = np.array([1.0, 1.0, 1.0])
        
        state = Mock()
        state.dof_vel = np.array([1.1, 0.9, 1.0])  # Small changes to avoid huge penalties
        
        calculator = Mock()
        calculator.dt = 0.05
        
        reward = component.calculate(state, calculator)
        
        # Should penalize acceleration (note: implementation uses last - current)
        # Calculate expected manually: (1.0-1.1, 1.0-0.9, 1.0-1.0) = (-0.1, 0.1, 0.0)
        # Divided by dt: (-2.0, 2.0, 0.0), squared: (4.0, 4.0, 0.0), sum: 8.0, negative: -8.0
        expected_reward = -8.0
        assert reward == pytest.approx(expected_reward)


class TestContactPenaltyComponent:
    """Test cases for ContactPenaltyComponent."""
    
    def test_calculate_no_contacts(self):
        """Test calculation with no unwanted contacts."""
        component = ContactPenaltyComponent("contact_penalty", weight=1.0)
        
        state = Mock()
        state.contact_floor_balls = []
        
        calculator = Mock()
        
        reward = component.calculate(state, calculator)
        
        assert reward == 0.0
        
    def test_calculate_with_contacts(self):
        """Test calculation with unwanted contacts."""
        component = ContactPenaltyComponent("contact_penalty", weight=1.0)
        
        state = Mock()
        state.contact_floor_balls = [1, 2, 3]  # 3 unwanted contacts
        
        calculator = Mock()
        
        reward = component.calculate(state, calculator)
        
        assert reward == -3.0


class TestActionRateComponent:
    """Test cases for ActionRateComponent."""
    
    def test_calculate(self):
        """Test action rate penalty calculation."""
        component = ActionRateComponent("action_rate", weight=1.0)
        
        state = Mock()
        state.action_history = Mock()
        state.action_history.last_last_action = np.array([0.5, -0.5, 0.0])
        state.action_history.last_action = np.array([1.0, -1.0, 0.5])
        
        calculator = Mock()
        calculator.dt = 0.05
        
        reward = component.calculate(state, calculator)
        
        # Should return action rate (positive value)
        action_diff = state.action_history.last_action - state.action_history.last_last_action
        expected_rate = np.sum(np.square(action_diff)) / calculator.dt
        assert reward == pytest.approx(expected_rate)


class TestRewardCalculator:
    """Test cases for RewardCalculator class."""
    
    @pytest.fixture
    def basic_reward_config(self):
        """Create a basic reward configuration using ConfigRegistry."""
        cfg = ConfigRegistry.create_from_name("basic_quadruped")
        return cfg
        
    def test_initialization(self, basic_reward_config):
        """Test RewardCalculator initialization."""
        calculator = RewardCalculator(basic_reward_config)
        
        assert len(calculator.components) == 2
        assert calculator.components[0].name == "forward_velocity"
        assert calculator.components[1].name == "go_straight"
        assert calculator.dt == 0.05
        
    def test_component_names_property(self, basic_reward_config):
        """Test component_names property."""
        calculator = RewardCalculator(basic_reward_config)
        
        names = calculator.component_names
        assert names == ["forward_velocity", "go_straight"]
        
    def test_get_component(self, basic_reward_config):
        """Test get_component method."""
        calculator = RewardCalculator(basic_reward_config)
        
        component = calculator.get_component("forward_velocity")
        assert component is not None
        assert component.name == "forward_velocity"
        
        non_existent = calculator.get_component("non_existent")
        assert non_existent is None
        
    def test_calculate_total_reward(self, basic_reward_config):
        """Test total reward calculation."""
        calculator = RewardCalculator(basic_reward_config)
        
        # Mock state with all required attributes
        state = Mock()
        state.accurate_vel_body = np.array([0.6, 0.0, 0.0])  # Match target velocity
        state.ang_vel_body = np.array([0.0, 0.0, 0.0])  # Match target angular velocity
        state.accurate_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        state.accurate_ang_vel_body = np.array([0.0, 0.0, 0.0])  # Angular velocity in body frame
        
        total_reward, info = calculator.calculate(state)
        
        # Forward velocity component should give a good reward, go_straight should also be good
        # Total: forward_velocity * 0.7 + go_straight * 0.3
        assert total_reward > 0  # Should be positive with good tracking
        
        # Check info dictionary structure
        assert "component_values" in info
        assert "forward_velocity" in info["component_values"]
        assert "go_straight" in info["component_values"]
        assert "total_reward" in info
        
    def test_reset_components(self, basic_reward_config):
        """Test reset functionality."""
        calculator = RewardCalculator(basic_reward_config)
        
        # Should not raise any errors
        calculator.reset()
        
    def test_string_representation(self, basic_reward_config):
        """Test string representation."""
        calculator = RewardCalculator(basic_reward_config)
        
        str_repr = str(calculator)
        assert "RewardCalculator" in str_repr
        assert "forward_velocity" in str_repr
        assert "go_straight" in str_repr


class TestRewardIntegration:
    """Integration tests for reward system with full environment."""
    
    @pytest.fixture
    def test_config(self):
        """Create a test configuration using ConfigRegistry."""
        cfg = ConfigRegistry.create_from_name("basic_quadruped")
        return cfg
    
    @patch('metamachine.environments.base.KBHit')
    def test_reward_calculation_in_environment(self, mock_kbhit, test_config):
        """Test reward calculation in full environment."""
        env = MetaMachine(test_config)
        obs, info = env.reset()
        
        # Take a step
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        # Check reward properties
        assert isinstance(reward, (int, float))
        assert np.isfinite(reward)
        
        # Check info contains reward breakdown
        assert "component_values" in info
        assert "forward_velocity" in info["component_values"]
        assert "go_straight" in info["component_values"]
        assert "total_reward" in info
        
        env.close()
        
    @patch('metamachine.environments.base.KBHit')
    def test_reward_consistency_across_steps(self, mock_kbhit, test_config):
        """Test reward consistency across multiple steps."""
        env = MetaMachine(test_config)
        obs, info = env.reset()
        
        rewards = []
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            # Check reward is finite
            assert np.isfinite(reward)
            rewards.append(reward)
            
            # Check info structure is consistent
            assert "component_values" in info
            assert "forward_velocity" in info["component_values"]
            assert "go_straight" in info["component_values"]
            assert "total_reward" in info
            
            if done or truncated:
                break
                
        # Rewards should be reasonable (not all exactly the same)
        assert len(set(rewards)) > 1 or len(rewards) == 1  # Allow for single step
        
        env.close()
        
    @patch('metamachine.environments.base.KBHit')
    def test_reward_component_weights(self, mock_kbhit, test_config):
        """Test that reward component weights are applied correctly."""
        env = MetaMachine(test_config)
        obs, info = env.reset()
        
        # Take a step
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        # Check that individual components are weighted
        forward_velocity_reward = info["component_values"]["forward_velocity"]
        go_straight_reward = info["component_values"]["go_straight"]
        total_reward = info["total_reward"]
        
        # Total should be weighted sum (approximately, due to floating point)
        expected_total = forward_velocity_reward * 0.7 + go_straight_reward * 0.3
        assert total_reward == pytest.approx(expected_total, abs=1e-6)
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__]) 