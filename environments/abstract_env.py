"""
Abstract Environment Interface for CloudSimPy DRL Scheduler

This module defines the abstract base class and common interfaces for
both training and testing environments.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from enum import Enum


class EnvironmentType(Enum):
    """Environment type enumeration"""
    TRAINING = "training"
    TESTING = "testing"


@dataclass
class EnvironmentConfig:
    """Configuration for environment creation"""
    env_type: EnvironmentType
    dataset_args: Dict[str, Any]
    reward_weights: Dict[str, float] = None
    performance_mode: str = "balanced"  # "speed", "accuracy", "balanced"
    debug_mode: bool = False
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Set default reward weights if not provided"""
        if self.reward_weights is None:
            self.reward_weights = {"makespan": 0.7, "energy": 0.3}


class AbstractSchedulingEnvironment(gym.Env, ABC):
    """
    Abstract base class for workflow scheduling environments.
    
    This class defines the common interface that both training and testing
    environments must implement, ensuring consistency while allowing
    implementation-specific optimizations.
    """
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        self.config = config
        self.metadata = {"render_modes": ["human", "rgb_array"]}
        self._episode_count = 0
        self._step_count = 0
        
        # Initialize spaces (will be updated by subclasses)
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
    
    @abstractmethod
    def _setup_spaces(self) -> Tuple[spaces.Space, spaces.Space]:
        """
        Setup observation and action spaces based on dataset.
        
        Returns:
            Tuple of (observation_space, action_space)
        """
        pass
    
    @abstractmethod
    def _calculate_reward(self, state: Any, action: Any, next_state: Any) -> float:
        """
        Calculate reward for state transition.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Reward value
        """
        pass
    
    @abstractmethod
    def _is_terminal(self, state: Any) -> bool:
        """
        Check if current state is terminal.
        
        Args:
            state: Current state
            
        Returns:
            True if terminal, False otherwise
        """
        pass
    
    @abstractmethod
    def _get_info(self, state: Any) -> Dict[str, Any]:
        """
        Get additional information about current state.
        
        Args:
            state: Current state
            
        Returns:
            Dictionary with additional information
        """
        pass
    
    @abstractmethod
    def _validate_action(self, action: Any) -> bool:
        """
        Validate if action is legal in current state.
        
        Args:
            action: Action to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def _reset_implementation(self, seed: Optional[int], options: Optional[Dict[str, Any]]):
        """
        Environment-specific reset implementation.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        pass
    
    @abstractmethod
    def _step_implementation(self, action):
        """
        Environment-specific step implementation.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        pass
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed, options=options)
        
        # Use config seed if no seed provided
        if seed is None:
            seed = self.config.seed
            
        self._episode_count += 1
        self._step_count = 0
        
        return self._reset_implementation(seed, options)
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1
        
        if not self._validate_action(action):
            return self._handle_invalid_action(action)
        
        return self._step_implementation(action)
    
    def _handle_invalid_action(self, action):
        """
        Handle invalid actions with appropriate penalties.
        
        Args:
            action: Invalid action
            
        Returns:
            Tuple of (observation, penalty_reward, terminated, truncated, info)
        """
        penalty_reward = -1000.0
        info = {
            "error": f"Invalid action: {action}",
            "penalty": penalty_reward,
            "episode": self._episode_count,
            "step": self._step_count
        }
        
        # Return current observation with penalty
        obs = self.observation_space.sample()  # Fallback observation
        return obs, penalty_reward, True, False, info
    
    def get_episode_count(self) -> int:
        """Get current episode count"""
        return self._episode_count
    
    def get_step_count(self) -> int:
        """Get current step count"""
        return self._step_count
    
    def get_config(self) -> EnvironmentConfig:
        """Get environment configuration"""
        return self.config
    
    def render(self, mode="human"):
        """
        Render the environment (placeholder implementation).
        
        Args:
            mode: Rendering mode
        """
        if mode == "human":
            print(f"Environment: {self.__class__.__name__}")
            print(f"Episode: {self._episode_count}, Step: {self._step_count}")
        elif mode == "rgb_array":
            # Return a placeholder RGB array
            return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def close(self):
        """Close the environment and clean up resources"""
        pass


class EnvironmentMetrics:
    """Helper class for tracking environment metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0
        self.total_episodes = 0
    
    def record_episode(self, reward: float, length: int):
        """Record episode metrics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.total_steps += length
        self.total_episodes += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.episode_rewards:
            return {"episodes": 0}
        
        return {
            "episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "avg_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "avg_length": np.mean(self.episode_lengths),
            "std_length": np.std(self.episode_lengths),
            "min_reward": np.min(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards)
        }

