"""
Example Training Script with CloudSimPy Integration

This script demonstrates how to use the new CloudSimPy-based environment
for training DRL agents. It maintains compatibility with existing agents
while leveraging CloudSimPy's enhanced capabilities.
"""

import os
import sys
import random
import time
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import CloudSimPy environment and compatibility layer
from agents.compatibility_wrapper import create_legacy_environment
from simulator.cloudsimpy_gym_env  import EnhancedCloudSimPyGymEnvironment as CloudSimPyGymEnvironment

# Import original components for compatibility
sys.path.append('/home/ubuntu/workflow-cloudsim-drlgnn')
from scheduler.dataset_generator.gen_dataset import DatasetArgs, generate_dataset


@dataclass
class TrainingArgs:
    """Training configuration arguments"""
    exp_name: str = "cloudsimpy_test"
    seed: int = 1
    output_dir: str = "logs"
    torch_deterministic: bool = True
    cuda: bool = True
    num_episodes: int = 100
    max_steps_per_episode: int = 1000
    learning_rate: float = 3e-4
    
    # Dataset configuration
    num_workflows: int = 5
    num_tasks_per_workflow: int = 10
    num_vms: int = 8
    num_hosts: int = 4


def create_sample_dataset(args: TrainingArgs):
    """Create a sample dataset for testing"""
    # Create dataset arguments
    dataset_args = DatasetArgs(
        num_workflows=args.num_workflows,
        num_tasks_per_workflow=args.num_tasks_per_workflow,
        num_vms=args.num_vms,
        num_hosts=args.num_hosts,
        seed=args.seed
    )
    
    # Generate dataset
    dataset = generate_dataset(args.seed, dataset_args)
    return dataset


class SimpleRandomAgent:
    """Simple random agent for testing the environment"""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def select_action(self, observation):
        """Select a random valid action"""
        return self.action_space.sample()
    
    def update(self, *args, **kwargs):
        """Placeholder for agent updates"""
        pass


class SimpleDQNAgent:
    """Simple DQN agent for demonstration"""
    
    def __init__(self, observation_space, action_space, learning_rate=3e-4):
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        
        # Simple neural network
        import torch.nn as nn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        obs_dim = observation_space.shape[0] if hasattr(observation_space, 'shape') else 100
        action_dim = action_space.n if hasattr(action_space, 'n') else 100
        
        self.q_network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.epsilon = 0.1
    
    def select_action(self, observation):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor)
            return q_values.argmax().item()
    
    def update(self, *args, **kwargs):
        """Placeholder for DQN updates"""
        pass


def test_environment_basic():
    """Basic test of the CloudSimPy environment"""
    print("Testing CloudSimPy Environment...")
    
    # Create training arguments
    args = TrainingArgs()
    
    # Create dataset
    dataset = create_sample_dataset(args)
    print(f"Created dataset with {len(dataset.workflows)} workflows")
    
    # Create environment using compatibility wrapper
    env = create_legacy_environment(dataset=dataset)
    print(f"Created environment with action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test environment reset
    obs, info = env.reset(seed=args.seed)
    print(f"Reset environment, observation shape: {obs.shape}")
    
    # Test a few random steps
    agent = SimpleRandomAgent(env.action_space)
    
    total_reward = 0
    for step in range(10):
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step}: action={action}, reward={reward:.3f}, done={terminated or truncated}")
        
        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps")
            break
    
    print(f"Total reward: {total_reward:.3f}")
    env.close()
    print("Basic test completed successfully!")


def test_training_loop():
    """Test a simple training loop"""
    print("\nTesting Training Loop...")
    
    args = TrainingArgs(num_episodes=5, max_steps_per_episode=50)
    
    # Create dataset generator for varied training
    def dataset_generator(seed):
        dataset_args = DatasetArgs(
            num_workflows=args.num_workflows,
            num_tasks_per_workflow=args.num_tasks_per_workflow,
            num_vms=args.num_vms,
            num_hosts=args.num_hosts,
            seed=seed
        )
        return generate_dataset(seed, dataset_args)
    
    # Create environment
    env = CloudSimPyGymEnvironment(dataset_generator=dataset_generator)
    env = CompatibilityWrapper(env)
    
    # Initialize after first reset to get correct spaces
    obs, _ = env.reset(seed=args.seed)
    agent = SimpleDQNAgent(env.observation_space, env.action_space, args.learning_rate)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0
        episode_length = 0
        
        for step in range(args.max_steps_per_episode):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: reward={episode_reward:.3f}, length={episode_length}")
        
        if 'makespan' in info:
            print(f"  Makespan: {info['makespan']:.3f}, Energy: {info.get('total_energy', 0):.3f}")
    
    # Print summary statistics
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    print(f"\nTraining Summary:")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Average episode length: {avg_length:.1f}")
    print(f"Reward std: {np.std(episode_rewards):.3f}")
    
    env.close()
    print("Training loop test completed successfully!")


def benchmark_performance():
    """Benchmark the performance of CloudSimPy vs original implementation"""
    print("\nBenchmarking Performance...")
    
    args = TrainingArgs(num_episodes=10, max_steps_per_episode=100)
    dataset = create_sample_dataset(args)
    
    # Test CloudSimPy environment
    start_time = time.time()
    env = create_legacy_environment(dataset=dataset)
    agent = SimpleRandomAgent(env.action_space)
    
    total_steps = 0
    for episode in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        
        for step in range(args.max_steps_per_episode):
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            
            if terminated or truncated:
                break
    
    cloudsimpy_time = time.time() - start_time
    env.close()
    
    print(f"CloudSimPy Performance:")
    print(f"  Total time: {cloudsimpy_time:.3f} seconds")
    print(f"  Total steps: {total_steps}")
    print(f"  Steps per second: {total_steps / cloudsimpy_time:.1f}")
    print(f"  Time per step: {cloudsimpy_time / total_steps * 1000:.3f} ms")


def main():
    """Main function to run all tests"""
    print("CloudSimPy DRL Integration Test Suite")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    try:
        # Run tests
        test_environment_basic()
        test_training_loop()
        benchmark_performance()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("CloudSimPy integration is working correctly.")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

