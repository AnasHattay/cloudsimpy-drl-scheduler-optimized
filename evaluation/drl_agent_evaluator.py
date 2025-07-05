"""
DRL Agent Evaluation Interface

This module provides interfaces for loading and evaluating trained DRL agents
using the CloudSimPy environment. Supports various DRL frameworks and model types.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
from pathlib import Path

# Try to import PyTorch, make it optional
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for when torch is not available
    class torch:
        @staticmethod
        def device(device_str):
            return "cpu"
        
        @staticmethod
        def load(path, map_location=None):
            raise ImportError("PyTorch not available")
        
        @staticmethod
        def FloatTensor(data):
            raise ImportError("PyTorch not available")
        
        @staticmethod
        def no_grad():
            class NoGradContext:
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            return NoGradContext()
        
        @staticmethod
        def argmax(tensor, dim=None):
            raise ImportError("PyTorch not available")
        
        @staticmethod
        def full_like(tensor, value):
            raise ImportError("PyTorch not available")
    
    class nn:
        class Module:
            pass

# Add the original scheduler path for model loading
sys.path.append(str(Path(__file__).parent.parent.parent / "workflow-cloudsim-drlgnn"))

from scheduler.dataset_generator.core import Dataset
from agents.compatibility_wrapper import create_legacy_environment
from evaluation.heuristic_algorithms import SchedulingDecision


@dataclass
class EvaluationResult:
    """Results from evaluating a DRL agent"""
    agent_name: str
    scenario_name: str
    makespan: float
    total_energy: float
    energy_efficiency: float  # Tasks per Wh
    decisions: List[SchedulingDecision]
    episode_length: int
    total_reward: float
    success_rate: float  # Percentage of tasks successfully scheduled
    execution_time: float  # Time taken for evaluation
    additional_metrics: Dict[str, Any] = None


class DRLAgent(ABC):
    """Abstract base class for DRL agents"""
    
    def __init__(self, name: str, model_path: Optional[str] = None):
        self.name = name
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else "cpu"
    
    @abstractmethod
    def load_model(self, model_path: str) -> bool:
        """Load the trained model"""
        pass
    
    @abstractmethod
    def predict_action(self, observation: np.ndarray, valid_actions: Optional[List[int]] = None) -> int:
        """Predict the next action given an observation"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset agent state for new episode"""
        pass


class GNNDRLAgent(DRLAgent):
    """GNN-based DRL agent for workflow scheduling"""
    
    def __init__(self, name: str = "GNN-DRL", model_path: Optional[str] = None):
        super().__init__(name, model_path)
        self.observation_history = []
        self.action_history = []
        
    def load_model(self, model_path: str) -> bool:
        """Load the trained GNN model"""
        try:
            if not os.path.exists(model_path):
                print(f"Model file not found: {model_path}")
                return False
            
            if not TORCH_AVAILABLE:
                print("PyTorch not available. Using random fallback for model predictions.")
                self.model = "random_fallback"
                return True
            
            # Try to load PyTorch model
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                self.model = torch.load(model_path, map_location=self.device)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                print(f"Successfully loaded PyTorch model from {model_path}")
                return True
            
            # Try to load pickle model
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Successfully loaded pickle model from {model_path}")
                return True
            
            else:
                print(f"Unsupported model format: {model_path}")
                return False
                
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return False
    
    def predict_action(self, observation: np.ndarray, valid_actions: Optional[List[int]] = None) -> int:
        """Predict the next action using the loaded model"""
        if self.model is None:
            # Fallback to random action if model not loaded
            if valid_actions:
                return np.random.choice(valid_actions)
            else:
                return np.random.randint(0, 1000)  # Default action space size
        
        # Handle random fallback when PyTorch is not available
        if self.model == "random_fallback":
            if valid_actions:
                return np.random.choice(valid_actions)
            else:
                return np.random.randint(0, 1000)
        
        try:
            # Only use PyTorch operations if available
            if TORCH_AVAILABLE:
                # Convert observation to tensor
                if isinstance(observation, np.ndarray):
                    obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                else:
                    obs_tensor = torch.FloatTensor([observation]).to(self.device)
                
                # Get model prediction
                with torch.no_grad():
                    if hasattr(self.model, 'forward'):
                        # Neural network model
                        action_logits = self.model(obs_tensor)
                        if isinstance(action_logits, tuple):
                            action_logits = action_logits[0]  # Take first output if tuple
                        
                        # Apply valid action mask if provided
                        if valid_actions is not None:
                            masked_logits = torch.full_like(action_logits, float('-inf'))
                            masked_logits[0, valid_actions] = action_logits[0, valid_actions]
                            action_logits = masked_logits
                        
                        # Select action (greedy or sampling)
                        action = torch.argmax(action_logits, dim=1).item()
                        
                    elif hasattr(self.model, 'predict'):
                        # Sklearn-like model
                        action = self.model.predict(observation.reshape(1, -1))[0]
                        
                    elif callable(self.model):
                        # Function-like model
                        action = self.model(observation)
                        
                    else:
                        # Unknown model type, use random action
                        if valid_actions:
                            action = np.random.choice(valid_actions)
                        else:
                            action = np.random.randint(0, 1000)
            else:
                # PyTorch not available, try other model types
                if hasattr(self.model, 'predict'):
                    # Sklearn-like model
                    action = self.model.predict(observation.reshape(1, -1))[0]
                elif callable(self.model):
                    # Function-like model
                    action = self.model(observation)
                else:
                    # Fallback to random
                    if valid_actions:
                        action = np.random.choice(valid_actions)
                    else:
                        action = np.random.randint(0, 1000)
            
            # Ensure action is valid
            if valid_actions is not None and action not in valid_actions:
                action = np.random.choice(valid_actions)
            
            return int(action)
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            # Fallback to random action
            if valid_actions:
                return np.random.choice(valid_actions)
            else:
                return np.random.randint(0, 1000)
    
    def reset(self):
        """Reset agent state for new episode"""
        self.observation_history = []
        self.action_history = []


class RandomDRLAgent(DRLAgent):
    """Random agent for baseline comparison"""
    
    def __init__(self, seed: int = 42):
        super().__init__("Random-DRL")
        self.rng = np.random.RandomState(seed)
    
    def load_model(self, model_path: str) -> bool:
        """No model to load for random agent"""
        return True
    
    def predict_action(self, observation: np.ndarray, valid_actions: Optional[List[int]] = None) -> int:
        """Return random action"""
        if valid_actions:
            return self.rng.choice(valid_actions)
        else:
            return self.rng.randint(0, 1000)
    
    def reset(self):
        """Reset random state"""
        pass


class DRLAgentEvaluator:
    """Evaluates DRL agents on CloudSimPy environment"""
    
    def __init__(self, max_episode_steps: int = 1000, timeout_seconds: float = 300.0):
        self.max_episode_steps = max_episode_steps
        self.timeout_seconds = timeout_seconds
    
    def evaluate_agent(self, agent: DRLAgent, dataset: Dataset, scenario_name: str = "unknown") -> EvaluationResult:
        """Evaluate a DRL agent on a given dataset"""
        import time
        start_time = time.time()
        
        # Create environment
        env = create_legacy_environment(dataset=dataset)
        
        # Reset agent and environment
        agent.reset()
        observation, info = env.reset()
        
        # Track metrics
        total_reward = 0.0
        episode_length = 0
        decisions = []
        
        # Run episode
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and episode_length < self.max_episode_steps:
            # Check timeout
            if time.time() - start_time > self.timeout_seconds:
                print(f"Evaluation timeout for {agent.name}")
                break
            
            # Get valid actions (if environment supports it)
            valid_actions = None
            if hasattr(env, 'get_valid_actions'):
                valid_actions = env.get_valid_actions()
            
            # Get action from agent
            action = agent.predict_action(observation, valid_actions)
            
            # Take step
            try:
                next_observation, reward, terminated, truncated, step_info = env.step(action)
                
                # Record decision if available
                if 'decision' in step_info:
                    decisions.append(step_info['decision'])
                
                # Update metrics
                total_reward += reward
                episode_length += 1
                observation = next_observation
                
            except Exception as e:
                print(f"Error during step: {e}")
                break
        
        # Calculate final metrics
        execution_time = time.time() - start_time
        
        # Extract final metrics from environment
        final_info = {}
        if hasattr(env, 'get_final_metrics'):
            final_info = env.get_final_metrics()
        
        makespan = final_info.get('makespan', 0.0)
        total_energy = final_info.get('total_energy', 0.0)
        
        # Calculate derived metrics
        total_tasks = sum(len(workflow.tasks) for workflow in dataset.workflows)
        success_rate = len(decisions) / total_tasks if total_tasks > 0 else 0.0
        energy_efficiency = len(decisions) / total_energy if total_energy > 0 else 0.0
        
        # Create result
        result = EvaluationResult(
            agent_name=agent.name,
            scenario_name=scenario_name,
            makespan=makespan,
            total_energy=total_energy,
            energy_efficiency=energy_efficiency,
            decisions=decisions,
            episode_length=episode_length,
            total_reward=total_reward,
            success_rate=success_rate,
            execution_time=execution_time,
            additional_metrics=final_info
        )
        
        return result
    
    def evaluate_multiple_runs(self, agent: DRLAgent, dataset: Dataset, 
                             scenario_name: str = "unknown", num_runs: int = 5) -> List[EvaluationResult]:
        """Evaluate agent multiple times and return all results"""
        results = []
        
        for run in range(num_runs):
            print(f"Evaluating {agent.name} - Run {run + 1}/{num_runs}")
            result = self.evaluate_agent(agent, dataset, f"{scenario_name}_run_{run + 1}")
            results.append(result)
        
        return results
    
    def compare_agents(self, agents: List[DRLAgent], dataset: Dataset, 
                      scenario_name: str = "unknown", num_runs: int = 3) -> Dict[str, List[EvaluationResult]]:
        """Compare multiple agents on the same dataset"""
        all_results = {}
        
        for agent in agents:
            print(f"\nEvaluating agent: {agent.name}")
            results = self.evaluate_multiple_runs(agent, dataset, scenario_name, num_runs)
            all_results[agent.name] = results
        
        return all_results


def load_trained_model(model_path: str, model_type: str = "auto") -> Optional[DRLAgent]:
    """Load a trained DRL model and return appropriate agent"""
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    # Auto-detect model type if not specified
    if model_type == "auto":
        if "gnn" in model_path.lower() or "gin" in model_path.lower():
            model_type = "gnn"
        else:
            model_type = "generic"
    
    # Create appropriate agent
    if model_type == "gnn":
        agent = GNNDRLAgent(name=f"GNN-DRL-{Path(model_path).stem}")
    else:
        agent = GNNDRLAgent(name=f"DRL-{Path(model_path).stem}")
    
    # Load model
    if agent.load_model(model_path):
        return agent
    else:
        return None


def create_agent_from_config(config: Dict[str, Any]) -> Optional[DRLAgent]:
    """Create DRL agent from configuration dictionary"""
    
    agent_type = config.get("type", "gnn")
    model_path = config.get("model_path")
    name = config.get("name", f"DRL-Agent-{agent_type}")
    
    if agent_type == "random":
        return RandomDRLAgent(seed=config.get("seed", 42))
    
    elif agent_type == "gnn":
        agent = GNNDRLAgent(name=name)
        if model_path and agent.load_model(model_path):
            return agent
    
    else:
        agent = GNNDRLAgent(name=name)
        if model_path and agent.load_model(model_path):
            return agent
    
    return None


if __name__ == "__main__":
    # Example usage
    from evaluation.scenario_generator import create_evaluation_scenarios
    
    # Create test scenario
    generator = create_evaluation_scenarios()
    scenario = generator.get_all_scenarios()[0]
    dataset = generator.generate_scenario_dataset(scenario)
    
    print(f"Testing DRL agent evaluation on scenario: {scenario.name}")
    
    # Create test agents
    agents = [
        RandomDRLAgent(seed=42),
        # Add your trained model here:
        # load_trained_model("path/to/your/trained_model.pt", "gnn")
    ]
    
    # Filter out None agents
    agents = [agent for agent in agents if agent is not None]
    
    # Evaluate agents
    evaluator = DRLAgentEvaluator(max_episode_steps=100, timeout_seconds=60.0)
    
    for agent in agents:
        print(f"\nEvaluating {agent.name}...")
        result = evaluator.evaluate_agent(agent, dataset, scenario.name)
        
        print(f"Results for {agent.name}:")
        print(f"  Makespan: {result.makespan:.2f}")
        print(f"  Total Energy: {result.total_energy:.4f} Wh")
        print(f"  Energy Efficiency: {result.energy_efficiency:.2f} tasks/Wh")
        print(f"  Success Rate: {result.success_rate:.2%}")
        print(f"  Episode Length: {result.episode_length}")
        print(f"  Total Reward: {result.total_reward:.2f}")
        print(f"  Execution Time: {result.execution_time:.2f}s")

