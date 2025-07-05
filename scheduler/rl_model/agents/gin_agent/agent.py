from typing import Optional, Tuple
import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from torch.nn.functional import softmax

from torch_geometric.nn.models import GIN
from torch_geometric.nn.glob import global_mean_pool

from scheduler.config.settings import MAX_OBS_SIZE
from scheduler.rl_model.agents.agent import Agent
from scheduler.rl_model.agents.gin_agent.mapper import GinAgentMapper, GinAgentObsTensor


# Base Gin Network
# ----------------------------------------------------------------------------------------------------------------------


class BaseGinNetwork(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        self.task_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(self.device)
        self.vm_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        ).to(self.device)
        self.graph_network = GIN(
            in_channels=embedding_dim,
            hidden_channels=hidden_dim,
            num_layers=3,
            out_channels=embedding_dim,
        ).to(self.device)

    def __call__(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__call__(*args, **kwargs)

    def forward(self, obs: GinAgentObsTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tasks = obs.task_state_scheduled.shape[0]
        num_vms = obs.vm_completion_time.shape[0]

        task_features = [obs.task_state_scheduled, obs.task_state_ready, obs.task_length, obs.task_completion_time]
        vm_features = [obs.vm_completion_time, 1 / (obs.vm_speed + 1e-8), obs.vm_energy_rate]

        # Encode tasks
        task_x = torch.stack(task_features, dim=-1)
        task_h: torch.Tensor = self.task_encoder(task_x)

        # Encode VMs
        vm_x = torch.stack(vm_features, dim=-1)
        vm_h: torch.Tensor = self.vm_encoder(vm_x)

        # Structuring nodes as [0, 1, ..., T-1] [T, T+1, ..., T+VM-1], edges are between Tasks -> Compatible VMs
        task_vm_edges = obs.compatibilities.clone()
        task_vm_edges[1] = task_vm_edges[1] + num_tasks  # Reindex VMs

        # Get features
        node_x = torch.cat([task_h, vm_h])
        edge_index = torch.cat([task_vm_edges, obs.task_dependencies], dim=-1)

        # Get embeddings
        batch = torch.zeros(num_tasks + num_vms, dtype=torch.long, device=self.device)
        node_embeddings = self.graph_network(node_x, edge_index=edge_index)
        edge_embeddings = torch.cat([node_embeddings[edge_index[0]], node_embeddings[edge_index[1]]], dim=1)
        graph_embedding = global_mean_pool(node_embeddings, batch=batch)

        return node_embeddings, edge_embeddings, graph_embedding


# Gin Actor
# ----------------------------------------------------------------------------------------------------------------------


class GinActor(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device

        self.network = BaseGinNetwork(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            device=device,
        )
        self.edge_scorer = nn.Sequential(
            nn.Linear(3 * embedding_dim, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return super().__call__(*args, **kwargs)

    def forward(self, obs: GinAgentObsTensor) -> torch.Tensor:
        num_tasks = obs.task_completion_time.shape[0]
        num_vms = obs.vm_completion_time.shape[0]

        _, edge_embeddings, graph_embedding = self.network(obs)

        # Get edge embedding scores
        rep_graph_embedding = graph_embedding.expand(edge_embeddings.shape[0], self.embedding_dim)
        edge_embeddings = torch.cat([edge_embeddings, rep_graph_embedding], dim=1)
        edge_embedding_scores: torch.Tensor = self.edge_scorer(edge_embeddings)

        # Extract the exact edges
        task_vm_edge_scores = edge_embedding_scores.flatten()
        task_vm_edge_scores = task_vm_edge_scores[: obs.compatibilities.shape[1]]

        # Actions scores should be the value in edge embedding, but -inf on invalid actions
        action_scores = torch.ones((num_tasks, num_vms), dtype=torch.float32).to(self.device) * -1e8
        action_scores[obs.compatibilities[0], obs.compatibilities[1]] = task_vm_edge_scores
        action_scores[obs.task_state_ready == 0, :] = -1e8  # Remove scores of actions with not ready tasks

        return action_scores


# Gin Critic
# ----------------------------------------------------------------------------------------------------------------------


class GinCritic(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int, device: torch.device) -> None:
        super().__init__()
        self.device = device

        self.network = BaseGinNetwork(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            device=device,
        )
        self.graph_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return super().__call__(*args, **kwargs)

    def forward(self, obs: GinAgentObsTensor) -> torch.Tensor:
        # Critic value is derived from global graph state
        _, _, graph_embedding = self.network(obs)
        return self.graph_scorer(graph_embedding.unsqueeze(0))


# Gin Agent
# ----------------------------------------------------------------------------------------------------------------------


class GinAgent(Agent, nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        self.mapper = GinAgentMapper(MAX_OBS_SIZE)
        self.actor = GinActor(hidden_dim=64, embedding_dim=16, device=device)
        self.critic = GinCritic(hidden_dim=64, embedding_dim=16, device=device)


    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        batch_size = x.shape[0]
        values = []

        for batch_index in range(batch_size):
            decoded_obs = self.mapper.unmap(x[batch_index])
            value = self.critic(decoded_obs)
            values.append(value)

        return torch.stack(values).to(self.device)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        batch_size = x.shape[0]
        all_chosen_actions, all_log_probs, all_entropies, all_values = [], [], [], []

        for batch_index in range(batch_size):
            decoded_obs = self.mapper.unmap(x[batch_index])
            action_scores = self.actor(decoded_obs)
            action_scores = action_scores.flatten()
            action_probabilities = softmax(action_scores, dim=0)

            probs = Categorical(action_probabilities)
            chosen_action = action[batch_index] if action is not None else probs.sample()
            value = self.critic(decoded_obs)

            all_chosen_actions.append(chosen_action)
            all_log_probs.append(probs.log_prob(chosen_action))
            all_entropies.append(probs.entropy())
            all_values.append(value)

        chosen_actions = torch.stack(all_chosen_actions).to(self.device)
        log_probs = torch.stack(all_log_probs).to(self.device)
        entropies = torch.stack(all_entropies).to(self.device)
        values = torch.stack(all_values).to(self.device)

        return chosen_actions, log_probs, entropies, values
