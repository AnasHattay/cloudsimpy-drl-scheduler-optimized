import abc
from typing import Optional

import torch
from torch import nn


class Agent(nn.Module, abc.ABC):
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gets the value of a given state.

        :param x: (batch_size, N)
        :return values: (batch_size,)
        """
        raise NotImplementedError()

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gets the action and value of a given state.

        :param x: (batch_size, N)
        :param action: (batch_size,)
        :return chosen_actions: (batch_size,)
        :return log_probs: (batch_size,)
        :return entropies: (batch_size,)
        :return values: (batch_size,)
        """
        raise NotImplementedError()
