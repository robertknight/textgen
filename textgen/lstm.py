"""
An implementation of LSTM cells using PyTorch.

PyTorch has a built-in implementation of LSTM (torch.nn.LSTM). This is a
re-implementation purely for educational purposes.

References:
    - http://colah.github.io/posts/2015-08-Understanding-LSTMs/
"""

from typing import Tuple, Optional

from torch import Tensor
import torch
import torch.nn as nn


def uniform(shape, min_, max_, requires_grad=False):
    result = torch.rand(shape) * (max_ - min_) + min_
    result.requires_grad_(requires_grad)
    return result


class LSTMCell(nn.Module):
    __constants__ = ["hidden_size"]

    def __init__(self, input_size, hidden_size):
        super().__init__()

        def weight_param(for_hidden=False):
            if for_hidden:
                weight_shape = (hidden_size, hidden_size)
            else:
                weight_shape = (hidden_size, input_size)
            return nn.Parameter(uniform(weight_shape, -0.2, 0.2))

        def bias_param():
            return nn.Parameter(torch.zeros(hidden_size, 1))

        # Parameters for forget gate that chooses what to keep/forget in
        # cell state at each step.
        self.forget_gate_hidden_weights = weight_param(for_hidden=True)
        self.forget_gate_input_weights = weight_param()
        self.forget_gate_bias = bias_param()

        # Parameters for input gate that chooses what to add to cell state from
        # input at current step.
        self.input_gate_hidden_weights = weight_param(for_hidden=True)
        self.input_gate_input_weights = weight_param()
        self.input_gate_bias = bias_param()

        # Parameters for creating update to add to cell state at current step.
        self.cell_update_hidden_weights = weight_param(for_hidden=True)
        self.cell_update_input_weights = weight_param()
        self.cell_update_bias = bias_param()

        # Parameters for generating output at current step.
        self.output_gate_hidden_weights = weight_param(for_hidden=True)
        self.output_gate_input_weights = weight_param()
        self.output_gate_bias = bias_param()

        self.hidden_size = hidden_size

    def forward(self, x, state: Tuple[Tensor, Tensor]):
        """
        Process a batch through the LSTM cell.

        :param x: (batch_size, input_size) Tensor
        :param state: (hidden_state, cell_state) Tensor tuple
        """

        (prev_output, cell_state) = state

        x_t = x.t()
        prev_output_t = prev_output.t()

        forget_probs = torch.sigmoid(
            (self.forget_gate_input_weights @ x_t)
            + (self.forget_gate_hidden_weights @ prev_output_t)
            + self.forget_gate_bias
        )

        input_probs = torch.sigmoid(
            (self.input_gate_input_weights @ x_t)
            + (self.input_gate_hidden_weights @ prev_output_t)
            + self.input_gate_bias
        )

        cell_state_update = torch.tanh(
            (self.cell_update_input_weights @ x_t)
            + (self.cell_update_hidden_weights @ prev_output_t)
            + self.cell_update_bias
        )
        cell_state_update = input_probs * cell_state_update

        cell_state = cell_state * forget_probs.t() + cell_state_update.t()

        output = torch.sigmoid(
            (self.output_gate_input_weights @ x_t)
            + (self.output_gate_hidden_weights @ prev_output_t)
            + self.output_gate_bias
        )
        output = output.t() * torch.tanh(cell_state)

        return (output, cell_state)


class LSTM(nn.Module):
    """
    Long Short-Term Memory layer.

    This has the same basic API as torch.nn.LSTM to make comparisons easy.
    """

    __constants__ = ["num_layers"]

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()

        self.cell = LSTMCell(input_size, hidden_size)
        self.add_module("lstm_cell", self.cell)

        if num_layers != 1:
            # Multiple layers are not supported, but the API includes a layer
            # dimension in its output tensors for compatibility with PyTorch's
            # native LSTM.
            raise ValueError("num_layers parameter can only be 1")
        self.num_layers = num_layers

    def forward(self, x, state: Optional[Tuple[Tensor, Tensor]] = None):
        """
        Process a batch of sequences and return a batch of outputs.

        :param x: Input as (seq_length, batch_size, input_dims) tensor
        :param state: Tuple of (hidden_state, cell_state) tensors, each of size
          (num_layers, batch_size, hidden_size). Initialized to zeros if None.
        """
        seq_length, batch_size, input_dims = x.shape

        if state is None:
            last_output = torch.zeros(
                (self.num_layers, batch_size, self.cell.hidden_size), device=x.device
            )
            cell_state = torch.zeros(
                (self.num_layers, batch_size, self.cell.hidden_size), device=x.device
            )
        else:
            last_output, cell_state = state

        outputs = []
        for i in range(seq_length):
            last_output, cell_state = self.cell(x[i], (last_output[0], cell_state[0]))
            # Re-insert the layer dimension.
            cell_state = torch.unsqueeze(cell_state, dim=0)
            last_output = torch.unsqueeze(last_output, dim=0)
            # Return the output from the last layer.
            outputs.append(last_output[0])
        return torch.stack(outputs), (last_output, cell_state)
