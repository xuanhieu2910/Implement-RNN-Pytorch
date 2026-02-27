import torch.nn as nn
import torch
from torch.nn.parameter import Parameter

class RNN_Cell(nn.Module):

    def __init__(self, input_dim, hidden_dim, sigma=0.01):
        super().__init__()
        self.W_xh = Parameter(torch.rand(input_dim, hidden_dim) * sigma)
        self.W_hh = Parameter(torch.rand(hidden_dim, hidden_dim) * sigma)
        self.b_h = Parameter(torch.zeros(hidden_dim))

    def forward(self, inputs, state=None):
        """
            forward to calculate RNN cell
        :param inputs: [seq_len, batch_size, input_dim]
            - seq_len: sequence length
            - batch_size: batch size
            - input_dim: input dimension
        :param state: status state
        :return: sequence state
        """
        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            # Because: batch_size -> numbers of sample in inputs.
            # hidden_dim -> dimension of hidden of calculate W_x_h * h_t
            state = torch.zeros( (inputs[1], self.hidden_dim))
        else:
            state = state

        outputs = []
        for X in inputs:
            state = torch.tanh(
                torch.matmul(X, self.W_xh) + torch.matmul(state, self.W_hh) + self.b_h
            )
            outputs.append(state)
        return torch.stack(outputs), state
