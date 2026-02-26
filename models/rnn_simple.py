import torch.nn as nn
import torch
from torch.nn.parameter import Parameter



class RNN_Simple(nn.Module):


    def __init__(self, input_dim, hidden_dim, output_dim, sigma=0.01):
        super(RNN_Simple, self).__init__()
        self.W_xh = Parameter(torch.rand(input_dim, hidden_dim) * sigma)
        self.W_hh = Parameter(torch.rand(hidden_dim, hidden_dim) * sigma)
        self.b_h = Parameter(torch.zeros(hidden_dim))


    def forward(self, inputs, state=None):
        """
        :param inputs: [num_steps,batch_size,num_inputs]
            - num_steps: int -> Sequence length of sequence
            - batch_size: int -> Batch size
            - num_inputs: int -> Feature dimension
            For example: In computer vision
                We have:
                    + 8 frames for action recognition
                    + 16 clips -> Batch sizes
                    + 1 frames -> 264 features
        :param state:
        :return:
        """

        if state is None:
            state = torch.zeros((inputs.shape[1], self.num_hiddens),
                                device=inputs.device)
        else:
            state, = state

        outputs = []
        for X in inputs:
            state = torch.tanh(torch.matmul(state, self.W_hh) + torch.matmul(X, self.W_xh) + self.b_h)
            outputs.append(state)
        return outputs, state
