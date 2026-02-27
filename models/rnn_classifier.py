import torch.nn as nn
from models.rnn_cell import RNN_Cell
class RNNClassifier(nn.Module):
    def __init__(self, input_size,hidden_size, num_classes):
        super(RNNClassifier, self).__init__()

        self.rnn = RNN_Cell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, last_state = self.rnn(x)
        out =  self.fc(last_state)
        return out