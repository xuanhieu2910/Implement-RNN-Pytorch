import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math

BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 1



class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Linear(DIM_IN, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
        )
        self.layer_stack_2 = torch.nn.Sequential(
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_SIZE, DIM_OUT),
        )

    def forward(self, x):
        layer_out_1 = self.layer_stack(x)
        return self.layer_stack_2(layer_out_1)



def main():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
    ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)
    for i in range(20):
        optimizer.zero_grad()
        prediction = model(some_input)
        print("Weight before:\n", model.layer_stack_2[2].weight[:, :10])
        loss = (ideal_output - prediction).pow(2).sum()
        loss.backward()
        print("---------------------------------------")
        print("Weight dao ham:\n")
        print(model.layer_stack_2[2].weight.grad[:, :10])
        optimizer.step()
        print("---------------------------------------")
        print("Weight after update:\n")
        print(model.layer_stack_2[2].weight[:, :10])

if __name__ == '__main__':
    main()


