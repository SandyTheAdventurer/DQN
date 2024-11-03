
import torch.nn as nn
import torch

class Base(torch.nn.Module):
    def __init__(self, input_size, output_size, probs=False) -> None:
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.Softmax() if probs else nn.Identity()
        )

        self.optimizer = torch.optim.Adam(self.linear_relu_stack.parameters())
        self.loss = nn.MSELoss()

    def forward(self, X: torch.Tensor):
        return self.linear_relu_stack(X)
    
    def step_optimizer(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update(self, X: torch.Tensor, y: torch.Tensor):
        y_pred = self(X)
        loss = self.loss(y_pred, y)
        self.step_optimizer(loss)
        return loss.item()
    
def printb(*messages):
    width = max(len(message) for message in messages) + 4
    print("+" + "-" * width + "+")
    for message in messages:
        print("| " + message.ljust(width - 2) + " |")
    print("+" + "-" * width + "+")

class Buffer:
    def __init__(self, data=[], max_size=5):
        self.max_size = max_size
        self.data = data

    def append(self, item):
        if len(self.data) >= self.max_size:
            self.data.pop(0)
        self.data.append(item)

    def __len__(self):
        return len(self.data)

    def remove(self, item):
        if item in self.data:
            self.data.remove(item)

    def __repr__(self):
        return repr(self.data)
