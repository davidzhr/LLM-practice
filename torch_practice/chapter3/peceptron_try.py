
import torch
import torch.nn as nn
import torch.optim as optim


class Perceptron(nn.Module):

    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        return torch.sigmoid(self.fc1(x_in)).squeeze()


input_dim = 2
lr = 0.001

perceptron = Perceptron(input_dim)
bce_loss = nn.BCELoss()
optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)


# for epoch_i in range(n_epochs):


