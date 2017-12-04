import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)   #
        self.conv2 = nn.Conv2d(6, 16, 5) # vectorized node
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)    # linear transform Ax+b

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # of each matrix pick largest element
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x)) # max(0,x)
        x = F.relu(self.fc2(x)) # max (0,x)
        x = self.fc3(x)
        return x


def loss():
    return nn.CrossEntropyLoss()


def optimizer(net):
    return optim.SGD(net.parameters(), lr=0.001, momentum=0.9)