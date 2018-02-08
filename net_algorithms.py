# Convnet algorithms
# contains:
#   BaseNet: pytroch tutorials convnet
#   Net : poor experiment of making a convnet
#   TransferNet: takes an Alexnet base from the ImageNet contest and performs fine tunning for the specific problem
#   optimizer(): performs stochastic gradient decent along the vector space of images
#
#   function descriptions:
#       nn.conv2d performs
#       nn. MaxPool(a,b) takes the maximum out of a*b segment of matrix
#       nn.relu is an activator function max(0,x) where x is the tensor.
#       nnleaky_relu is an activator function
#       nn.Linear(a,b) linear tensor transformation from a dimension to b dimension
#
#
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
initial_model = models.alexnet(pretrained=True)


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 *5 )
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TransferNet(nn.Module):
    def __init__(self):
        super(TransferNet, self).__init__()
        self.features = initial_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 11 * 11, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0),256*11*11)
        y = self.classifier(f)
        return y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 36, 1)
        self.fc1 = nn.Linear(82944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 60)
        self.fc4 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view([x.size()[0], -1])
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MinimalNet(nn.Module):
    def __init__(self):
        super(MinimalNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv = nn.Conv2d(3, 6, 5)
        self.fc1 = nn.Linear(36 * 2 * 2, 120)

    def forward(self, x):
        x = x
        return x


def loss():
    return nn.CrossEntropyLoss()


def optimizer(net):
    return optim.SGD(net.parameters(), lr=0.001, momentum=0.9)