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
alexnet = models.alexnet(pretrained=True)
resnet = models.resnet18(pretrained=True)

class BaseNet(nn.Module):
    def __init__(self, output_size):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 *5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TransferNet(nn.Module):
    def __init__(self, output_size):
        super(TransferNet, self).__init__()
        self.features = alexnet.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 11 * 11, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, output_size)
        )

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), 256*11*11)
        y = self.classifier(f)
        return y


class ResNet(nn.Module):
    def __init__(self,output_size):
        super(ResNet, self).__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            # nn.Linear(1568, 256),
            # print(nn.Linear(512 * 9, output_size)),
            nn.Linear(512 * 7 * 7, output_size)
        )

    def forward(self, x):
        # print(x)
        f = self.features(x)
        # print(f)
        f = f.view(f.size(0), 512 * 7 * 7) # the size of tensor is 512 * 7 * 7
        # print(f)
        y = self.classifier(f)
        return y


class Net(nn.Module):
    def __init__(self, output_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 36, 1)
        self.fc1 = nn.Linear(82944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 60)
        self.fc4 = nn.Linear(60, output_size)

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


class GrayNet(nn.Module):
    def __init__(self, output_size):
        super(GrayNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 36, 1)
        self.fc1 = nn.Linear(82944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 60)
        self.fc4 = nn.Linear(60, output_size)

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


nets = {"transfer": TransferNet, "simple": Net, "resnet":ResNet, "grayscale": GrayNet}


def get_net(net_name, num_classes):
    if net_name in nets:
        return nets[net_name](num_classes)
    else:
        raise Exception("{} is not a recognized net structure".format(net_name))


def cross_entropy():
    return nn.CrossEntropyLoss()


criterion = {"crossentropy": cross_entropy}


def get_criterion(criterion_name):
    if criterion_name in criterion:
        return criterion[criterion_name]()
    else:
        raise Exception("{} is not a recognized criterion".format(criterion_name))


def sgd(net,lrate,moment):
    # return optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    return optim.SGD(net.parameters(), lr=lrate, momentum=moment)


optimizers = {"sgd": sgd}


def get_optimizer(optimizer_name, net,lrate,moment):
    if optimizer_name in optimizers:
        return optimizers[optimizer_name](net,lrate,moment)
    else:
        raise Exception("{} is not a recognized criterion".format(optimizer_name))
