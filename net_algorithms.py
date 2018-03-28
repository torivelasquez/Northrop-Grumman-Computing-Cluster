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
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import runtime_parameters
alexnet = models.alexnet(pretrained=True)
resnet = models.resnet18(pretrained=True)

class BaseNet(nn.Module):
    def __init__(self, output_size,params_t):
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
    def __init__(self, output_size,params_t):
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
    def __init__(self,output_size,params_t):
        super(ResNet, self).__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            # nn.Linear(1568, 256),
            # print(nn.Linear(512 * 9, output_size)),
            nn.Linear(512 * 7 * 7, output_size)
        )

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), 512 * 7 * 7) # the size of tensor is 512 * 7 * 7
        y = self.classifier(f)
        return y


# params = runtime_parameters.Parameters()
#layers = runtime_parameters.Parameters().get_layers()


class LayeredResNet(nn.Module):
    def __init__(self,output_size,params_t):
        super(LayeredResNet, self).__init__()
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.layers = params_t.get_layers()
        self.layersequence=[]
        self.make_layers()
        print(runtime_parameters.Parameters().list())
        self.classifier = nn.Sequential(
            *self.layersequence,
            # [[nn.ReLU(inplace=True) for i in range(layers[j])] for j in range(len(layers))],
            nn.Linear(512 * 7 * 7, output_size)
        )
        print(self.classifier)
        # self._initialize_weights()

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), 512 * 7 * 7) # the size of tensor is 512 * 7 * 7
        y = self.classifier(f)
        return y

    def make_layers(self):
        for i in range(self.layers[0]):
            self.layersequence+=[nn.ReLU(inplace=True)] #,nn.Linear(512*7*7,512*7*7)]



    #  def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #             m.weight.data.normal_(0, math.sqrt(2. / n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
    #         elif isinstance(m, nn.Linear):
    #             n = m.weight.size(1)
    #             m.weight.data.normal_(0, 0.01)
    #             m.bias.data.zero_()


# def added_layers(cfg,batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)
#
# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
#
# def new_Resnet():
#     model = LayeredResNet(added_layers(cfg['A']))
#     return model



class Net(nn.Module):
    def __init__(self, output_size,params_t):
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
    def __init__(self, output_size,params_t):
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


nets = {"transfer": TransferNet, "simple": Net, "resnet":ResNet, "grayscale": GrayNet, "layeredresnet": LayeredResNet}


def get_net(net_name, num_classes,params_t):
    if net_name in nets:
        return nets[net_name](num_classes,params_t)
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
