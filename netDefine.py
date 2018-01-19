import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #3 in channels for RGB image,
        self.pool = nn.MaxPool2d(2, 2)   #take maximum of the 2 by 2 section
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # linear transform Ax+b, 400->120
                                              # vector 16=outchannels 5= kernel size of conv
        self.fc2 = nn.Linear(120, 84)   # linear transform Ax+b, 120->84 vector
        self.fc3 = nn.Linear(84, 10)  # linear transform Ax+b, 84->41 vector

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # maxpooling, Relu Activation function, Convolution , Relu function is max(0,x)
        x = self.pool(F.relu(self.conv2(x))) # maxpooling, Relu Activation function, Convolution , Relu function is max(0,x)
        x = x.view(-1, 16 * 5 *5 )
        x = F.relu(self.fc1(x)) # max(0,x) , means gradient function is step function of {0,1}
        x = F.relu(self.fc2(x)) #max (0,x), means gradient function is step function of {0,1}
        x = self.fc3(x)
        return x
class TransferNet(nn.Module):
    def __init__(self,initmodel):
        super(TransferNet, self).__init__()
        self.features = initmodel.features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 11 * 11, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 10),
        )

    def forward(self, x):
        #print(x,"initial")
        f=self.features(x)
        #print(f,"features")
        f=f.view(f.size(0),256*11*11)
        #print(f,"view")
        y=self.classifier(f)
        #print(y,"y")
        return y
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) #3 in channels for RGB image,
        self.pool = nn.MaxPool2d(2, 2)   #take maximum of the 2 by 2 section
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 36, 1)
        self.fc1 = nn.Linear(36 * 2 * 2, 120) # linear transform Ax+b, 400->120
                                              # vector 16=outchannels 5= kernel size of conv
        self.fc2 = nn.Linear(120, 84)   # linear transform Ax+b, 120->84 vector
        self.fc3 = nn.Linear(84, 60)  # linear transform Ax+b, 84->41 vector
        self.fc4 = nn.Linear(60, 10)    # linear transform Ax+b, 41->10 vector

    def forward(self, x):
        #print(x,"init")
        x = self.pool(F.leaky_relu(self.conv1(x))) # maxpooling, Relu Activation function, Convolution , Relu function is max(0,x)
        #print(x,"layer1")
        x = self.pool(F.leaky_relu(self.conv2(x))) # maxpooling, Relu Activation function, Convolution , Relu function is max(0,x)
        #print(x,"layer2")
        #print(F.leaky_relu(self.conv3(x)))
        x = self.pool(F.leaky_relu(self.conv3(x))) # maxpooling, Relu Activation function, Convolution , Relu function is max(0,x)

        #print(x,"layer3")
        x = x.view(-1, 36 * 2 *2 )
        #print(x,"xview")
        x = F.leaky_relu(self.fc1(x)) # max(0,x) , means gradient function is step function of {0,1}
        #print(x)
        x = F.leaky_relu(self.fc2(x)) #max (0,x), means gradient function is step function of {0,1}
        x = F.leaky_relu(self.fc3(x)) #max (0,x), means gradient function is step function of {0,1}
        #print(x)
        x = self.fc4(x)
        #print(x)
        #exit()
        return x
def loss():
    return nn.CrossEntropyLoss()


def optimizer(net):
    return optim.SGD(net.parameters(), lr=0.001, momentum=0.9)