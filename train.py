# this file contains functions involved in training the neural net
# functions:
#   train(): trains the passed in neural net with the training data and functions passed in
import torch
import numpy as np
from torch.autograd import Variable
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(net, trainloader, optimizer, criterion):
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        # print(trainloader,"output")
        for i, data in enumerate(trainloader):
            # print(data,"testing")
            inputs, target = data
            print(target)
            # target = torch.LongTensor(np.asarray(target, int))
            # print(target)
            # wrap them in Variable
            inputs, target = Variable(inputs), Variable(target)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
