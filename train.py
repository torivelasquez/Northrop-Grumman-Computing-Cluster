# this file contains functions involved in training the neural net
# functions:
#   train(): trains the passed in neural net with the training data and functions passed in
import torch
import numpy as np
from torch.autograd import Variable
from timeit import default_timer as timer

"""
train() trains the net and returns  how long to took to train.
trainloader: the loader for the training images
optimizer: the optimizer algorithm
criterion: the loss function
epohcs: how many epochs it trains for
"""


def train(net, trainloader, optimizer, criterion, epochs):
    start = timer()
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        # print(trainloader,"output")
        for i, data in enumerate(trainloader):
            inputs, target = data
            # target = torch.LongTensor(np.asarray(target, int))
            # print(target)

            # Check if GPUs are available and wrap data in variables
            if torch.cuda.is_available():
                inputs, target = Variable(inputs.cuda()), Variable(target.cuda())
            else:
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
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
    end = timer()
    training_time = end - start
    print('Finished Training in', training_time, 'seconds')

    return training_time
