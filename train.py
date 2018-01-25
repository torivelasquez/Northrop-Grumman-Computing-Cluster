#train.py
#trains ConvNet
#
from torch.autograd import Variable

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(net,trainloader, optimizer, criterion):
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i,data in enumerate(trainloader):
            # get the inputs
            inputs, target = data
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
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
    print('Finished Training')