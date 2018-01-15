from torch.autograd import Variable
import netDefine as netDefine
import data as data
from testing import getaccuracy
from testing import getaccuracybyclass
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(net,trainloader, optimizer, criterion):
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i,data in enumerate(trainloader):
            # get the inputs
            #print(data)
            inputs, target = data
            #print(data,target)
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

if __name__ == "__main__":
	transform = data.transformation()
	net = netDefine.Net()
	criterion = netDefine.loss()
	optimizer = netDefine.optimizer(net)
	dataiter = iter(data.traindata(transform))
	images, labels = dataiter.next()
	#print(transform)
	train(net,data.traindata(transform), optimizer, criterion)