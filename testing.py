# file of testing metrics of the ConvNet Algorithms
# functions:
# 	getaccuracy(): compares prediction to the actual label in test data set
# 	getaccuracybyclass(): compares prediction to the actual label for every class


import torch
from torch.autograd import Variable


def get_accuracy(testloader, net):
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


def get_accuracy_by_class(testloader, net, classes):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        batch_size = images.size()[0]
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        print(images, labels, predicted)
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

