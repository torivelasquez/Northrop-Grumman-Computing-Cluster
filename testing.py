# file of testing metrics of the ConvNet Algorithms
# functions:
# 	getaccuracy(): compares prediction to the actual label in test data set
# 	getaccuracybyclass(): compares prediction to the actual label for every class


import torch
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def classify(img_name, net, transform, classes):
    image = Image.open(img_name)
    image = transform(image).view((1, 3, 400, 400))
    output = net(Variable(image))
    _, c = torch.max(output.data, 1)
    print(classes[c[0]])


def get_accuracy(testloader, net):
    correct = 0
    incorrect=0
    total = 0
    TP=0
    TN=0
    FP=0
    FN=0
    for data in testloader:
        images, labels = data
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        incorrect += (predicted != labels).sum()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def get_accuracy_by_class(testloader, net, classes):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for data in testloader:
        images, labels = data
        batch_size = images.size()[0]
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(batch_size):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


def compute_confusion_matrix(testloader,net,classes):
    confusionmatrix = np.zeros((len(classes), len(classes)),dtype=int)
    print(confusionmatrix)
    for data in testloader:
        images, labels = data
        batch_size = images.size()[0]
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        for i in range(batch_size):
            confusionmatrix[labels[i]][predicted[i]]+=1
    print(confusionmatrix,confusionmatrix.sum())

