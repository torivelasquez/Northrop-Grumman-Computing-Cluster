import torch
import torchvision
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

def getaccuracy(testloader,net,images):
	correct = 0
	total = 0
	for data in testloader:
		images, labels = data
		outputs = net(Variable(images))
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()
	print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

########################################################################
# That looks waaay better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:
def getaccuracybyclass(testloader,net,images,classes):
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	for data in testloader:
		images, labels = data
		outputs = net(Variable(images))
		_, predicted = torch.max(outputs.data, 1)
		c = (predicted == labels).squeeze()
		for i in range(4):
			label = labels[i]
			class_correct[label] += c[i]
			class_total[label] += 1
	for i in range(10):
		print('Accuracy of %5s : %2d %%' % (
		classes[i], 100 * class_correct[i] / class_total[i]))

