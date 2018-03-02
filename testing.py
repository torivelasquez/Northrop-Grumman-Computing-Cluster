# file of testing metrics of the ConvNet Algorithms
# functions:
# 	getaccuracy(): compares prediction to the actual label in test data set
# 	getaccuracybyclass(): compares prediction to the actual label for every class
#   getconfusionmatrix(): creates a confusion matrix for classification
#   multiclass_simplify_to_binary(): for a specific class simplifies the matrix to a two by two matrix based on one vs all manner.

import torch
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import metrics

def classify(img_name, net, transform, classes):
    image = Image.open(img_name)
    image = transform(image).view((1, 3, 400, 400))
    output = net(Variable(image))
    _, c = torch.max(output.data, 1)
    print(classes[c[0]])


def get_accuracy(matrix,classes):
    correct = 0
    total = matrix.sum()
    for i in range(len(classes)):
        correct += matrix[i][i]
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def get_accuracy_by_class(matrix,classes):
    class_correct=list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    for i in range(len(classes)):
        class_correct[i]=matrix[i][i]
        class_total[i]=matrix[i].sum()

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


def compute_confusion_matrix(testloader,net,classes):
    confusion_matrix = np.zeros((len(classes), len(classes)),dtype=int)
    ypred = []
    yactual = []
    for data in testloader:
        images, labels = data
        batch_size = images.size()[0]
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        for i in range(batch_size):
            ypred.append(predicted[i])
            yactual.append(labels[i])
            confusion_matrix[labels[i]][predicted[i]]+=1
    print(confusion_matrix,confusion_matrix.sum())
    return confusion_matrix,ypred,yactual


def multi_class_simplify_to_binary(matrix,classtype):
    binary_matrix = np.zeros((2,2),dtype=int)
    side=int(math.sqrt(matrix.size))
    for i in range(side):
        for j in range(side):
            if classtype == i and classtype == j:
                binary_matrix[0][0] += matrix[i][j]
            elif classtype != i and classtype == j:
                binary_matrix[0][1] += matrix[i][j]
            elif classtype == i and classtype != j:
                binary_matrix[1][0] += matrix[i][j]
            else:
                binary_matrix[1][1] += matrix[i][j]
    return binary_matrix


def roc_curve(predicted, labels, classes):
    for i in range(len(classes)):
        fpr, tpr, _ = metrics.roc_curve(predicted, labels,i)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.01])
        plt.ylim([0.0, 1.01])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for %s %%' % (classes[i]))
        plt.xlabel('FPR (1 - Specificity)')
        plt.ylabel('TPR (Sensitivity)')
        plt.grid(True)
        plt.show()


def mcc_score(binary_matrix):
    tp = binary_matrix[0][0]
    tn = binary_matrix[1][1]
    fp = binary_matrix[0][1]
    fn = binary_matrix[1][0]
    mcc_val = (tp * tn - fp * fn)/(math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)))
    return mcc_val

def auc_metric(predicted, labels, classes):
    for i in range(len(classes)):
        fpr, tpr, _ = metrics.roc_curve(predicted,labels,i)
        auc_val = metrics.auc(fpr,tpr)
        print('AUC score of', classes[i], ':', auc_val)

def get_mcc_by_class(matrix,classes):
    for i in range(len(classes)):
        binary_matrix = multi_class_simplify_to_binary(matrix,i)
        score = mcc_score(binary_matrix)
        print('MCC Score of', classes[i], ":", score)
