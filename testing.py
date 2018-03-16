# file of testing metrics of the ConvNet Algorithms
# functions:
# 	getaccuracy(): compares prediction to the actual label in test data set
# 	getaccuracybyclass(): compares prediction to the actual label for every class
#   getconfusionmatrix(): creates a confusion matrix for classification
#   multiclass_simplify_to_binary(): for a specific class simplifies the matrix to a two by two matrix based on one vs all manner.
#   computeMAUCScore(): takes each pairwise AUC and takes the average to make a MAUC metric

import torch
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import metrics
import torch.nn.functional as F

def classify(img_name, net, transform, classes):
    image = Image.open(img_name)
    if torch.cuda.is_available():
        image = transform(image).view((1, 3, 400, 400))
        output = net(Variable(image.cuda()))
    else:
        image = transform(image).view((1, 3, 400, 400))
        output = net(Variable(image))
    _, c = torch.max(output.data, 1)
    print(classes[c[0]])


def get_accuracy(matrix, classes):
    correct = 0
    total = matrix.sum()
    for i in range(len(classes)):
        correct += matrix[i][i]
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def get_accuracy_by_class(matrix, classes):
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    for i in range(len(classes)):
        class_correct[i] = matrix[i][i]
        class_total[i] = matrix[i].sum()

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


def compute_confusion_matrix(testloader, net, classes):
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    ypred = []
    yactual = []
    yscore = np.zeros([0,len(classes)])
    for data in testloader:
        images, labels = data
        batch_size = images.size()[0]
        if torch.cuda.is_available():
            outputs = net(Variable(images.cuda()))
        else:
            outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        for i in range(batch_size):
            yscore=np.vstack((yscore,F.softmax(outputs[i],0).data.cpu().numpy().astype('float'))) # the softmax fetches the probabilities of the net. the data.cpu().numpy() converts tensor to numpy even with cuda
            ypred.append(predicted[i])
            yactual.append(labels[i])
            confusion_matrix[labels[i]][predicted[i]] += 1
    print(confusion_matrix, confusion_matrix.sum())
    return confusion_matrix, ypred, yactual, yscore


def multi_class_simplify_to_binary(matrix, classtype):
    binary_matrix = np.zeros((2, 2), dtype=int)
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


def roc_curve(score, labels, classes):
    for i in range(len(classes)):
        iscore = score[:, i]
        fpr, tpr, _ = metrics.roc_curve(labels, iscore, i)
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
    if math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) != 0:
        mcc_val = (tp * tn - fp * fn)/(math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)))
    else:
        mcc_val = "no entries in class"
    return mcc_val


def auc_metric(score, labels, classes):
    #print(score)
    for i in range(len(classes)):
        iscore = score[:,i]
        #print(iscore)
        if len(score) >0 and len(labels) >0:
            fpr, tpr, thresholds = metrics.roc_curve(labels, iscore, i)
            # print(tpr)
            # print(fpr)
            # print(thresholds)
            auc_val = metrics.auc(fpr, tpr)
        else:
            auc_val = "predicted has no entries"
        print('AUC score of', classes[i], ':', auc_val)

def MAUCscore(score,labels,classes):
    sumAuc=0
    for i in range(len(classes)):
        for j in range(len(classes)):
            if(i!=j):
                ijlabels=[x for x in labels if (x == i or x == j)]
                ijscore=  [x[i] for x in score if (x == i or x == j)]
                print(ijlabels , '\n', ijscore)
                fpr,tpr,_ = metrics.roc_curve(ijlabels,ijscore)
                sumAuc += metrics.auc(fpr,tpr)
    avg=1/((classes)*(classes-1))
    print("MAUC score:",avg)

def get_mcc_by_class(matrix, classes):
    for i in range(len(classes)):
        binary_matrix = multi_class_simplify_to_binary(matrix,i)
        score = mcc_score(binary_matrix)
        print('MCC Score of', classes[i], ":", score)
