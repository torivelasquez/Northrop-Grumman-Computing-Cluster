# file of testing metrics of the ConvNet Algorithms
# functions:
# 	getaccuracy(): compares prediction to the actual label in test data set
# 	getaccuracybyclass(): compares prediction to the actual label for every class
#   compute_confusion_matrix(): creates a confusion matrix for classification and s
#       -returns
#       -confusion matrix
#       -probabilities of each class being classified
#       -true labels of classification
#   multiclass_simplify_to_binary(): for a specific class simplifies the matrix to a two by two matrix based on one vs all manner.
#   computeMAUCScore(): takes each pairwise AUC and takes the average to make a MAUC metric

import torch
import math
import warnings
import os.path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from sklearn import metrics
from torch.autograd import Variable
from pandas_ml import ConfusionMatrix


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
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (accuracy))

    return accuracy


def get_accuracy_by_class(matrix, classes):
    class_accuracies = []
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    for i in range(len(classes)):
        class_correct[i] = matrix[i][i]
        class_total[i] = matrix[i].sum()

    for i in range(len(classes)):
        accuracy = 100 * class_correct[i] / class_total[i]
        class_accuracies.append(accuracy)
        print('Accuracy of %5s : %2d %%' % (classes[i], accuracy))

    return class_accuracies


def compute_confusion_matrix(testloader, net, classes):
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    ypred = []
    yactual = []
    yscore = np.zeros([0, len(classes)])
    for data in testloader:
        images, labels = data
        batch_size = images.size()[0]
        if torch.cuda.is_available():
            outputs = net(Variable(images.cuda()))
        else:
            outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        for i in range(batch_size):
            yscore = np.vstack((yscore, F.softmax(outputs[i],0).data.cpu().numpy().astype('float'))) # the softmax fetches the probabilities of the net. the data.cpu().numpy() converts tensor to numpy even with cuda
            ypred.append(predicted[i])
            yactual.append(labels[i])
            confusion_matrix[labels[i]][predicted[i]] += 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = ConfusionMatrix(yactual, ypred)
        statistics = cm.stats()
    print(confusion_matrix, confusion_matrix.sum())

    return confusion_matrix, statistics, yactual, yscore


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


def multi_class_labels_to_binary(labels,pos_class):
    binarylabels=[]
    for i in range(len(labels)):
        if(labels[i]==pos_class):
            binarylabels.append(1)
        else:
            binarylabels.append(0)
    return binarylabels


def roc_curve(score, labels, classes):
    for i in range(len(classes)):
        print(score)
        iscore = score[:, i] # probabilities of the positive class
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
        j = 0
        while os.path.exists(classes[i] + "%s.png" %j):
            j += 1
        plt.savefig(classes[i] + "%s.png" %j)


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
    auc_values = []
    for i in range(len(classes)):
        iscore = score[:,i]
        if len(score) > 0 and len(labels) > 0:
            fpr, tpr, thresholds = metrics.roc_curve(labels, iscore, i)
            auc_val = metrics.auc(fpr, tpr)
        else:
            auc_val = "predicted has no entries"
        auc_values.append(auc_val)
        print('AUC score of', classes[i], ':', auc_val)

    return auc_values


def auc_confidence_interval(score,labels,classes):
    confidence_intervals=[]
    for k in range(len(classes)):
        nplabels=np.array(multi_class_labels_to_binary(labels,k))
        iscore = np.array(score[:, k])
        n_bootstraps = 1000
        rng_seed = 42  # control reproducibility
        bootstrapped_scores = []

        rng = np.random.RandomState(rng_seed)
        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.random_integers(0, len(iscore) - 1, len(iscore))
            if len(np.unique(nplabels[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
            a_score = metrics.roc_auc_score(nplabels[indices], iscore[indices])
            bootstrapped_scores.append(a_score)
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
        print(" 95% Confidence interval for the class{:s} : [{:0.3f} - {:0.3}]".format(" "+classes[k],confidence_lower, confidence_upper))
        confidence_intervals.append([confidence_lower,confidence_upper])
    return confidence_intervals


def aucpair(i, j, score, labels):
    ijlabels = []
    ijscore = []
    for k in range(len(score)):
        if(labels[k] == i or labels[k] == j):
            ijlabels.append(labels[k])
            ijscore.append(score[k])
    return ijlabels, ijscore


def MAUCscore(score, labels, classes):
    sumAuc = 0
    for i in range(len(classes)):
        iscore = score[:, i]
        for j in range(len(classes)):
            if(i != j):
                ijlabels, ijscore = aucpair(i, j, iscore, labels)
                fpr, tpr, _ = metrics.roc_curve(ijlabels, ijscore, pos_label=i)
                sumAuc += metrics.auc(fpr, tpr)
    avg = 1/((len(classes))*(len(classes)-1))*sumAuc
    print("MAUC score:", avg)
    return avg


def get_mcc_by_class(matrix, classes):
    class_mcc = []
    for i in range(len(classes)):
        binary_matrix = multi_class_simplify_to_binary(matrix, i)
        score = mcc_score(binary_matrix)
        class_mcc.append(score)
        print('MCC Score of', classes[i], ":", score)

    return class_mcc
