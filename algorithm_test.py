import sys
import os.path
import runtime_parameters
import net_algorithms
import parser.parser as parser
import transformations
import torch

from data_spliter import data_spliter
from testing import MAUCscore, auc_metric, roc_curve, get_accuracy, get_accuracy_by_class, classify, compute_confusion_matrix, mcc_score, multi_class_simplify_to_binary, get_mcc_by_class
from train import train
from pandas_ml import ConfusionMatrix
from datetime import datetime


def run_test():
    if len(sys.argv) > 2:
        print("Wrong number of arguments.")
        print("Proper Usage: python algorithm_test.py <config file>")
        sys.exit()

    params = runtime_parameters.Parameters()
    if len(sys.argv) == 2:
        params.read_file(sys.argv[1])
    nets_to_run = params.get_net_type().split()
    number_of_nets = len(nets_to_run)
    print("Testing:", nets_to_run)
    net_results = []

    # Get timestamp for file
    runtime = datetime.now().replace(microsecond=0).isoformat().replace(':', '-')
    result_file = open("./testresults/testresults_" + runtime + ".csv", 'a')
    csv_header = "Neural Net Type,Training Time (s),MAUC Score,Overall Accuracy"

    transform = transformations.get_transform(params.train_transform)
    training_set, training_classes = parser.get_data(transform, params.images_loc, params.train_data_loc, params.grayscale)
    print(type(training_classes))

    transform = transformations.get_transform(params.test_transform)
    test_set, test_classes = parser.get_data(transform, params.images_loc, params.test_data_loc, params.grayscale)

    csv_header = csv_header + ',' + ','.join([s + " Accuracy" for s in test_classes]) + ',' + ','.join([s + " AUC" for s in test_classes]) + '\n'
    result_file.write(csv_header)

    for net_number, net in enumerate(nets_to_run, 1):
        print("Running {0:d}/{1:d}: {2:s}".format(net_number, number_of_nets, net))
        net_statistics = [net]

        # Train net
        model = net_algorithms.get_net(net, len(training_classes))
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        if torch.cuda.is_available():
            model.cuda()
        criterion = net_algorithms.get_criterion(params.criterion)
        optimizer = net_algorithms.get_optimizer(params.optimizer, model)
        training_time = train(model, training_set, optimizer, criterion, params.epochs)
        net_statistics.append(training_time)

        # Test net
        confusion_matrix, statistics, predicted, labels, score = compute_confusion_matrix(test_set, model, test_classes)
        net_statistics.append(MAUCscore(score, labels, test_classes))
        net_statistics.append(str(get_accuracy(confusion_matrix, test_classes)) + '%')
        net_statistics = [*net_statistics, *[str(i) + '%' for i in get_accuracy_by_class(confusion_matrix, test_classes)]]
        net_statistics = [*net_statistics, *auc_metric(score, labels, test_classes)]
        #  get_mcc_by_class(confusion_matrix, test_classes)
        #  roc_curve(score, labels, test_classes)

        entry = ','.join(map(str, net_statistics)) + '\n'
        result_file.write(entry)

    result_file.close()

if __name__ == '__main__':
    run_test()
