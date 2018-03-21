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
from datetime import datetime


def run_test(params):
    nets_to_run = params.get_net_type().split()
    net_save_names = params.get_save_loc().split()
    net_load_names = params.get_load_loc().split()
    net_epochs = list(map(int, params.get_epochs().split()))
    net_criterions = params.get_criterion().split()
    net_optimizers = params.get_optimizer().split()
    number_of_nets = len(nets_to_run)
    print("Testing:", nets_to_run)
    net_results = []

    # Get timestamp for result file and saved models
    runtime = datetime.now().replace(microsecond=0).isoformat().replace(':', '-')
    result_file = open("./testresults/" + runtime + "_testresults.csv", 'a')
    csv_header = "Neural Net Type,Epochs,Training Time (s),MAUC Score,Overall Accuracy"

    transform = transformations.get_transform(params.train_transform)
    training_set, training_classes = parser.get_data(transform, params.images_loc, params.train_data_loc, params.grayscale)

    transform = transformations.get_transform(params.test_transform)
    test_set, test_classes = parser.get_data(transform, params.images_loc, params.test_data_loc, params.grayscale)

    csv_header = csv_header + ',' + ','.join([s + " Accuracy" for s in test_classes]) + ',' + ','.join([s + " AUC" for s in test_classes]) + '\n'
    result_file.write(csv_header)

    for net_number, net in enumerate(nets_to_run):
        print("\nRunning {0:d}/{1:d}: {2:s}".format(net_number + 1, number_of_nets, net))
        net_statistics = [net, net_epochs[net_number]]

        model = net_algorithms.get_net(net, len(training_classes))
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        if torch.cuda.is_available():
            model.cuda()

        # Train or load net
        if net_load_names[net_number] == '0':
            criterion = net_algorithms.get_criterion(net_criterions[net_number])
            optimizer = net_algorithms.get_optimizer(net_optimizers[net_number], model)
            training_time = train(model, training_set, optimizer, criterion, net_epochs[net_number])
            net_statistics.append(training_time)
        else:
            model.load_state_dict(torch.load(net_load_names[net_number]))
            net_statistics.append("N/A")

        # Test net
        confusion_matrix, predicted, labels, score = compute_confusion_matrix(test_set, model, test_classes)
        net_statistics.append(MAUCscore(score, labels, test_classes))
        net_statistics.append(str(get_accuracy(confusion_matrix, test_classes)) + '%')
        net_statistics = [*net_statistics, *[str(i) + '%' for i in get_accuracy_by_class(confusion_matrix, test_classes)]]
        net_statistics = [*net_statistics, *auc_metric(score, labels, test_classes)]
        #  get_mcc_by_class(confusion_matrix, test_classes)
        #  roc_curve(score, labels, test_classes)

        entry = ','.join(map(str, net_statistics)) + '\n'
        result_file.write(entry)

        if net_save_names[net_number] != '0':
            model_name = runtime + '_' + net_save_names[net_number]
            torch.save(model.state_dict(), model_name)


    result_file.close()

def test_ui():
    if len(sys.argv) > 2:
        print("Wrong number of arguments.")
        print("Proper Usage: python algorithm_test.py <config file>")
        sys.exit()
    params = runtime_parameters.Parameters()
    if len(sys.argv) == 2:
        print("Config file detected, loading config file . . .")
        params.read_file(sys.argv[1])
        run_test(params)
    # else:
    #     ans = ''
    #     while True:
    #         print ("""
    #         1. Add net to test
    #         2. Set training dataset
    #         3. Set testing dataset
    #         4. Set images location
    #         5. Run test
    #         """)
    #         ans = input("Select an option: ")
    #         nets_to_run = []
    #         net_save_flags = []
    #         net_load_flags = []
    #         net_epochs = []
    #
    #         if ans == '1':
    #             net_type = input("Enter a net type: ").lower()
    #             params.set_net_type(net_type)
    #             load_flag = input("Load an existing {}? [y/n]: ".format(net_type))
    #             if

if __name__ == '__main__':
    test_ui()
