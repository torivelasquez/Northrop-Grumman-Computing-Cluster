# this file contains the interface which serves as the driver for the rest of the software

import sys
import os.path
import runtime_parameters
import net_algorithms
import parser.parser as parser
import transformations
import itertools
from data_spliter import data_spliter
from testing import MAUCscore, auc_metric, roc_curve, get_accuracy, get_accuracy_by_class, classify, compute_confusion_matrix, mcc_score, multi_class_simplify_to_binary, get_mcc_by_class
from train import train
import torch


def len_test(cmd_split, num):
    if len(cmd_split) == num:
        return True
    print(cmd_split[0], " expects ", num, " arguments ", len(cmd_split), " given. Use <help> for help")
    return False


def train_macro(params_t):
    transform = transformations.get_transform(params_t.train_transform[0])
    data_set, classes = parser.get_data(transform, params_t.images_loc[0], params_t.train_data_loc[0], params_t.grayscale[0])
    net_t = net_algorithms.get_net(params_t.net_type[0], len(classes))
    if torch.cuda.device_count() > 1:
        net_t = torch.nn.DataParallel(net_t)
    if torch.cuda.is_available():
        net_t.cuda()
    criterion = net_algorithms.get_criterion(params_t.criterion[0])
    optimizer = net_algorithms.get_optimizer(params_t.optimizer[0], net_t, params_t.learning_rate[0], params_t.momentum[0])
    train(net_t, data_set, optimizer, criterion, params_t.epochs[0])
    return net_t


def test_macro(net_t, params_t):
    transform = transformations.get_transform(params_t.test_transform[0])
    data_set, classes = parser.get_data(transform, params_t.images_loc[0], params_t.test_data_loc[0], params_t.grayscale[0])
    confusion_matrix, predicted, labels, score = compute_confusion_matrix(data_set, net_t, classes)
    print(confusion_matrix)
    get_accuracy(confusion_matrix, classes)
    get_accuracy_by_class(confusion_matrix, classes)
    #  get_mcc_by_class(confusion_matrix , classes)
    auc_metric(score, labels, classes)
    MAUCscore(score, labels, classes)
    roc_curve(score, labels, classes)


params = runtime_parameters.Parameters()
if len(sys.argv) == 2:
    params.set("file", sys.argv[1])
while True:
    cmd = input(">>>")
    cmd_split = cmd.split()
    if cmd_split[0] == "quit":
        break
    elif cmd_split[0] == "train":
        if len_test(cmd_split, 1):
            net = train_macro(params)

    elif cmd_split[0] == "test":
        if len_test(cmd_split, 1):
            test_macro(net, params)

    elif cmd_split[0] == "iter":
        if len_test(cmd_split, 1):
            l = params.list()
            combinations = list(itertools.product(*l))
            for param_list in combinations:
                net_t2 = train_macro(runtime_parameters.TempParams(param_list))
                test_macro(net_t2, runtime_parameters.TempParams(param_list))


    elif cmd_split[0] == "class":
        if len_test(cmd_split, 2):
            if os.path.isfile(params.images_loc[0] + cmd_split[1]):
                transform = transformations.get_transform(params.test_transform[0])
                data_set, classes = parser.get_data(transform, params.images_loc[0], params.train_data_loc[0], params.grayscale[0])
                classify(params.images_loc[0] + cmd_split[1], net, transform, classes)
            else:
                print("image not found")

    elif cmd_split[0] == "save":
        if len_test(cmd_split, 1):
            torch.save(net.state_dict(), params.save_loc[0])
            # torch.save(net, params.save_loc[0])

    elif cmd_split[0] == "load":
        if len_test(cmd_split, 1):
            file = params.load_loc[0]
            if os.path.isfile(file):
                transform = transformations.get_transform(params.test_transform[0])
                _, classes = parser.get_data(transform, params.images_loc[0], params.test_data_loc[0], params.grayscale[0])
                net = net_algorithms.get_net(params.net_type[0], len(classes))
                if torch.cuda.device_count() > 1:
                    net = torch.nn.DataParallel(net)
                if torch.cuda.is_available():
                    net.cuda()
                net.load_state_dict(torch.load(params.load_loc[0]))
                # net = torch.load(params.load_loc[0])
            else:
                print("neural net not found")

    elif cmd_split[0] == 'set':
        if len_test(cmd_split, 3):
            params.set(cmd_split[1], cmd_split[2])

    elif cmd_split[0] == "split":
        if len_test(cmd_split, 3):
            source_file = cmd_split[1]
            if os.path.isfile(source_file):
                number_of_splits = int(cmd_split[2])
                target_files = []
                file_div_percents = []
                for i in range(0, number_of_splits):
                    label_request = "What would you like to label subset " + str(i + 1) + ":\n"
                    target_files.append(input(label_request))
                    frac_request = "What fraction of the dataset should make up subset " + str(i + 1) + "?\n"
                    file_div_percents.append(float(input(frac_request)))
                result = data_spliter(source_file, target_files, file_div_percents)
                if result == 0:
                    print("Success")
                elif result == 1:
                    print("Specified Fractions do not add up to 1")
                else:
                    print("unidentified error")
            else:
                print("csv source not found")

    elif cmd_split[0] == "help":
        if len_test(cmd_split, 1):
            print(" <train> to train model\n <test> to test model\n <save> saves net\n <load> loads net\n <class> take"
                  " image and classify it with the net \n <settings> input file to configure the file and hyperparameters")

    else:
        print(cmd_split[0], " is not recognized, please give correct input, <help> for help")
