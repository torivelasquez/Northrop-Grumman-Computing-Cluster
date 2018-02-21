# this file contains the interface which serves as the driver for the rest of the software

import sys
import runtime_parameters
import net_algorithms
import data
from testing import get_accuracy, get_accuracy_by_class, classify,compute_confusion_matrix,mcc_score, multi_class_simplify_to_binary,get_mcc_by_class
from train import train
import torch


def test_len(cmd_split, num):
    if len(cmd_split) == num:
        return True
    print(cmd_split[0], " expects ", num, " arguments ", len(cmd_split), " given. Use <help> for help")
    return False


params = runtime_parameters.Parameters()
print(sys.argv)
if len(sys.argv) == 2:
    print(sys.argv[1])
    params.read_file(sys.argv[1])
while True:
    cmd = input(">>>")
    cmd_split = cmd.split()
    if cmd_split[0] == "quit":
        break
    elif cmd_split[0] == "train":
        if test_len(cmd_split, 1):
            transform = data.get_transform(params.train_transform)
            data_set, classes = data.get_data(transform, params.images_loc, params.train_data_loc)
            net = net_algorithms.get_net(params.net_type, len(classes))
            criterion = net_algorithms.get_criterion(params.criterion)
            optimizer = net_algorithms.get_optimizer(params.optimizer, net)
            train(net, data_set, optimizer, criterion, params.epochs)
    elif cmd_split[0] == "test":
        if test_len(cmd_split, 1):
            transform = data.get_transform(params.test_transform)
            data_set, classes = data.get_data(transform, params.images_loc, params.test_data_loc)
            get_accuracy(data_set, net)
            get_accuracy_by_class(data_set, net, classes)
            confusion_matrix=compute_confusion_matrix(data_set, net, classes)
            #one_vs_all_matrix=multi_class_simplify_to_binary(confusion_matrix,1)
            #mcc_score(one_vs_all_matrix)
            get_mcc_by_class(confusion_matrix , classes)

    elif cmd_split[0] == "class":
        if test_len(cmd_split, 2):
            transform = data.get_transform(params.test_transform)
            data_set, classes = data.get_data(transform, params.images_loc, params.train_data_loc)
            classify(params.images_loc + cmd_split[1], net, transform, classes)
    elif cmd_split[0] == "save":
        if test_len(cmd_split, 1):
            torch.save(net, params.save_loc)
    elif cmd_split[0] == "load":
        if test_len(cmd_split, 1):
            net = torch.load(params.load_loc)
    elif cmd_split[0] == 'set':
        if test_len(cmd_split, 3):
            params.set(cmd_split[1], cmd_split[2])
    elif cmd_split[0] == "settings":
        if test_len(cmd_split, 2):
            params.read_file(cmd_split[1])
    elif cmd_split[0] == "help":
        if test_len(cmd_split, 1):
            print(" <train> to train model\n <test> to test model\n <save> saves net\n <load> loads net\n <class> take"
                  " image and classify it with the net")
    else:
        print(cmd_split[0], " is not recognized, please give correct input, <help> for help")
