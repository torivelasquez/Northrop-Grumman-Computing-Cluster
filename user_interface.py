# this file contains the interface which serves as the driver for the rest of the software

import sys
import csv
import os.path
import datetime
import runtime_parameters
import net_algorithms
import parser.parser as parser
import transformations
import itertools
from data_spliter import data_spliter
from testing import auc_confidence_interval,MAUCscore, auc_metric, roc_curve, get_accuracy, get_accuracy_by_class, classify, compute_confusion_matrix, mcc_score, multi_class_simplify_to_binary, get_mcc_by_class
from train import train
import torch


def len_test(cmd_split, num):
    if len(cmd_split) == num:
        return True
    print(cmd_split[0], " expects ", num, " arguments ", len(cmd_split), " given. Use <help> for help")
    return False

"""
train macro performs the neccesary steps to train a net given a paramaters object params_t. it then returns the trained net.
"""
def train_macro(params_t):
    try:
        transform = transformations.get_transform(params_t.train_transform[0])
        data_set, classes = parser.get_data(transform, params_t.images_loc[0], params_t.train_data_loc[0], params_t.grayscale[0])
        net_t = net_algorithms.get_net(params_t.net_type[0], len(classes), params_t.layers[0])
        if torch.cuda.device_count() > 1:
            net_t = torch.nn.DataParallel(net_t)
        if torch.cuda.is_available():
            net_t.cuda()
        criterion = net_algorithms.get_criterion(params_t.criterion[0])
        optimizer = net_algorithms.get_optimizer(params_t.optimizer[0], net_t, params_t.learning_rate[0], params_t.momentum[0])
        train(net_t, data_set, optimizer, criterion, params_t.epochs[0])
        return net_t
    except Exception as e:
        print("Error: ", e)

"""
test_macro performs the neccesary steps to test a net net_t given a paramaters object params_t.
It outputs the results of the test to the terminal, and records the results as specified in the params _t object
"""
def test_macro(net_t, params_t):
    try:
        transform = transformations.get_transform(params_t.test_transform[0])
        data_set, classes = parser.get_data(transform, params_t.images_loc[0], params_t.test_data_loc[0], params_t.grayscale[0])
        confusion_matrix, statistics, labels, score = compute_confusion_matrix(data_set, net_t, classes)
        acc = get_accuracy(confusion_matrix, classes)
        acclist = get_accuracy_by_class(confusion_matrix, classes)
        auc_values = auc_metric(score, labels, classes)
        mauc = MAUCscore(score, labels, classes)
        confidence_intervals = auc_confidence_interval(score, labels, classes)
        confidence_intervals = [[str(s) for s in sub] for sub in confidence_intervals]
        confidence_intervals = [':'.join(sub) for sub in confidence_intervals]
        overall_stats = list(statistics['overall'].items())
        overall_stats = [tup for tup in overall_stats if tup[1] != "ToDo"]
        keys, values = map(list, zip(*overall_stats))
        class_stats = statistics['class']
        class_stats_values = class_stats.values.tolist()
        class_stats_values = [[str(s) for s in sub] for sub in class_stats_values]
        class_stats_values = [';'.join(s) for s in class_stats_values]

        if params_t.record[0]:
            roc_curve(score, labels, classes, params_t.plots_loc[0])

            file_exists = os.path.isfile(params_t.record_location[0])
            dir_path = os.path.dirname(params_t.record_location[0])
            if dir_path != '':
                os.makedirs(os.path.dirname(params_t.record_location[0]), exist_ok=True)
            result_file = open(params_t.record_location[0], 'a')
            writer = csv.writer(result_file, delimiter=',')

            if not file_exists:
                csv_header = ["Timestamp", "Net Type", "Train Data Location",
                              "Test Data Location", "Images Location", "Plots Location",
                              "Save Location", "Load Location", "Epochs", "Layers",
                              "Momentum", "Learning Rate", "Criterion", "Optimizer",
                              "Train Transform", "Test Transform", "Grayscale",
                              "Overall Accuracy", "MAUC Score", "Classes",
                              "Class Accuracies", "Class AUC Scores", "Class 95% CIs"]

                csv_header += keys[2:] + list(class_stats.index) + ["Confusion Matrix", '\n']
                writer.writerow(csv_header)

            output = []

            # Add timestamp to output
            runtime = datetime.datetime.now().replace(microsecond=0).isoformat()
            output.append(runtime)

            # Add net parameters to outputs
            param_keys = list(params_t.set_map.keys())[1:]
            layers_idx = param_keys.index("layers")
            out_params = [i[0] if len(i) == 1 else i for i in params_t.list()][:-2]
            out_params[layers_idx] = list(map(str, out_params[layers_idx]))
            out_params[layers_idx] = ':'.join(out_params[layers_idx])
            output += out_params

            # Add test results and metrics to output
            output.extend((str(acc), str(mauc)))
            output += [';'.join(classes)]
            output += [';'.join(list(map(str, acclist)))]
            output += [';'.join(list(map(str, auc_values)))]
            output += [';'.join(list(map(str, confidence_intervals)))]
            output += values[2:]
            output += class_stats_values
            cm_list = confusion_matrix.tolist()
            cm_list = [[str(s) for s in sub] for sub in cm_list]
            cm_list = [';'.join([':'.join(s) for s in cm_list])]
            output += cm_list
            output.append('\n')

            writer.writerow(output)
            result_file.close()
    except Exception as e:
        print("Error: ", e)

"""
The main user interface accepts a command from the user and matched the command to the first word in the input.
It then checks if the input command has the correct number of paramaters specified and aborts the command execution
if it does not.
"""
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
                try:
                    transform = transformations.get_transform(params.test_transform[0])
                    data_set, classes = parser.get_data(transform, params.images_loc[0], params.train_data_loc[0], params.grayscale[0])
                    classify(params.images_loc[0] + cmd_split[1], net, transform, classes)
                except Exception as e:
                    print("Error: ", e)
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
                net = net_algorithms.get_net(params.net_type[0], len(classes),params.layers[0])
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

    elif cmd_split[0] == "view":
        for i in range(len(params.list())):
            print(params.list()[i])

    elif cmd_split[0] == "help":
        if len_test(cmd_split, 1):
            print(" <train> to train model\n <test> to test model\n <save> saves net\n <load> loads net\n <class> take"
                  " image and classify it with the net \n <settings> input file to configure the file and hyperparameters")

    else:
        print(cmd_split[0], " is not recognized, please give correct input, <help> for help")
