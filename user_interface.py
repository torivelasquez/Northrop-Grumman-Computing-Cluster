# this file contains the interface which serves as the driver for the rest of the software

import net_algorithms
import data
from testing import get_accuracy, get_accuracy_by_class, classify
from train import train
import torch

pathtype = ''
while True:
    cmd = input(">>>")
    if cmd == "quit":
        break
    elif cmd == "train":
        transform = data.transform2()
        data_set, classes = data.get_data(transform, pathtype)
        net = net_algorithms.Net(len(classes))
        criterion = net_algorithms.loss()
        optimizer = net_algorithms.optimizer(net)
        train(net, data_set, optimizer, criterion)
    elif cmd == "test":
        transform = data.transform2()
        data_set, classes = data.get_data(transform, pathtype)
        get_accuracy(data_set, net)
        get_accuracy_by_class(data_set, net, classes)
    elif cmd == "class":
        transform = data.transform2()
        data_set, classes = data.get_data(transform, pathtype)
        cmd = input("file:")
        classify(cmd, net, transform, classes)
    elif cmd == "save":
        torch.save(net, "classifier.pt")
    elif cmd == "load":
        net = torch.load("classifier.pt")
    elif cmd == "traini":
        transform = data.transform2()
        net = net_algorithms.TransferNet()
        criterion = net_algorithms.loss()
        optimizer = net_algorithms.optimizer(net)
        dataiter = iter(data.get_data(transform, pathtype))
        images, labels = dataiter.next()
        train(net, data.get_data(transform), optimizer, criterion)
    elif cmd == 'pathtype':
        inputpath = input('please input path type:')
        if inputpath == 'r' or inputpath == 'a':
            pathtype = inputpath
        else:
            print("incorrect input for path")
    elif cmd == "help":
        print("<train> to train model, <test> to test model")
    else:
        print("incorrect input please give correct input, <help> for help")
