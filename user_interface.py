# this file contains the interface which serves as the driver for the rest of the software

import net_algorithms
import data
from testing import get_accuracy, get_accuracy_by_class
from train import train
import torch

pathtype=''
while True:
    cmd = input(">>>")
    if cmd == "quit":
        break
    elif cmd == "train":
        transform = data.transformation()
        net = net_algorithms.Net()
        criterion = net_algorithms.loss()
        optimizer = net_algorithms.optimizer(net)
        train(net, data.get_data(transform,pathtype), optimizer, criterion)
    elif cmd == "test":
        get_accuracy(data.get_data(transform), net, images)
        get_accuracy_by_class(data.get_data(transform,pathtype), net, images, data.classes())
    elif cmd == "save":
        torch.save(net,"classifier.pt")
    elif cmd == "load":
        net=torch.load("classifier.pt")
    elif cmd=="traini":
        transform = data.transform2()
        net = net_algorithms.TransferNet()
        criterion = net_algorithms.loss()
        optimizer = net_algorithms.optimizer(net)
        dataiter = iter(data.get_data(transform,pathtype))
        images, labels = dataiter.next()
        train(net, data.get_data(transform), optimizer, criterion)
    elif cmd=='pathtype':
        inputpath=input('please input path type:')
        if(inputpath=='r' or inputpath=='a'):
            pathtype= inputpath
        else:
            print("incorrect input for path")
    elif cmd == "help":
        print("<train> to train model, <test> to test model")
    else:
        print("incorrect input please give correct input, <help> for help")
