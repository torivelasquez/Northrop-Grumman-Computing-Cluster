import netDefine as netDefine
import data as data
from testing import get_accuracy
from testing import get_accuracy_by_class
from train import train
import torch

while True:
    cmd = input(">>>")
    if cmd == "quit":
        break
    elif cmd == "train":
        transform = data.transformation()
        net = netDefine.Net()
        criterion = netDefine.loss()
        optimizer = netDefine.optimizer(net)
        train(net, data.get_data(transform), optimizer, criterion)
    elif cmd == "test":
        get_accuracy(data.get_data(transform), net, images)
        get_accuracy_by_class(data.get_data(transform), net, images, data.classes())
    elif cmd == "save":
        torch.save(net,"classifier.pt")
    elif cmd == "load":
        net=torch.load("classifier.pt")
    elif cmd=="traini":
        transform = data.transform2()
        net = netDefine.TransferNet()
        criterion = netDefine.loss()
        optimizer = netDefine.optimizer(net)
        dataiter = iter(data.get_data(transform))
        images, labels = dataiter.next()
        train(net, data.get_data(transform), optimizer, criterion)
    elif cmd == "help":
        print("<train> to train model, <test> to test model")
    else:
        print("incorrect input please give correct input, <help> for help")
