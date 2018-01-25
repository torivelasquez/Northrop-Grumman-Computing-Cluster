import netDefine as netDefine
import data as data
from testing import getaccuracy
from testing import getaccuracybyclass
from train import train
import torch

while True:
    cmd = input(">>>")
    #print(cmd)
    if cmd == "quit":
        break
    elif cmd == "train":
        transform = data.transformation()
        net = netDefine.Net()
        criterion = netDefine.loss()
        optimizer = netDefine.optimizer(net)
        #dataiter = iter(data.getData(transform))
        #images, labels = dataiter.next()
        # print(transform)
        train(net, data.getData(transform), optimizer, criterion)
    elif cmd == "test":
         getaccuracy(data.getData(transform), net, images)
         getaccuracybyclass(data.getData(transform), net, images, data.classes())
    elif cmd == "save":
	    torch.save(net,"classifier.pt")
    elif cmd == "load":
        transform = data.transformation()
        dataiter = iter(data.testdata(transform))
        images, labels = dataiter.next()
        net=torch.load("classifier.pt")
    elif cmd == "help":
	    print("<train> to train model, <test> to test model")
    else:
        print("incorrect input please give correct input, <help> for help")
