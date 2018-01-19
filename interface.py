import netDefine as netDefine
import data as data
from testing import getaccuracy
from testing import getaccuracybyclass
from train import train
import torch
import torchvision.models as models
initialmodel=models.alexnet(pretrained=True)
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
        dataiter = iter(data.traindata(transform))
        images, labels = dataiter.next()
        # print(transform)
        train(net, data.traindata(transform), optimizer, criterion)
    elif cmd == "trainbase":
        transform = data.transformation()
        net = netDefine.BaseNet()
        criterion = netDefine.loss()
        optimizer = netDefine.optimizer(net)
        dataiter = iter(data.traindata(transform))
        images, labels = dataiter.next()
        # print(transform)
        train(net, data.traindata(transform), optimizer, criterion)
    elif cmd == "test":
         getaccuracy(data.testdata(transform), net, images)
         getaccuracybyclass(data.testdata(transform), net, images, data.classes())
    elif cmd == "save":
	    torch.save(net,"classifier.pt")
    elif cmd == "load":
        transform = data.transformation()
        dataiter = iter(data.testdata(transform))
        images, labels = dataiter.next()
        net=torch.load("classifier.pt")
    elif cmd=="traini":
        transform = data.transform2()
        net = netDefine.TransferNet(initialmodel)
        criterion = netDefine.loss()
        optimizer = netDefine.optimizer(net)
        dataiter = iter(data.traindata(transform))
        images, labels = dataiter.next()
        # print(transform)
        train(net, data.traindata(transform), optimizer, criterion)
    #elif cmd=="testi":
       #transform = data.transform2()
       #dataiter = iter(data.testdata(transform))
       #images, labels = dataiter.next()
       #getaccuracy(data.testdata(transform), initialmodel, images)
       # getaccuracybyclass(data.testdata(transform), initmodel=initialmodel, images, data.classes())
    elif cmd == "help":
	    print("<train> to train model, <test> to test model")
    else:
        print("incorrect input please give correct input, <help> for help")