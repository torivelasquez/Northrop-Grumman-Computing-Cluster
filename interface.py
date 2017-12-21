import netDefine as netDefine
import data as data
from testing import getaccuracy
from testing import getaccuracybyclass
from train import train

while True:
    cmd = input(">>>")
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
    elif cmd == "test":
        getaccuracy(data.testdata(transform), net, images)
        getaccuracybyclass(data.testdata(transform), net, images, data.classes())