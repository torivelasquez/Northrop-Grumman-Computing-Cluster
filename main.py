from torch.autograd import Variable
import netDefine as netDefine
import data as data
import train as train
from testing import get_accuracy
from testing import get_accuracy_by_class
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == "__main__":
    transform = data.transformation()
    net = netDefine.Net()
    criterion = netDefine.loss()
    optimizer = netDefine.optimizer(net)
    dataiter = iter(data.train_data(transform))
    images, labels = dataiter.next()
    train.train(net, data.train_data(transform))
    get_accuracy(data.test_data(transform), net, images)
    get_accuracy_by_class(data.test_data(transform), net, images, classes)
