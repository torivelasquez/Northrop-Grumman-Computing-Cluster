from torch.autograd import Variable
import netDefine as netDefine
import data as data
import train as train
from testing import getaccuracy
from testing import getaccuracybyclass
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == "__main__":
	transform = data.transformation()
	net = netDefine.Net()
	criterion = netDefine.loss()
	optimizer = netDefine.optimizer(net)
	dataiter = iter(data.traindata(transform))
	images, labels = dataiter.next()
	#print(transform)
	train.train(net,data.traindata(transform))
	getaccuracy(data.testdata(transform),net,images)
	getaccuracybyclass(data.testdata(transform),net,images,classes)
