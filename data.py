import torch
import torchvision
import torchvision.transforms as transforms

def transformation():
	return transforms.Compose(
	    [transforms.ToTensor(),
	     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def transform2():
	return transforms.Compose(
        [transforms.Resize((400,400)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
         #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
def traindata(transform):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    return torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


def testdata(transform):
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

def classes():
    return ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
