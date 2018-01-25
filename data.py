import torch
import torchvision
import torchvision.transforms as transforms
import parser


def transformation():
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def getData(transform):
    csv_file = '/home/trocket/PycharmProjects/parser/cartrainingsetmini.csv'
    root_dir = '/home/trocket/images/'
    car_dataset = parser.CarDataset(csv_file, root_dir)
    return torch.utils.data.DataLoader(car_dataset, batch_size=4, shuffle=True, num_workers=2)


def testdata(transform):
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

def classes():
    return ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
