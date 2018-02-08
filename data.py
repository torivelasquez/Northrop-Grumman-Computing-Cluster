# data
# pathing to files
# absolute path: path from home
# relative path: path starting at the base file directory of northrop_Grumman_computing_cluster directory
#
# functions:
#   transformation(): tranform of data into tensor with the pytorch example
#   transform2(): transform of the data to work with fine tuned alexnet (32 by 32 image size causes an error as the
#   image size becomes {1,1,n} which breaks maxpooling)

import torch
import torchvision
import torchvision.transforms as transforms
import parser.parser as parser


def transformation():
    return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def transform2():
    return transforms.Compose(
        [transforms.Resize((400, 400)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_data(transform, path_type='r'):
    if path_type == 'a':
        csv_file = '/home/trocket/PycharmProjects/parser/cartrainingsetmini.csv'
        root_dir = '/home/trocket/images/'
    else:
        csv_file = 'parser/cartrainingsetmini.csv'
        root_dir = 'images/'
    car_dataset = parser.CarDataset(csv_file, root_dir, transform)
    return torch.utils.data.DataLoader(car_dataset, batch_size=4, shuffle=True, num_workers=2)


def classes():
    return ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
