import csv
import os
from skimage import io
from torch.utils.data import Dataset
from PIL import Image
import parser.car as car
import torch
import numpy as np


# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class CarDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.car_dict = {}
        self.classes = []
        with open(csv_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = 0
            for row in reader:
                if not row[1] in self.classes:
                    self.classes.append(row[1])
                self.car_dict[i] = car.Car(self.classes.index(row[1]), row[2])
                i += 1
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_dict)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.car_dict[index]["img_name"])
        # image = io.imread(img_name)
        image = Image.open(img_name)
        style = self.car_dict[index]["style"]
        # numstyle=index
        # print(style)
        if self.transform:
            image = self.transform(image)
        # sample = {'image': image, 'style': style}
        sample = (image, int(style)) # needed to covert '1' to 1
        # print(sample)
        return sample

    def get_classes(self):
        return self.classes
