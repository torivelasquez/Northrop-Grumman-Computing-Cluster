import csv
import os
from torch.utils.data import Dataset
from PIL import Image
import parser.car as car
import torch


def get_data(transform, img_path, csv_path, grayscale):
    car_dataset = CarDataset(csv_path, img_path, transform, grayscale)
    return torch.utils.data.DataLoader(car_dataset, batch_size=4, shuffle=True, num_workers=2), \
        car_dataset.get_classes()


class CarDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, grayscale = False):
        self.grayscale = grayscale
        self.car_list = []
        self.classes = []
        with open(csv_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if not row[1] in self.classes:
                    self.classes.append(row[1])
                self.car_list.append(car.Car(self.classes.index(row[1]), row[2],
                                             (int(row[3]), int(row[4]), int(row[5]), int(row[6]))))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_list)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.car_list[index]["img_name"])
        bbox = self.car_list[index]["bbox"]
        og_image = Image.open(img_name)
        image = og_image.crop(bbox)
        if self.grayscale:
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        style = self.car_list[index]["style"]
        if self.transform:
            image = self.transform(image)
        sample = (image, int(style)) # needed to covert '1' to 1
        return sample

    def get_classes(self):
        return self.classes
