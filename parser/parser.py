import csv
import os
from torch.utils.data import Dataset
from PIL import Image
import parser.car as car
import torch


def get_data(transform, img_path, csv_path, grayscale):
    """
    :param transform: A transformation function.
    :param img_path: The actual or relative image directory path.
    :param csv_path: The actual or relative CSV file path.
    :param grayscale: Boolean value whether to convert 3 channel RGB format.
    :return: Dataset loader for the CarDataset class.
    """
    car_dataset = CarDataset(csv_path, img_path, transform, grayscale)
    return torch.utils.data.DataLoader(car_dataset, batch_size=4, shuffle=True, num_workers=2), \
        car_dataset.get_classes()


class CarDataset(Dataset):
    """
    An inherited Pytorch Dataset class. Stores images and specs into a database.
    """

    def __init__(self, csv_file, root_dir, transform=None, grayscale=False):
        """
        :param csv_file: The actual or relative CSV file path.
        :param root_dir: The actual or relative image directory path.
        :param transform: A transformation function.
        :param grayscale: Boolean value whether to convert to 3 channel RGB format.
        """
        self.grayscale = grayscale
        self.car_list = []
        self.classes = []
        with open(csv_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if not row[0] in self.classes:
                    self.classes.append(row[0])
                self.car_list.append(car.Car(self.classes.index(row[0]), row[1],
                                             (int(row[2]), int(row[3]), int(row[4]), int(row[5]))))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
        :return: The number of dataset entries.
        """
        return len(self.car_list)

    def __getitem__(self, index):
        """
        :param index: A numerical number of the desired dataset entry.
        :return: A transformed image and numerical class type of the corresponding image.
        """
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
        sample = (image, int(style))

        return sample

    def get_classes(self):
        """
        :return: All available classes as a list.
        """
        return self.classes
