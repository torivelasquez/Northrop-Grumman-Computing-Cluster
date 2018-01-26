import csv
import car
import os
from skimage import io
from torch.utils.data import Dataset


# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class CarDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.car_dict = {}
        with open(csv_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = 0
            for row in reader:
                self.car_dict[i] = car.car(row[1], row[2])
                i += 1
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_dict)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.car_dict[index]["img_name"])
        image = io.imread(img_name)
        style = self.car_dict[index]["style"]
        sample = {'image': image, 'style': style}
        if self.transform:
            sample = self.transform(sample)

        return sample
