import csv
import os
from skimage import io
from torch.utils.data import Dataset
from PIL import Image
import parser.car as car

# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class CarDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.car_dict = {}
        with open(csv_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = 0
            for row in reader:
                self.car_dict[i] = car.Car(row[1], row[2])
                i += 1
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_dict)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.car_dict[index]["img_name"])
        #image = io.imread(img_name)
        image = Image.open(img_name)
        style = self.car_dict[index]["style"]
        if self.transform:
            image = self.transform(image)
        #sample = {'image': image, 'style': style}
        sample = (image, style)

        return sample
