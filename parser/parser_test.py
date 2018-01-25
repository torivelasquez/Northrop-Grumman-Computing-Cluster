import matplotlib.pyplot as plt
from skimage import data, io
import parser

csv_file = '/home/trocket/PycharmProjects/parser/cartrainingsetmini.csv'
root_dir = '/home/trocket/images/'
car_dataset = parser.CarDataset(csv_file, root_dir)

fig = plt.figure()
for i in range(len(car_dataset)):
    sample = car_dataset[i]
    io.imshow(sample["image"])
    plt.show()
