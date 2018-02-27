import parser.parser as parser

csv_file = 'parser/cartrainingsetmini.csv'
root_dir = 'images/'
car_dataset = parser.CarDataset(csv_file, root_dir)

for i in range(len(car_dataset)):
    sample = car_dataset[i]
    sample[0].show()
