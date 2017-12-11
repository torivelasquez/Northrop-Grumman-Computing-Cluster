import csv
import car

def parser():
    new_dict = {}
    with open('cartrainingsetmini.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        for row in reader:
            new_dict[row[2]] = car.car(row[1])

    return new_dict

parser()