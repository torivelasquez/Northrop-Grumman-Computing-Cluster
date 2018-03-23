# this file contains a class that holds and manages updating of global paramaters

import os.path

class Parameters:
    def __init__(self):
        self.net_type = "transfer"
        self.train_data_loc = "parser/cartrainingsetmini.csv"
        self.test_data_loc = "parser/cartrainingsetmini.csv"
        self.images_loc = "images/"
        self.save_loc = "classifier.pt"
        self.load_loc = "classifier.pt"
        self.epochs = 1
        self.criterion = "crossentropy"
        self.optimizer = "sgd"
        self.train_transform = "main"
        self.test_transform = "main"
        self.grayscale = False
        self.set_map = {"file": self.read_file, "net_type": self.set_net_type,
                        "train_data_loc": self.set_train_data_loc,
                        "test_data_loc": self.set_test_data_loc, "images_loc": self.set_images_loc,
                        "save_loc": self.set_save_loc, "load_loc": self.set_load_loc, "epochs": self.set_epochs,
                        "criterion": self.set_criterion, "optimizer": self.set_optimizer,
                        "train_transform": self.set_train_transform, "test_transform": self.set_test_transform,
                        "grayscale": self.set_grayscale}

    def set(self, param, new_variable):
        if param in self.set_map:
            self.set_map[param](new_variable)
        else:
            print(param, " is not a recognized parameter")

    def set_net_type(self, new_variable):
        self.net_type = new_variable

    def set_train_data_loc(self, new_variable):
        self.train_data_loc = new_variable

    def set_test_data_loc(self, new_variable):
        self.test_data_loc = new_variable

    def set_images_loc(self, new_variable):
        self.images_loc = new_variable

    def set_save_loc(self, new_variable):
        self.save_loc = new_variable

    def set_load_loc(self, new_variable):
        self.load_loc = new_variable

    def set_epochs(self, new_variable):
        if ' ' not in new_variable:
            self.epochs = int(new_variable)
        else:
            self.epochs = new_variable

    def set_criterion(self, new_variable):
        self.criterion = new_variable

    def set_optimizer(self, new_variable):
        self.optimizer = new_variable

    def set_train_transform(self, new_variable):
        self.train_transform = new_variable

    def set_test_transform(self, new_variable):
        self.test_transform = new_variable

    def set_grayscale(self, new_variable):
        if new_variable in ["Yes", "yes", "True", "true", "1"]:
            self.grayscale = True
        elif new_variable in ["No", "no", "False", "false", "0"]:
            self.grayscale = False


    def get_net_type(self):
        return self.net_type

    def get_train_data_loc(self):
        return self.train_data_loc

    def get_test_data_loc(self):
        return self.test_data_loc

    def get_images_loc(self):
        return self.images_loc

    def get_save_loc(self):
        return self.save_loc

    def get_load_loc(self):
        return self.load_loc

    def get_epochs(self):
        return self.epochs

    def get_criterion(self):
        return self.criterion

    def get_optimizer(self):
        return self.optimizer

    def get_train_transform(self):
        return self.train_transform

    def get_test_transform(self):
        return self.test_transform

    def read_file(self, file_name):
        if os.path.isfile(file_name):
            with open(file_name, "r") as file:
                print("Reading file: ", file_name)
                for line in file:
                    line = line.partition("#")[0]
                    line = line.rstrip().lstrip()
                    if not line == '':
                        line_p = line.partition("=")
                        if not line_p[2] == '' and line_p[2].count('=') == 0:
                            self.set(line_p[0].lstrip().rstrip(), line_p[2].lstrip().rstrip())
                        else:
                            print("not formatted correctly: ", line)
        else:
            print("settings file not found")
