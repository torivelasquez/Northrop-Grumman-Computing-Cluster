# this file contains a class that holds and manages updating of global paramaters

import os.path

"""
this class is used to store paramaters the software needs to decide how to execute. 
"""


class Parameters:
    def __init__(self):
        self.net_type = ["transfer"]
        self.train_data_loc = ["parser/carsetmini.csv"]
        self.test_data_loc = ["parser/carsetmini.csv"]
        self.images_loc = ["images/"]
        self.plots_loc = ["plots/"]
        self.save_loc = ["classifier.pt"]
        self.load_loc = ["classifier.pt"]
        self.epochs = [1]
        self.layers = [1]
        self.momentum = [0.9]
        self.learning_rate = [0.001]
        self.criterion = ["crossentropy"]
        self.optimizer = ["sgd"]
        self.train_transform = ["main"]
        self.test_transform = ["main"]
        self.grayscale = [False]
        self.record = [False]
        self.record_location = ["results.csv"]
        # set_map is a mapping of the name of a parameter that parameter corresponding setter method
        self.set_map = {"file": self.read_file,
                        "net_type": self.set_net_type,
                        "train_data_loc": self.set_train_data_loc,
                        "test_data_loc": self.set_test_data_loc,
                        "images_loc": self.set_images_loc,
                        "plots_loc" : self.set_plots_loc,
                        "save_loc": self.set_save_loc,
                        "load_loc": self.set_load_loc,
                        "epochs": self.set_epochs,
                        "layers": self.set_layers,
                        "momentum": self.set_momentum,
                        "learning_rate": self.set_learning_rate,
                        "criterion": self.set_criterion,
                        "optimizer": self.set_optimizer,
                        "train_transform": self.set_train_transform,
                        "test_transform": self.set_test_transform,
                        "grayscale": self.set_grayscale,
                        "record": self.set_record,
                        "record_location": self.set_record_location}
    """
    The list() function returns a list containing all the parameters tracked by the parameters object
    """
    def list(self):
        return [self.net_type, self.train_data_loc, self.test_data_loc, self.images_loc, self.plots_loc, self.save_loc,
                self.load_loc, self.epochs, self.layers, self.momentum, self.learning_rate, self.criterion,
                self.optimizer, self.train_transform, self.test_transform, self.grayscale, self.record,
                self.record_location]

    """
    set() takes a string parameter name and a new variable value and uses the set map to pass the new value to the
     appropriate setter method
    """
    def set(self, param, new_variable):
        if param in self.set_map:
            self.set_map[param](new_variable.split())
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

    def set_plots_loc(self, new_variable):
        self.plots_loc = new_variable

    def set_save_loc(self, new_variable):
        self.save_loc = new_variable

    def set_load_loc(self, new_variable):
        self.load_loc = new_variable

    def set_epochs(self, new_variable):
        self.epochs = [int(var) for var in new_variable]

    def to_int_if_int(self, string):
        try:
            return int(string)
        except ValueError:
            return string

    def set_layers(self, new_variable):
        self.layers = []
        for var in new_variable:
            new_var = []
            new_var += [self.to_int_if_int(layer) for layer in var.split(',')]
            self.layers.append(new_var)

    def set_momentum(self, new_variable):
        self.momentum = [float(var) for var in new_variable]

    def set_learning_rate(self, new_variable):
        self.learning_rate = [float(var) for var in new_variable]

    def set_criterion(self, new_variable):
        self.criterion = new_variable

    def set_optimizer(self, new_variable):
        self.optimizer = new_variable

    def set_train_transform(self, new_variable):
        self.train_transform = new_variable

    def set_test_transform(self, new_variable):
        self.test_transform = new_variable

    def string_to_bool(self, string):
        if string in ["Yes", "yes", "True", "true", "1"]:
            return True
        elif string in ["No", "no", "False", "false", "0"]:
            return False

    def set_grayscale(self, new_variable):
        self.grayscale = [self.string_to_bool(val) for val in new_variable]

    def set_record(self, new_variable):
        self.record = [self.string_to_bool(val) for val in new_variable]

    def set_record_location(self, new_variable):
        self.record_location = new_variable

    def get_net_type(self):
        return self.net_type

    def get_train_data_loc(self):
        return self.train_data_loc

    def get_test_data_loc(self):
        return self.test_data_loc

    def get_images_loc(self):
        return self.images_loc

    def get_plots_loc(self):
        return self.plots_loc

    def get_save_loc(self):
        return self.save_loc

    def get_load_loc(self):
        return self.load_loc

    def get_epochs(self):
        return self.epochs

    def get_layers(self):
        return self.layers

    def get_momentum(self):
        return self.momentum

    def get_learning_rate(self):
        return self.learning_rate

    def get_criterion(self):
        return self.criterion

    def get_optimizer(self):
        return self.optimizer

    def get_train_transform(self):
        return self.train_transform

    def get_test_transform(self):
        return self.test_transform

    """
    read_file parsers a file for parameters and values and passes them to the set method.
    """
    def read_file(self, file_name):
        file_name = file_name[0]
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


"""
temp parameters holds the same parameter structure as a paramaters object, but can be instantiated by passing it a list
of the parameters.
"""


class TempParams(Parameters):
    def __init__(self, input):
        super().__init__()
        self.net_type = [input[0]]
        self.train_data_loc = [input[1]]
        self.test_data_loc = [input[2]]
        self.images_loc = [input[3]]
        self.plots_loc = [input[4]]
        self.save_loc = [input[5]]
        self.load_loc = [input[6]]
        self.epochs = [input[7]]
        self.layers = [input[8]]
        self.momentum = [input[9]]
        self.learning_rate = [input[10]]
        self.criterion = [input[11]]
        self.optimizer = [input[12]]
        self.train_transform = [input[13]]
        self.test_transform = [input[14]]
        self.grayscale = [input[15]]
        self.record = [input[16]]
        self.record_location = [input[17]]
