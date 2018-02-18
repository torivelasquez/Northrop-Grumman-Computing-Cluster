# this file contains a class that holds and manages updating of global paramaters


class Parameters:
    def __init__(self):
        self.dict = {}
        self.dict["net_type"] = "transfer"
        self.dict["train_data_loc"] = "parser/cartrainingsetmini.csv"
        self.dict["test_data_loc"] = "parser/cartrainingsetmini.csv"
        self.dict["images_loc"] = "images/"
        self.dict["save_loc"] = "classifier.pt"
        self.dict["load_loc"] = "classifier.pt"
        self.dict["epochs"] = 1
        self.dict["criterion"] = "crossentropy"
        self.dict["optimizer"] = "sgd"
        self.dict["train_transform"] = "main"
        self.dict["test_transform"] = "main"

    def set(self, param, new_variable):
        if param in self.dict:
            self.dict[param] = new_variable

    def get(self, param):
        if param in self.dict:
            return self.dict[param]
        return ""

    def read_file(self, file):
        pass
