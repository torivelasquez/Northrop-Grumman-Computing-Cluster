class Car:

    def __init__(self, style, iname):
        self.style = style
        self.iname = iname

    def __getitem__(self, index):
        if index == "style":
            return self.style
        elif index == "img_name":
            return self.iname
        else:
            raise ValueError("%s is not defined" % index)
