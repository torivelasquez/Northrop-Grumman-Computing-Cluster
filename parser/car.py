class Car:

    def __init__(self, style, iname, bbox):
        self.style = style
        self.iname = iname
        self.bbox = bbox

    def __getitem__(self, index):
        if index == "style":
            return self.style
        elif index == "img_name":
            return self.iname
        elif index == "bbox":
            return self.bbox
        else:
            raise ValueError("%s is not defined" % index)
