class Car:
    """
    Subclass of entries in the CarDataset.
    """

    def __init__(self, style, iname, bbox):
        """
        :param style: Numerical class of vehicle.
        :param iname: Name of image.
        :param bbox: Bounding box values.
        """
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
