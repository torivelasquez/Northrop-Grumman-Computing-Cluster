# this file contains functions for the purpose of visualizing information to the user
# functions:
#   im_show(): displays an image


import matplotlib.pyplot as plt
import numpy as np


def im_show(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))