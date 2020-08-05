#####################################################################

# Example : test if pytorch environment is working in injunction with OpenCV

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2019 Toby Breckon Durham University, UK

# License : MIT

#####################################################################

import torch
import cv2
import numpy as np
import struct
import sys
import matplotlib

#####################################################################


# check if the OpenCV we are using has the extra modules available

def extraOpenCVModulesPresent():
    (is_built, not_built) = cv2.getBuildInformation().split("Disabled:")
    return ('xfeatures2d' in is_built)

def nonFreeAlgorithmsPresent():
    (before, after) = cv2.getBuildInformation().split("Non-free algorithms:")
    output_list = after.split("\n")
    return ('YES' in output_list[0])

#####################################################################

# check pytorch

print("We are using pytorch: " + torch.__version__)
print("We believe we have the following # of GPU:")
print(torch.cuda.device_count())
print()
print("The first GPU available is:")
print(torch.cuda.get_device_name(0))
print()

# check OpenCV and other stuff

print("We are using OpenCV: " + cv2.__version__)
print(".. do we have the OpenCV Contrib Modules: " + str(extraOpenCVModulesPresent()))
print(".. do we have the OpenCV Non-free algorithms: " + str(nonFreeAlgorithmsPresent()))
print("We are using numpy: " + np.__version__)
print("We are using matplotlib: " + matplotlib.__version__)
print(".. and this is in Python: " + sys.version + " (" + str(struct.calcsize("P") * 8) + " bit)")

#####################################################################
