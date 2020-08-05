#####################################################################

# Example : test if pytorch environment is working

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2018 Toby Breckon Durham University, UK

# License : MIT

#####################################################################

import torch
import numpy as np
import sys
import matplotlib

#####################################################################

print("We are using pytorch: " + torch.__version__)
print("We believe we have the following # of GPU:")
print(torch.cuda.device_count())
print()
if (torch.cuda.device_count() > 0):
    print("The first GPU available is:")
    print(torch.cuda.get_device_name(0))
    print()

#####################################################################

from torch.autograd import Variable
from torchvision import models

print ("Testing pytorch with CPU ....")

# BEGIN CPU TEST

alexnet = models.alexnet(pretrained=True)
x = Variable(torch.randn(1, 3, 227, 227))
y = alexnet(x)
print ("CPU computation *** success ***.")
print()


print ("Testing pytorch with GPU ....")

try:
    # BEGIN GPU TEST
    x = x.cuda()
    alexnet = alexnet.cuda()
    y = alexnet(x) #<--------- potential GPU FAIL here
    print ("GPU computation *** success ***.")
    print()
except:
    print ("GPU computation *** FAILURE ***.")
    print()



#####################################################################

# check other stuff

print("We are using numpy: " + np.__version__)
print("We are using matplotlib: " + matplotlib.__version__)
print(".. and this is in Python: " + sys.version)

#####################################################################
