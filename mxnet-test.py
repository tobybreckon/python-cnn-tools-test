#####################################################################

# Example : test if mxnet environment is working

# Author : Bruna Maciel-Pearson

# Copyright (c) 2018 Bruna Maciel-Pearson / Toby Breckon
# Dept. Computer Science, Durham University, UK

# License : MIT

#####################################################################

import mxnet

import numpy as np
import sys
import matplotlib
import os

#####################################################################

print("We are using MXNet: " + mxnet.__version__)
print()
# list GPUs available
gpus = mxnet.test_utils.list_gpus()
if len(gpus) >= 1:
    print("MXNET has " + str(len(gpus)) + " GPUs available.")
    for i in gpus:
        print("The GPUs available are: ")
        print("- GPU number:", i)
else:
    print("MXNet says - sorry no GPU found")

print()

#####################################################################

# check other stuff

print("We are using numpy: " + np.__version__)
print("We are using matplotlib: " + matplotlib.__version__)
print(".. and this is in Python: " + sys.version)

#####################################################################
