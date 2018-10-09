#####################################################################

# Example : test if keras environment is working

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2018 Toby Breckon Durham University, UK

# License : MIT

#####################################################################

import tensorflow as tf
import keras

import numpy as np
import sys
import matplotlib

#####################################################################

print("We are using keras: " + keras.__version__);
print();

#####################################################################

print("We are using the following keras backend:");
print(keras.backend.backend());
print();

###############################################print();######################

# check other stuff

print("We are using numpy: " + np.__version__);
print("We are using matplotlib: " + matplotlib.__version__);
print(".. and this is in Python: " + sys.version)

#####################################################################
