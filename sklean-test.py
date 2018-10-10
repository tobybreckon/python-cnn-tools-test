#####################################################################

# Example : test if sci-kit-learn is available

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2018 Toby Breckon Durham University, UK

# License : MIT

#####################################################################

import sklearn
import numpy as np
import matplotlib
import sys

#####################################################################

print("scikit-learn is available - version: " + sklearn.__version__);
print();

#####################################################################

# check other stuff

print("We are using numpy: " + np.__version__);
print("We are using matplotlib: " + matplotlib.__version__);
print(".. and this is in Python: " + sys.version)

#####################################################################
