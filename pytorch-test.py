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

print("We are using pytorch: " + torch.__version__);
print("We believe we have the following # of GPU:");
print(torch.cuda.device_count());
print();
print("The first GPU available is:");
print(torch.cuda.get_device_name(0));
print();

#####################################################################

# check other stuff

print("We are using numpy: " + np.__version__);
print("We are using matplotlib: " + matplotlib.__version__);
print(".. and this is in Python: " + sys.version)

#####################################################################
