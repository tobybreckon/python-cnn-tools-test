#####################################################################

# Example : test if tensorflow environment is working

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2018 Toby Breckon Durham University, UK

# License : MIT

#####################################################################

import tensorflow as tf
import numpy as np
import sys
import matplotlib

#####################################################################

print("We are using tensorflow: " + tf.__version__);
print();

#####################################################################

print("We believe we have the following devices available:");
print();

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print();

#####################################################################

# test CPU first

print ("Testing tensorflow with CPU ....")

with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

    with tf.Session() as sess:
        print (sess.run(c))
        print ("CPU computation success.");
        print();

#####################################################################

# test GPU next

print ("Testing tensorflow with GPU ....");

try:
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

        with tf.Session() as sess:
            print (sess.run(c))
            print ("GPU computation success.");
except:
        print ("GPU computation failure.")
        print();

###############################################print();######################

# check other stuff

print("We are using numpy: " + np.__version__);
print("We are using matplotlib: " + matplotlib.__version__);
print(".. and this is in Python: " + sys.version)

#####################################################################
