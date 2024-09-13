#####################################################################

# Example : test if tensorflow environment is working

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2018 Toby Breckon Durham University, UK

# License : MIT

#####################################################################

from tensorflow.python.client import device_lib
import tensorflow as tf
import numpy as np
import sys
import matplotlib

#####################################################################

print("We are using tensorflow: " + tf.__version__)
print()

#####################################################################

print("We believe we have the following devices available:")
print()

print(device_lib.list_local_devices())
print()

#####################################################################

# test CPU first

print("Testing tensorflow with CPU ....")

with tf.device('/device:CPU:0'):
    g = tf.Graph()
    with g.as_default():

        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

        with tf.compat.v1.Session() as sess:
            print(sess.run(c))
            print("CPU computation *** success ***.")
            print()

#####################################################################

# test GPU next

print("Testing tensorflow with GPU ....")

gpus = tf.config.experimental.list_physical_devices('GPU')

try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    with tf.device('/device:GPU:0'):
        g = tf.Graph()
        with g.as_default():
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                            shape=[2, 3], name='a')
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                            shape=[3, 2], name='b')
            c = tf.matmul(a, b)

            with tf.compat.v1.Session() as sess:
                print(sess.run(c))
                print("GPU computation *** success ***.")
except BaseException:
    print("GPU computation *** FAILURE ***.")
    print()


#####################################################################

# check OpenCV and other stuff

print()
print("We are using OpenCV: " + cv2.__version__)
print(".. do we have the OpenCV Contrib Modules: " +
      str(extra_opencv_modules_present()))
try:
    print(".. do we have the OpenCV Non-free algorithms: " +
          str(non_free_opencv_algorithms_present()))
except BaseException:
    print(".. OpenCV version pre-dates (or does not have) non-free algorithms module")

print("We are using numpy: " + np.__version__)
print("We are using matplotlib: " + matplotlib.__version__)
print(".. and this is in Python: " + sys.version +
      " (" + str(struct.calcsize("P") * 8) + " bit)")

#####################################################################

print()
print("Check Video I/O (OS identifier: " + sys.platform + ")")
print("... available camera backends: ", end='')
for backend in cv2.videoio_registry.getCameraBackends():
    print(" " + cv2.videoio_registry.getBackendName(backend), end='')
print()
print("... available stream backends: ", end='')
for backend in cv2.videoio_registry.getStreamBackends():
    print(" " + cv2.videoio_registry.getBackendName(backend), end='')
print()
print("... available video writer backends: ", end='')
for backend in cv2.videoio_registry.getWriterBackends():
    print(" " + cv2.videoio_registry.getBackendName(backend), end='')
print()
print()

#####################################################################

# credit to: https://tinyurl.com/y529vzc3

print("Available Cuda Information: ")
cuda_info = [re.sub('\\s+', ' ', ci.strip()) for ci in
             cv2.getBuildInformation().strip().split('\n')
             if len(ci) > 0 and re.search(r'(nvidia*:?)|(cuda*:)|(cudnn*:)',
                                          ci.lower()) is not None]
print("... " + str(cuda_info))
print()

print("OpenCL available (within OpenCV) ? : " + str(cv2.ocl.haveOpenCL()))
print()

###############################################print()######################

# check other stuff

print("We are using numpy: " + np.__version__)
print("We are using matplotlib: " + matplotlib.__version__)
print(".. and this is in Python: " + sys.version)

#####################################################################

