# Verification Testing for Deep Learning CNN Tools

As Tensorflow, Keras and Pytorch are complex beasts, to ensure the correct GPU installation of each is working correctly we perform the following simple tests.

All tested with Tensorflow, Keras and Pytorch and Python 3.x **on Linux**.

_Assumes that git tools are available on the command line or that similar tools are available to access git / download files._

---

## Test #1 - check tensorflow:

```
git clone https://github.com/tobybreckon/python-cnn-tools-test.git
cd python-cnn-tools-test
python3 ./tf-test.py
```
### Result #1:

Text output to console such that:

```
We are using tensorflow: T.T.T

We believe we have the following devices available:

[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: MMMM
???
]
[name: "/device:GPU:0"
device_type: "GPU"
memory_limit: MMMM
???
]


Testing tensorflow with CPU ....
[[22. 28.]
 [49. 64.]]
CPU computation success.

Testing tensorflow with GPU ....
[[22. 28.]
 [49. 64.]]
GPU computation success.

We are using numpy: ???
We are using matplotlib: ???
.. and this is in Python: PPP

```
...  where T.T.T >= 1.9.x; MMM > 0; PPP > 3.x; ??? = (doesn't matter)

---

## Test #2 - check Keras


```
.. (as per test 1 for steps 1 + 2)
python3 ./keras-test.py
```

### Result #2:

Text output to console such that:

```
Using TensorFlow backend.
We are using keras: K.K.K

We are using the following keras backend:
tensorflow


We are using numpy: ???
We are using matplotlib: ???
.. and this is in Python: PPP

```
... where K.K.K >= 2.2.x (or higher); PPP > 3.x; ??? = (doesn't matter)

---

## Test #3 - check Pytorch

```
.. (as per test 1 for steps 1 + 2)
python3 ./pytorch-test.py
```

### Result #3:

Text output to console such that:

```
We are using pytorch: PT.PT.PT
We believe we have the following # of GPU:
1
The first GPU available is:
<DEVICE STRING NAME>

We are using numpy: ???
We are using matplotlib: ???
.. and this is in Python: PPP

```
... where PT.PT.PT >= 0.x.x (or higher); "DEVICE STRING NAME" looks sensible given the GPU in the machine; PPP > 3.x; ??? = (doesn't matter)

---

## Test #4 - TensorFlow and OpenCV full system check - low performance

* this tests we can use basic OpenCV (hopefully 3.x) with TensorFlow in the same python script

```
git clone https://github.com/tobybreckon/fire-detection-cnn.git
cd fire-detection-cnn
sh ./download-models.sh
python3 firenet.py models/test.mp4

```

### Result #4:
- model download (step 3) will depend on network speed
- video displayed in real-time, is not slow or jerky, and appropriate  Red Fire/Green Clear labels displayed depending on contents of image frame

---

## Test #5 - TensorFlow and OpenCV full system check - high performance

* this tests we can use advanced extra module functionality within OpenCV (hopefully 3.x) with TensorFlow in the same python script

```
(as per steps 1-3 of Test 4)
python3 superpixel-inceptionV1-OnFire.py models/test.mp4

```

### Result #5:
- video displayed in real-time, is not slow or jerky **with update of several frames per second observed**, and appropriate Red/Green labels displayed depending on contents of  (where Green is fire regions)
