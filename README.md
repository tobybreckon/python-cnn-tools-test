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

- Text output to console such that:

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
- such that T.T.T >= 1.9.x; MMM > 0; PPP > 3.x; ??? = (doesn't matter)

---

## Test #2 - test Keras


```
.. (as per test 1 for steps 1 + 2)
python3 ./keras-test.py
```

### Result #2:

- Text output to console such that:

```
Using TensorFlow backend.
We are using keras: K.K.K

We are using the following keras backend:
tensorflow


We are using numpy: ???
We are using matplotlib: ???
.. and this is in Python: PPP

```
- such that K.K.K >= 2.2.x (or higher); PPP > 3.x; ??? = (doesn't matter)

---

## Test #3 - test Pytorch

```
.. (as per test 1 for steps 1 + 2)
python3 ./pytorch-test.py
```

## Result #3:

- Text output to console such that:

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
- such that PT.PT.PT >= 0.x.x (or higher); <DEVICE STRING NAME> looks sensible given the GPU in the machine; PPP > 3.x; ??? = (doesn't matter)

---
