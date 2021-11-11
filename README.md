# Verification Testing for Deep Learning CNN Tools

As Tensorflow, Keras and Pytorch are complex pieces of software, to ensure the GPU installation of each is working correctly we perform the following simple tests.

All tested with Tensorflow, Keras, Pytorch and Python 3.x (with OpenCV as needed) **on Linux**.

 * For **TensorFlow 1.x** only - use tests 1, 4 and 5 only.
 * For **TensorFlow 2.x** only - use tests 1 and 6 only.
 * For **Keras** (which uses TensorFlow as a backend) - use tests 1, 2, 4, 5 only.
 * For **PyTorch** only - use test 3 only (and 3a for testing with OpenCV also).
 * _See very simple test 1a for additionally testing sci-kit-learn is available in the same python environment_

N.B. Durham Students - ***if testing/using on the Durham University LDS (linux) system*** you need to first run ```tensorflowX.X.X.init```, ```pytorchX.X.X.init``` and ```opencvX.X.X.init``` in the (Linux, not the Python) command shell where X.X.X is the version number, or alternatively without it which should hopefully default to the latest version installed (e.g. `tensorflow.init```, ```pytorch.init```, ```opencv.init```), to add the relevant paths for these tools to the ```PYTHONPATH``` environment variable (and be sitting at a machine with a GPU in it!).

_Assumes that the git, wget, md5sum and curl tools are available on the command line or that similar tools are available to access git / download files._

**Tests 3a, 4 and 5 assume you have OpenCV aleady installed** (with the extra modules also for Test 5) - OpenCV has its own testing page and test suite here - https://github.com/tobybreckon/python-examples-ip/blob/master/TESTING.md

An additional test script for MxNet is also available (```mxnet-test.py```) but this is not currently included within the the set of software supported by the set of explicit tests below
(which is primarily used to test our on campus teaching lab provision of TensorFlow, Keras and Pytorch at [Durham University](https:/www.durham.ac.uk/)).

---

## Test #1 - check TensorFlow (1.x or 2.x):

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
???
[[22. 28.]
 [49. 64.]]
CPU computation *** success ***.

Testing tensorflow with GPU ....
???
[[22. 28.]
 [49. 64.]]
GPU computation *** success ***.

We are using numpy: ???
We are using matplotlib: ???
.. and this is in Python: PPP

```
...  where T.T.T >= 1.9.x; MMM > 0; PPP > 3.x; ??? = (doesn't matter)

---

## Test #1a - check Scikit-Learn toolkit

[this is also very useful to have for most machine learning]

```
.. (as per test 1 for steps 1 + 2 - no need to repeat if already completed)
python3 ./sklearn-test.py
```

### Result #1a:

Text output to console such that:

```
scikit-learn is available - version: S.S.S.

We are using numpy: ???
We are using matplotlib: ???
.. and this is in Python: PPP

```
... where S.S.S >= 0.20.x (or higher); PPP > 3.x; ??? = (doesn't matter)

---

## Test #2 - check Keras


```
.. (as per test 1 for steps 1 + 2 - no need to repeat if already completed)
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
.. (as per test 1 for steps 1 + 2 - no need to repeat if already completed)
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

Testing pytorch with CPU ....
CPU computation *** success ***.

Testing pytorch with GPU ....
GPU computation *** success ***.

We are using numpy: ???
We are using matplotlib: ???
.. and this is in Python: PPP

```
... where PT.PT.PT >= 1.x.x (or higher); "DEVICE STRING NAME" looks sensible given the GPU in the machine; PPP > 3.x; ??? = (doesn't matter)

---

## Test #3a - check Pytorch with OpenCV

```
.. (as per test 1 for steps 1 + 2 - no need to repeat if already completed)
python3 ./pytorch-opencv-test.py
```

### Result #3a:

Text output to console such that:

```
We are using pytorch: PT.PT.PT
We believe we have the following # of GPU:
1
The first GPU available is:
<DEVICE STRING NAME>

We are using OpenCV: CCC
.. do we have the OpenCV Contrib Modules: True
.. do we have the OpenCV Non-free algorithms: True
We are using numpy: <???>
We are using matplotlib: <???>
.. and this is in Python: PPP ??? (64 bit)

Check Video I/O (OS identifier: MMM)
... available camera backends:  LLL
... available stream backends:  LLL
... available video writer backends: LLL

Available Cuda Information:
... ['NVIDIA CUDA: YES (ver NNN, RRR)', 'NVIDIA GPU arch: ???', 'NVIDIA PTX archs: ZZZ']

OpenCL available (within OpenCV) ? : True

```
... where PT.PT.PT >= 1.x.x (or higher); "DEVICE STRING NAME" looks sensible given the GPU in the machine; CCC >= 4.3.x (or higher), PPP > 3.x, MMM is sensible for the OS in use, each of the LLL list are sensible (may not all be identical) and ideally include FFMPEG + GSTREAMER in addition to V4L/V4L (for MMM = linux..), QT (for MMM = darwin) or DSHOW / MSMF (for MMM = win..), NNN > 10.x, ZZZ includes ``cuDNN: Yes`` and ??? = (doesn't matter). In addition, for maximum performance RRR ideally includes ``CUFFT CUBLAS FAST_MATH``.

Ideally, the OpenCV install would also pass the full set of tests in [this document](https://github.com/tobybreckon/python-examples-ip/blob/master/TESTING.md).

[ to build with Non-free algorithms set OPENCV_ENABLE_NONFREE=TRUE in CMake ]

---

## Test #4 - TensorFlow 1.x and OpenCV full system check - low performance

* this tests we can use basic OpenCV (hopefully 4.x) with TensorFlow 1.x in the same python script

```
git clone https://github.com/tobybreckon/fire-detection-cnn.git
cd fire-detection-cnn
sh ./download-models.sh
python3 firenet.py models/test.mp4

```

### Result #4:

Text output to console such that:

```
???
Constructed FireNet ...
???
Loaded CNN network weights ...
Loaded video ..
```

... ??? = (detail doesn't matter but should clearly indicate GPU usage by type/name/bus or similar)

Then:

- model download (step 3) will depend on network speed
- **video displayed in real-time, is not slow or jerky,** and appropriate  Red Fire/Green Clear labels displayed depending on contents of image frame
- may need to resize window to display full image correctly. Or press "f" for fullscreen mode.
- Press "x" to exit

---

## Test #5 - TensorFlow 1.x and OpenCV full system check - high performance

* this tests we can use advanced extra module functionality within OpenCV (hopefully 4.x) with TensorFlow 1.x in the same python script

```
(as per steps 1-3 of Test 4 - no need to repeat if already completed)
python3 superpixel-inceptionV1OnFire.py models/test.mp4

```

### Result #5:

Text output to console such that:

```
???
Constructed SP-InceptionV1-OnFire ...
???
Loaded CNN network weights ...
Loaded video ..
```

... ??? = (detail doesn't matter but should clearly indicate GPU usage by type/name/bus or similar)

Then:

- video displayed in real-time, is not slow or jerky **with update of several frames per second observed**, and appropriate Red/Green labels displayed depending on contents of  (where Green is fire regions)
- Press "x" to exit

---

## Test #6 - TensorFlow 2.x and OpenCV full system check

* this tests we can use a OpenCV (hopefully 4.x) with TensorFlow 2.x in the same python script

```
wget -q https://raw.githubusercontent.com/SIlvaMFPedro/pyimagesearch/3e5c922b5f905078322d2283d704ef8875f043e0/region-proposal-object-detection/region_proposal_detection.py -O region_proposal_detection.py
wget -q https://raw.githubusercontent.com/jrosebr1/imutils/master/imutils/object_detection.py -O object_detection.py
wget -q https://raw.githubusercontent.com/SIlvaMFPedro/pyimagesearch/3e5c922b5f905078322d2283d704ef8875f043e0/region-proposal-object-detection/beagle.png -O beagle.png
cat region_proposal_detection.py | sed s/waitKey\(0\)/waitKey\(1000\)/g > tmp.py
cat tmp.py | sed s/imutils.object_detection/object_detection/g > region_proposal_detection-auto.py
python3 ./region_proposal_detection-auto.py --image beagle.png


```

### Result #6:

Text output to console such that:

```
???
[INFO] Performing selective search with 'fast' method...
[INFO] Found '[[  0 205  14  14]
 [317  44  27  16]
 [362 298  53  38]
 ...
 [  0   0 415 289]
 [376  88 124 248]
 [ 25  63 406 273]]'regions with 'fast' method of selective search!
[INFO] Proposals shape: (534, 224, 224, 3)
[INFO] Classifying proposals...
[INFO] Showing results for 'beagle'
[INFO] Showing results for 'quill'
[INFO] Showing results for 'clog'
[INFO] Showing results for 'paper_towel'
```

... ??? = (detail doesn't matter but should clearly indicate GPU usage by type/name/bus or similar where a GPU is available or CPU otherwise).

Then:

- an image of a dog is displayed in a window with a series of green bounding boxes on it that in turn surround varying objects in the scene.
