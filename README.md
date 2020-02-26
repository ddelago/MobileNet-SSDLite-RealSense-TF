# MobileNet-SSDLite-RealSense-TF
RaspberryPi3(Raspbian Stretch) + MobileNetv2-SSDLite(Tensorflow/MobileNetv2SSDLite) + RealSense D435 + Tensorflow + without Neural Compute Stick(NCS)

## Environment
- OpenCV 4.1.2
- MobileNetv2-SSDLite [MSCOCO]
- RealSense D435 (Firmware ver v5.10.6)
- Python3.7
- Tensorflow 2.1.0
- OpenGL

## Windows environment construction sequence
- `pip install tensorflow`
- `pip install tensorflow-gpu`
- Go [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/) and pip install the PyOpenGL and PyOpenGL_accelerate wheels.
- Clone the [Tensorflow models](https://github.com/tensorflow/models) repository.
- Go to the Object Detection folder under research.
- Clone this repository here
  - You can also add the object detection path to you Python path and clone this repo wherever you like.
  - You will have to modify line 138 in `object_deteection/utils/label_map_util.py`
