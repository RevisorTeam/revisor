# **The Revisor** Installation Guide

Before installation make sure your machine satisfies the [hardware requirements] 
and data satisfies the [source data requirements].

Following guide assumes that you perform installation on the "fresh" 
Ubuntu 18.04 *(other versions isn't supported)*. 

You may also use docker but don't forget to compile ffmpeg with CUDA support 
if you are planning to convert videos to mp4 format.

### Step 0

---

Update Ubuntu packages and install required ones.

```
sudo apt-get update
sudo apt-get upgrade -y
sudo apt install ubuntu-restricted-extras -y
sudo apt-get install -y build-essential cmake unzip zip git curl nano pkg-config libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libopenblas-dev libatlas-base-dev liblapack-dev gfortran libhdf5-serial-dev graphviz python3-dev python3-tk python-imaging-tk linux-image-generic linux-image-extra-virtual linux-source linux-headers-generic
```

### Step 1

---

Install NVIDIA GPU driver.

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
sudo apt-get install -y nvidia-driver-470

sudo reboot
```

### Step 2

---

Install CUDA (11.1) and cuDNN (8.1.0).

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-1-local_11.1.1-455.32.00-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1804-11-1-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-1
```

Copy following two "export" lines to the end of the files `~/.bashrc` and `/root/.bashrc`. You can access them via nano:

`sudo nano ~/.bashrc`

`sudo nano /root/.bashrc`

To save edits, press `Ctrl + O`, `Enter` and `Ctrl + X`.

Lines to add:

```
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-11.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
```

Download cuDNN archive from [here] ([direct link], registration needed):
1. `Download cuDNN v8.1.0 (January 26th, 2021), for CUDA 11.0, 11.1 and 11.2`
2. `cuDNN Library for Linux (x86_64)`

Go to the downloaded archive directory and execute following:

```
tar -zvxf cudnn-11.2-linux-x64-v8.1.0.77.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.1/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-11.1/lib64
sudo chmod a+r /usr/local/cuda-11.1/include/cudnn*.h /usr/local/cuda-11.1/lib64/libcudnn*

sudo reboot
```

To verify CUDA installation, execute `nvcc -V`.

If you can't access `nvcc -V` from console, edit `export PATH=...` line: change `/cuda/` to `/cuda-11.1/`

### Step 3

---

TensorRT installation. Download [TensorRT 7.2.3.4], go to the downloads directory and execute:

```
sudo apt-get -y install python3-pip
pip3 install --upgrade pip

sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrt -y
sudo apt-get install python3-libnvinfer-dev -y
sudo apt-get install onnx-graphsurgeon -y

sudo reboot
```

Type `dpkg -l | grep TensorRT` to verify TensorRT installation.

### Step 4

---

GPU accelerated FFmpeg installation. This step can be skipped if you 
aren't going to convert videos to mp4.

```
sudo apt-get update

mkdir ~/installs && cd ~/installs
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers && sudo make install && cd -
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/
cd ffmpeg && mkdir -p $(pwd)/build_x64_release_shared

sudo apt-get install -y build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev libfdk-aac-dev libass-dev libopus-dev libtheora-dev libvorbis-dev libvpx-dev libssl-dev libavcodec-dev libavformat-dev libswscale-dev libavutil-dev nasm libx264-dev libx265-dev libnuma-dev libvpx-dev libfdk-aac-dev libmp3lame-dev libopus-dev

sudo su

./configure --enable-gpl --enable-libx264 --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64

PATH="$HOME/bin:$PATH" \
make -j8 && \
make install && \
hash -r
```

Press `Ctrl + D` to exit from sudo.

### Step 5

---

Required python3 libraries installation.

```
sudo apt-get -y install protobuf-compiler libprotoc-dev
sudo pip3 install onnx==1.4.1
pip3 install --user pycuda==2021.1
pip3 install --user tensorflow-gpu==2.6.0
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install --user tf2onnx onnxruntime Cython==0.29.23
```

Download The Revisor archive with source code files, unpack it and install requirements:
```
wget https://github.com/RevisorTeam/revisor.git
unzip revisor-main.zip
pip3 install -r ~/revisor-main/requirements.txt
```

### Step 6

---

Build YOLOv4 TensorRT engine.

```
cd ~/revisor-main/yolov4_trt/plugins
make
```
If you get an error, try to edit Makefile located in `yolov4_trt/plugins/Makefile`:
1. Line 21: change `compute=75` to `compute=YOUR_GPU_CC`. Find your GPU's compute compatibility [you can here] 
2. Lines 37 and 38: replace `/cuda/` to `/cuda-11.1/`

After successful execution, `yolo_layer.o` and `libyolo_layer.so` will be created.

```
cd ~/revisor-main/yolov4_trt/yolo
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
ln -sf yolov4.weights yolov4-608.weights
python3 yolo_to_onnx.py -m yolov4-608
python3 onnx_to_tensorrt.py -m yolov4-608
```

### Step 7

---

Build EvoPose2D TensorRT engine. 

```
mkdir ~/revisor-main/evopose2d/models
```

Download [official EvoPose2D-M FP32 weights]
from Google Drive (`evopose2d_M_f32.h5` file) and place it to:

`evopose2d/models/evopose2d_M_f32.h5`

File `evo_onnx2trt.py` configured for RTX 3080 10 VRAM GB. 
If your GPU is different, you need to change following in the script:
1. Increase or decrease batch size (line 5): `MAX_BATCH_SIZE` variable
2. Input tensor name (lines 35, 36):
   * For RTX 20xx series - set `input_1:0` name
   * For RTX 30xx series - set `input_1` name
   * RTX 40xx series is not tested. Probably, `input_1` should work. 
   
   Input tensor name you can find in the last outputs of calling `python3 -m tf2onnx.convert ...`:

   `INFO - Model inputs: ['input_1']` line

```
cd ~/revisor-main/evopose2d
python3 evo_h52pb.py
python3 -m tf2onnx.convert --saved-model models/evopose2d_M_f32 --output "models/evopose2d_M.onnx"
python3 evo_onnx2trt.py
```

### Step 8

---

Download Action recognition model, YOLACT and QueryInst weights.

```
mkdir ~/revisor-main/action_recognition/weights
mkdir ~/revisor-main/yolact/weights
mkdir ~/revisor-main/QueryInst/weights
```

1. Download [action recognition weights file] and put it in `action_recognition/weights/votes_recognizer.pt`
2. Download [YOLACT weights file] and put it in `yolact/weights/boxes_finder.pth`
3. Download [QueryInst weights file] and put it in `QueryInst/weights/lid_finder.pth`

Install mmdet library.

```
cd ~/revisor-main/QueryInst
sudo python3 setup.py develop
```

## Good job!

Congratulations, you have successfully installed the Revisor!

Don't forget to place your source data to `data/` and edit `cfg.py`.

[hardware requirements]: HARDWARE_REQUIREMENTS.MD
[source data requirements]: DATA_REQUIREMENTS.MD
[here]: https://developer.nvidia.com/rdp/cudnn-archive
[direct link]: https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.0.77/11.2_20210127/cudnn-11.2-linux-x64-v8.1.0.77.tgz
[TensorRT 7.2.3.4]: https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/7.2.3/local_repos/nv-tensorrt-repo-ubuntu1804-cuda11.1-trt7.2.3.4-ga-20210226_1-1_amd64.deb
[you can here]: https://developer.nvidia.com/cuda-gpus
[official EvoPose2D-M FP32 weights]: https://drive.google.com/drive/folders/1dIiXdGMhJzEu_CW6ZBgiaOqQcixS7nvV?usp=sharing
[YOLACT weights file]: https://drive.proton.me/urls/KGVSKHSBE8#UyT04ktqgohp
[QueryInst weights file]: https://drive.proton.me/urls/7T4TNPJFFG#HN4asDil6Myx
[action recognition weights file]: https://drive.proton.me/urls/5AVPHW59DM#joxrL27eJcKA
