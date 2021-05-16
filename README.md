# Mask Detection with Yolo v4 Tiny
Performing mask detection with Yolo v4 Tiny due to the pandemic situation.

![Alt Text](https://github.com/dgkngzlr/mask_detection/blob/main/mask.gif?raw=true)

* [Overview](#overview)
* [Requirements](#requirements)
* [Pre trained models](#pre-trained-models)
* [Performing the task](#performing-the-task)
* [Conclude](#conclude)
* [Referances](#referances)

# Overview
Our world is going through a difficult time due to the pandemic. For us people, wearing masks has now become a part of our normal lives. This project has been a good opportunity both to contribute to this issue and to improve myself in image processing. I believe that what you will learn from here will further increase your interest in image processing.However, before I start, I should state that this study is not an original study.Therefore, you can find the resources that I have benefited from while carrying out this project please check Referances section.But I want to explain using of the trained model here in more detail with Python, so you can continue reading this article :).Before continuing I implemented this project in GNU/Linux because of the installation parts are easier than other os'.At least I felt that way.
Okey then let's get started.



# Requirements
My system specs :
* GPU --> Nvidia GTX 1050
* Cudnn --> 8.4
* CUDA --> 11.0

If you want to a real time system you should use GPU boosting. For GPU boosting you also should install CUDA,Cudnn and OpenCV from source. You can benefit from Cmake program to built OpenCV from source. Please check these options before built it :
   * -CMAKE_BUILD_TYPE=RELEASE \
   * -CMAKE_INSTALL_PREFIX=/usr/local \
   * -INSTALL_C_EXAMPLES=ON \
   * -INSTALL_PYTHON_EXAMPLES=ON \
   * -OPENCV_GENERATE_PKGCONFIG=ON \
   * -OPENCV_EXTRA_MODULES_PATH= PATH_TO_OPENCV_CONTRIB \
   * -BUILD_EXAMPLES=ON
   * -with_cuda=ON
   * -with_cudnn=ON
   * -with_cublas=ON
   * -OPENCV_DNN_CUDA=ON
   * -build_opencv_world=ON
   * -cuda_arch_bin=6.1 # compute capability it may changes to system
   * -OPENCV_ENABLE_NONFREE=ON
   * -ENABLE_FAST_MATH=1
   * -CUDA_FAST_MATH=1
   * -HAVE_opencv_python3=ON
 
 Breifly:
 * 1-) Install CUDA
 * 2-) Install Cudnn
 * 3-) Install Opencv with GPU support(Version 4.5 is okey)
 
These requirements are necessarily for GPU computing.


# Pre trained models
There are weights-file for different cfg-files. So please check https://github.com/AlexeyAB/darknet. But I used Yolov v4 Tiny pre trained model for this project you may use other yolo version but I did not try them. We will train it our custom data later.
* Yolov v4 Tiny Pre-trained model :
* https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
* Yolov v4 Tiny Config File :
* https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg

# Performing the task
I used Google Colab to train pre-trained yolv4_tiny model with our custom data-set.Please check this article https://medium.com/analytics-vidhya/yolov4-vs-yolov4-tiny-97932b6ec8ec and you can also find the data set here https://www.kaggle.com/techzizou/labeled-mask-dataset-yolo-darknet. Also you should create obj.names and obj.data according to dataset and number of classes in darknet/data folder .And I share my colab notebook with you so you can follow this article from there. https://colab.research.google.com/drive/1IQ9BMGSG5pKSTe4gqxwS9zk42pKq8LAW?usp=sharing . If you want to create own data set you can use labelIMG program. Before start training run process.py file in /darknet folder to create train.txt and test.txt in darknet/data folder.
Now edit options in Yolov v4 Tiny Config File :
*   change line batch to batch=64
*   change line subdivisions to subdivisions=16
*   change line max_batches to (classes*2000 but not less than number of training images, but not less than number of training images and not less than 6000), f.e. max_batches=6000 if you train for 3 classes
*    change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
*    set network size width=416 height=416 or any value multiple of 32
*    change line classes=80 to your number of objects in each of 2 [yolo]-layers
*    change [filters=255] to filters=(classes + 5)x3 in the 2 [convolutional] before each [yolo] layer, keep in mind that it only has to be the last              [convolutional] before each of the [yolo] layers. So if classes=1 then it should be filters=18. If classes=2 then write filters=21.

It was taken from https://github.com/techzizou/yolov4-custom_Training. Please check this web site.
To start training process run this command in /darknet folder.
 * !./darknet detector train data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show -map

# Conclude
Afte a few hours I took my yolov4-tiny-custom_best.weights to my computer because I want to use it in system.The results are :
![Alt Text](https://github.com/dgkngzlr/mask_detection/blob/main/mask.gif?raw=true)

# Referances
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Accumsan tortor posuere ac ut consequat semper. Elementum pulvinar etiam non quam lacus suspendisse. Nunc pulvinar sapien et ligula ullamcorper malesuada proin libero. Ultricies integer quis auctor elit sed vulputate mi. Amet justo donec enim diam vulputate. Et malesuada fames ac turpis egestas sed tempus. Morbi enim nunc faucibus a pellentesque sit. Volutpat consequat mauris nunc congue nisi vitae suscipit tellus mauris. Risus at ultrices mi tempus imperdiet nulla malesuada pellentesque elit. Amet porttitor eget dolor morbi non arcu risus. Auctor neque vitae tempus quam pellentesque. Consequat id porta nibh venenatis cras. Nibh cras pulvinar mattis nunc. Imperdiet massa tincidunt nunc pulvinar sapien. At auctor urna nunc id. Quis imperdiet massa tincidunt nunc pulvinar sapien et ligula ullamcorper. Condimentum mattis pellentesque id nibh tortor id aliquet lectus proin. Massa tincidunt nunc pulvinar sapien et.
