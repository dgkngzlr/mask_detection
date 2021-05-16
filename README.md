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
To start training process run this command in /darknet folder:
 * !./darknet detector train data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show -map

# Conclude
Afte a few hours I took my yolov4-tiny-custom_best.weights to my computer because I want to use it in my system.The results are :

![Alt Text](https://github.com/dgkngzlr/mask_detection/blob/main/chart.png?raw=true)

# How to use ?
If you complete GPU requeriments so you can use Yolov4Tiny.py file which includes Model class to use trained model appropreitly. If you dont want to use GPU please set "model.USE_GPU = False". You can use model wtih "from YolovTiny import Model" . There is a basic example to usage :

```python
if __name__ == "__main__":
    img = cv2.imread("./test_images/image7.jpg")
    model = Model("./model/yolov4-tiny-custom_best.weights","./model/yolov4-tiny-custom.cfg","./model/obj.names")
    model.load_yolo()
    boxes, confidences, classIDs, idxs = model.make_prediction(img)
    print(boxes,confidences,classIDs)
    img = model.draw_bounding_boxes(img,boxes,confidences,classIDs,idxs)
    
    cv2.imshow("window",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

The full code of Yolov4Tiny.py file is :

```python
import numpy as np
import cv2
import os
import time


class Model:
    
    def __init__(self,WEIGHTS_PATH,CONFIG_PATH,LABELS_PATH):
        self.WEIGHTS_PATH = WEIGHTS_PATH
        self.CONFIG_PATH = CONFIG_PATH
        self.LABELS_PATH = LABELS_PATH
        
        self.LABELS = []
        self.USE_GPU = True
        
        
        
        try:
            with open(self.LABELS_PATH, "r") as f:
                for line in f.readlines():
                    self.LABELS.append(line.strip("\n"))  # her bir class adından '\n' i sildik
            print("LABELS :",self.LABELS)
            self.__colors = np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype='uint8')
        except:
            print("[LOAD LABELS ERROR]")
            exit(1)
    
    def load_yolo(self):
        
        self.net = cv2.dnn.readNet(self.WEIGHTS_PATH, self.CONFIG_PATH)  # load YOLO algorithm.
        
        if self.USE_GPU:
            print('Using GPU')
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            print('Using CPU')
        
        self.layer_names = self.net.getLayerNames()
        
        print(f"Layers Names [{len(self.layer_names)}] :", self.layer_names)  # CNN modelinin katmanları

        self.output_layers = []  # Output layerlarin isimleri
        for i in range(len(self.layer_names)):
            if i + 1 in self.net.getUnconnectedOutLayers():  # Output layerlarin indisini doner
                self.output_layers.append(self.layer_names[i])
                
        print("Output Layers : ", self.output_layers)
        
        print("[WAIT] Model is loading...")
    
    def __extract_boxes_confidences_classids(self,outputs, confidence, width, height):
        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            for detection in output:            
                # Extract the scores, classid, and the confidence of the prediction
                scores = detection[5:]
                classID = np.argmax(scores)
                conf = scores[classID]
                
                # Consider only the predictions that are above the confidence threshold
                if conf > confidence:
                    # Scale the bounding box back to the size of the image
                    box = detection[0:4] * np.array([width, height, width, height])
                    centerX, centerY, w, h = box.astype('int')

                    # Use the center coordinates, width and height to get the coordinates of the top left corner
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))

                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(conf))
                    classIDs.append(classID)

        return boxes, confidences, classIDs
        
    def make_prediction(self, image, confidence=0.5, threshold=0.3):
        height, width = image.shape[:2]
        
        # Create a blob and pass it through the model
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # Extract bounding boxes, confidences and classIDs
        boxes, confidences, classIDs = self.__extract_boxes_confidences_classids(outputs, confidence, width, height)

        # Apply Non-Max Suppression
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

        return boxes, confidences, classIDs, idxs
    
    def draw_bounding_boxes(self,image, boxes, confidences, classIDs, idxs):
        if len(idxs) > 0:
            for i in idxs.flatten():
                # extract bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                # draw the bounding box and label on the image
                color = [int(c) for c in self.__colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image
    
    

```

If you want to use it with webcam , it is also available. Please run webcam.py file . You can find it above.

# Referances
* https://github.com/techzizou/
* https://github.com/AlexeyAB/
* https://pjreddie.com/darknet/yolo/
