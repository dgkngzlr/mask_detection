from Yolov4Tiny import Model

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

model = Model("./model/yolov4-tiny-custom_best.weights","./model/yolov4-tiny-custom.cfg","./model/obj.names")
model.load_yolo()
model.USE_GPU = True

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    boxes, confidences, classIDs, idxs = model.make_prediction(frame)
    frame = model.draw_bounding_boxes(frame,boxes,confidences,classIDs,idxs)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()