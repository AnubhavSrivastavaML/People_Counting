import numpy as np
import cv2
import os
import glob
import tempfile
import time


class DETECTOR:

    def __init__(self):
        self.labels = ['Person', 'Person']
        self.cfgfile = "yolov4-tiny.cfg"
        self.weightfile = "yolov4-tiny_7500.weights"
        self.vehicle_net = cv2.dnn.readNetFromDarknet(self.cfgfile, self.weightfile)
        self.ln = [self.vehicle_net.getLayerNames()[(i[0] - 1)] for i in self.vehicle_net.getUnconnectedOutLayers()]
        #print(ln)
        
    def detection(self, image , score_threshold = 0.25):
        H, W = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image,0.003,(416, 416), swapRB=True, crop=False)
        self.vehicle_net.setInput(blob)
        t=time.time()
        layerOutputs = self.vehicle_net.forward(self.ln)
        print("Time taken for feed formard : ",time.time()-t)
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence >= score_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, width, height = box.astype('int')
                    x = int(centerX - width / 2)
                    y = int(centerY - height / 2)
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
        info = []
        scores = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                print(x,y,w,h)
                if x < 0 :
                    x=0
                if y < 0 :
                    y=0
                conf = confidences[i]
                v_type = '{}'.format(self.labels[classIDs[i]])
                info.append([x, y, w, h])
                scores.append(conf)
 
        return (info,scores)



'''
detector = DETECTOR()
files  = glob.glob('/home/anubhav/DeepLearning/keras_object_detector/pedestriandataset/dataset/Images/*')
for image in files:
    img = cv2.imread(image)
    info = detector.detection(img)
    print(info)
    for box in info:
    	cv2.rectangle(img,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(0.255,0),1)
    cv2.imshow("Frame",img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
    	break
cv2.destroyAllWindows()
'''
