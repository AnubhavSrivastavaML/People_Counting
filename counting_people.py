# Usage example:  python3 counting_people.py --video=TownCentreXVID.avi


import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import math
import datetime
from counting.centroidtracker import CentroidTracker
from counting.trackableobject import TrackableObject
from people_detection import DETECTOR
from multiprocessing import Queue
from multiprocessing import Value


parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
        
    
    
  


class People_counting:
	def __init__(self,model_path,per=50):
		self.count = Queue(1000)
		self.ct = CentroidTracker(maxDisappeared=10, maxDistance=50)
		self.face_detector = DETECTOR()
		self.trackers = []
		self.trackableObjects = {}
		self.per = Value('i',per)
		self.up = 0
		self.down = 0
		
				
	def people_count(self,frame,per=50):
	    (boxes , scores) = self.face_detector.detection(frame, score_threshold=0.65)
	    print(boxes,scores)
	    self.per.value = per
	    if len(boxes) > 0:
		    for box in boxes:
		    	    box = list(map(int,box))
		    	    cv.rectangle(frame , (box[0],box[1]) , (box[0]+box[2] ,box[1]+box[3]) ,(0,255,0) , 1)
		    self.postprocess(frame, boxes)
		   
	    cv.imshow("Frame",frame)
	    cv.waitKey(1)
	    return
	             

	def postprocess(self,frame, outs):
		frameHeight = frame.shape[0]
		frameWidth = frame.shape[1]
		rects = []


		for box in outs:
			box = list(map(int,box))
			top = box[1]
			left = box[0]
			height = box[3]
			width = box[2]
			rects.append((left, top, left + width, top + height))
			#rects.append((left, top, right, bottom))
		objects = self.ct.update(rects)
		self.counting(objects)


	def counting(self,objects):
		frameHeight = int(self.per.value/100 * frame.shape[0])
		print("============================== Line Value {}    {}".format(frameHeight,self.per.value))
		print("====================================== up : {}  down : {}".format(self.up,self.down))
		if frame.shape[0]-frameHeight < 0 or frameHeight < 30:
			print("[error] Select other value for line percentage setting to 50% default of frame width")
			frameHeight = frame.shape[0]//2
		
		frameWidth = frame.shape[1]
		for (objectID, centroid) in objects.items():
			to = self.trackableObjects.get(objectID, None)

			if to is None:
				to = TrackableObject(objectID, centroid)

			else:
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)
				

				if not to.counted:
					#print(next( range(frameHeight//2 - 30, frameHeight//2 + 30)))
					#print(to.counted,direction,centroid[1] in range(frameHeight - 30, frameHeight + 30))				
					if direction < 0 and centroid[1] in range(frameHeight - 30, frameHeight + 30):
						#self.totalUp += 1
						s = datetime.datetime.now()
						print(1,"......................",s)
						self.count.put([1,s])
						#print("1",datetime.datetime.now())
						to.counted = True
						self.up +=1

					elif direction > 0 and centroid[1] in range(frameHeight - 30, frameHeight + 30):
						#self.totalDown += 1
						s = datetime.datetime.now()
						print(0,"......................",s)
						#self.count
						self.count.put([0,s])
						#print(datetime.datetime.now())
						to.counted = True
						self.down +=1
				#try:
				#    #print("counted :", self.count.get_nowait())
				#except:
				#    pass
			self.trackableObjects[objectID] = to
	    




cap = cv.VideoCapture(args.video)
per = 50
obj = People_counting(model_path = '',per = 50)


while cap.isOpened():
    
    hasFrame, frame = cap.read()
    if not hasFrame:
        key = cv.waitKey(3)
        cap.release()
        break
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    lineheight = int(per/100 *frameHeight)
    print((0, lineheight), (frameWidth, lineheight))
    frame = cv.line(frame, (0, lineheight), (frameWidth, lineheight), (255, 0, 0), 2)


    obj.people_count(frame.copy(),per)
    cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
    
    

