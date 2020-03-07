# -*- coding: utf-8 -*-
'''
@created by Mohamed salah

'''
import cv2
class Tracking:
    

    def start(self,roi,frame):
        tracker = cv2.TrackerKCF_create()
      
        
        #roi = cv2.selectROI(frame, False)
        ret = tracker.init(frame, roi)
        success, roi = tracker.update(frame)
        (x,y,w,h) = tuple(map(int,roi))
        if success:   
           p1 = (x, y)
           p2 = (x+w, y+h)
          
           return (p1,p2)
        else : 
           return (0,0)
           
        
   
     