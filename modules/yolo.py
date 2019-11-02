# -*- coding: utf-8 -*-

import cv2
import numpy as np

'''

@Created by Mohamed Salah
--------------------
@darkflow  repo https://github.com/thtrieu/darkflow

'''

class Yolo:
    tfnet=object()
    def label_is_(self,result,kind):
        """
        indicate whether the result contains kind or not
        
        Parameters
        ----------
        arg1 : dict
            predicted dict from yolo
        arg2 : string
            kind of object that you want
       
    
        Returns
        -------
        boolean
             kind or not
    
        """
        return result["label"]==kind
    
    def draw_pred(self,imgcv,confidence=0.5,kind="person"):
        """
        take an image and draw predicted contours with labels
    
        Parameters
        ----------
        arg1 : np.array (image)
            predicted dict from yolo
        arg2: float (default is 0.5) 
            threshold to output the predicted object as your needed kind
        arg3 : string (default is "person") 
            kind of object that you want
    
    
        Returns
        -------
        array
            image with predicted contours and labels
        boolean
            frame contains kind or not
    
        """
            
        flag=0
        results = self.tfnet.return_predict(imgcv)
        for (i,result) in enumerate(results):
            if(result["confidence"]>=confidence and self.label_is_(result,kind)):
                flag=1
                x=result["topleft"]["x"]
                w=result["bottomright"]['x']-result["topleft"]["x"]
                y=result["topleft"]["y"]
                h=result["bottomright"]['y']-result["topleft"]["y"]
                imgcv=cv2.rectangle(imgcv,(x,y),(x+w,y+h),(0,255,2),4)
                label_position=(x+int(w/2),abs (y-1))
                cv2.putText(imgcv,result['label'],label_position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
        return imgcv,flag
    def init(self,options,TFNet):
        """
        initialize yolo darkflow
        
        Parameters
        ----------
        arg1 : dict
            options
        arg2 : TFNet
            tfnet module
       
    
        Returns
        -------
        boolean
             kind or not
    
        """
        self.tfnet= TFNet(options)
    