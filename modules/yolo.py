# -*- coding: utf-8 -*-
#@Created by Mohamed Salah
import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from scipy.spatial import distance
'''
@created by

Mohamed salah

'''
class Yolo:
    tfnet=object()
    centers=[]
    dim=[]
    info=[]
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
    def draw_boundry(self,imgcv,draw):
        global_neighbours=[]
        for iindex,i in enumerate(range(len(self.centers))):
            local_neighbours=[]
            for jindex,j in enumerate(range(iindex,len(self.centers))):
               dst = distance.euclidean(self.centers[iindex], self.centers[jindex])
               local_neighbours.append({"node":jindex,"dst":dst})
            sorted_localneighbours=sorted(local_neighbours, key = lambda ii: ii['dst'])
            if(len(sorted_localneighbours)>1):
                global_neighbours.append(sorted_localneighbours[1])
            else:
                global_neighbours.append(sorted_localneighbours[0])
                
        print("FINAL",global_neighbours)
        for index,i in enumerate(global_neighbours):
            if(i["dst"]<300):
                x=self.dim[i["node"]][0]
                w=self.dim[i["node"]][1]
                y=self.dim[i["node"]][2]
                h=self.dim[i["node"]][3]
                x_=self.dim[index][0]
                w_=self.dim[index][1]
                y_=self.dim[index][2]
                h_=self.dim[index][3]
                if(x<x_ and y>y_):
                    imgcv=cv2.rectangle(imgcv,(x,y),(x_+w_,y_+h_),(0,0,255),6)
                elif(x>x_ and y<y_):
                    imgcv=cv2.rectangle(imgcv,(x_,y_),(x+w,y+h),(0,0,255),6)
                elif(x<x_ and y<y_) :
                     
                    imgcv=cv2.rectangle(imgcv,(x_,y_),(x,y+h),(0,0,255),6)
                elif(x>x_ and y>y_) :
                    imgcv=cv2.rectangle(imgcv,(x,y),(x_,y_+h_),(0,0,255),6)
               

        return imgcv      
                   
                
        #sortedCenters=sorted(self.centers, key = lambda i: i['center'])
        #print(sortedCenters)
        '''for i,iindex in enumerate(range(len(centers))):
                for j,jindex in enumerate(range(iindex,len(centers))):
                    dst = distance.euclidean(centers[iindex], centers[jindex])
                    print(dst)'''
    def draw_pred(self,imgcv,confidence=0.5,draw=True,kind="person"):
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
        self.centers=[]
        flag=0
        rois=[]
        results = self.tfnet.return_predict(imgcv)
        x,y,w,h=0,0,0,0
        for (i,result) in enumerate(results):
            if(result["confidence"]>=confidence and self.label_is_(result,kind)):
                flag=1
                
                x=result["topleft"]["x"]
                w=result["bottomright"]['x']-result["topleft"]["x"]
                y=result["topleft"]["y"]
                h=result["bottomright"]['y']-result["topleft"]["y"]
                if(draw):
                    imgcv=cv2.rectangle(imgcv,(x,y),(x+w,y+h),(0,255,2),4)
                rois.append({"x":x,"w":w,"y":y,"h":h})

                center = ((x+w)/2,(y+h)/2)
                self.centers.append(center)
                self.dim.append([x,w,y,h])
                self.info.append({"center":center,"x":x,"w":w,"y":y,"h":h})
                label_position=(x+int(w/2),abs (y-1))
                if(draw):
                    cv2.putText(imgcv,result['label']+" NO."+str(i),label_position,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
        #if(len(self.centers)>=2):
            #imgcv=self.draw_boundry(imgcv)      
        return imgcv,flag,rois,self.centers
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
    