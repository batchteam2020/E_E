# -*- coding: utf-8 -*-
import os
from shutil import copyfile
import cv2
import shutil
'''
@created by Mohamed salah

'''
class preprocessing:
    def folder_to_seq(self,folder_path,images_group,dest_path,addtional_char):
        
        '''
        images_group=[
        ["start","end"]
        ]
        '''
        for index,group in enumerate(images_group):
            start=group[0]
            end=group[1]
            diff=end-start
            length=int(diff/6)
            counter=0
            counter_=0
            zeros=""
            '''
            save 
            '''
            
            if(length>0):
                #shutil.rmtree(dest_path+str(0)+addtional_char+str(index))
                os.makedirs(dest_path+str(0)+addtional_char+str(index))

                
                for i in range(start,end):
                    if(i>99):
                        zeros="00"
                    elif(i>9):
                        zeros="000"
                    elif(i<9):
                        zeros="0000"
                        
                    image=folder_path+"/"+zeros+str(i)+".jpg"
                    copyfile(image, dest_path+str(counter_)+addtional_char+str(index)+"/"+str(i)+".jpg")
                    
                    counter+=1
                    if(counter%6==0 and counter!=0):
                        counter_+=1
                        if(counter_==length):
                            break
                        #shutil.rmtree(dest_path+str(counter_)+addtional_char+str(index))
                        os.makedirs(dest_path+str(counter_)+addtional_char+str(index))
                    
    def crop_person(self,folder_path,image,image_name,roi):
        y=roi["y"]
        x=roi["x"]
        w=roi["w"]
        h=roi["h"]
        crop_img = image[y:y+h, x:x+w]
        cv2.imshow("IMAGE",crop_img)
        #cv2.imwrite("TESTO.jpg",crop_img)
        cv2.imwrite(folder_path+"/"+image_name,crop_img)
    def frames_to_video(self,path,dest_path,name):
        import cv2
        import numpy as np
        import glob
         
        img_array = []
        print(path)
        for filename in glob.glob(path+'*.jpg'):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
         
        out = cv2.VideoWriter(path+name+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        

                    

                    
                    
                    
                    
            
        
