

#0->boxing 



# -*- coding: utf-8 -*-

#from preprocessing import *


import pickle as pkl
import tensorflow 

tensorflow.debugging.set_log_device_placement(True)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Dense, Dropout, Activation, Flatten, Convolution3D, MaxPooling3D
from tensorflow.keras.optimizers import Adam ,SGD
#from tensorflow.keras.utils import np_utils, generic_utils
import numpy 
from sklearn.model_selection import train_test_split
#from tensorflow.keras.backend import set_session
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,Callback
import tensorflow.keras.utils as np_utils

import cv2

import numpy as np



from darkflow.net.build import TFNet
import cv2
from modules.yolo import Yolo
from modules.tracking import Tracking
from modules.person_preprocessing import preprocessing

from scipy.spatial import distance

import os

options = {"model": "./cfg/yolo.cfg", "load": "./bin/yolo.weights", "threshold": 0.1,"gpu":0.7}



#create yolo object and intialize it 
yolo_=Yolo()


yolo_.init(options,TFNet)

import numpy as np
import pyautogui











print("Num GPUs Available: ", (tensorflow.config.experimental.list_physical_devices(device_type=None)))


#from keras.backend.tensorflow_backend import set_session


'''
set_session(tf.Session(config=config))
'''
#dataset=preprocessing("../dataset/HockeyFights",(112,112))
#pkl.dump(dataset,open("./HockyFights","wb"))
'''with open("./hocky_dataset_frames_no_v1","rb") as f:
    dataset=pkl.load(f)
    
dataset_frames=get_frames_from_indices("../dataset/HockeyFights/",dataset)    
pkl.dump(dataset_frames,open("./hocky_dataset_v1","wb"))
cv2.imshow("frame",dataset_frames[0][3])
cv2.waitKey(9999999)
cv2.destroyAllWindows()    
    

 ''' 
  
labels=[]

with open("./actions_dataset_v0","rb") as f:
    dataset_frames=pkl.load(f)

with open("./actions_labels_v0","rb") as f:
    labels=pkl.load(f)
        
img_rows,img_cols,frames_no=112,112,6
samples=len(labels)


train_data = [dataset_frames,labels]

dataset_frames = np.array(dataset_frames)


X_set,y_set=train_data
#y_binary_set = np_utils.to_categorical(y_set,2)
y_set = np_utils.to_categorical(y_set, 4)
#X_set = X_set.astype('float32')
X_set /=np.max(X_set)
X_train,X_test,y_train,y_test=train_test_split(X_set,y_set,test_size=0.2)

cv2.imshow("frame",X_train[0][0])
cv2.waitKey(9999)
cv2.destroyAllWindows()    

X_train=np.reshape(X_train,(len(X_train),frames_no,img_rows,img_cols,1))
X_test=np.reshape(X_test,(len(X_test),frames_no,img_rows,img_cols,1))


#builing our network
model = Sequential()
#first convolution layer(input layer)
model.add(Convolution3D(filters=32 , strides=(1, 1, 1) , kernel_size=(3,3,3)  , activation='relu' , input_shape = (frames_no ,  img_rows,img_cols,1) , padding='same' ))
#first pooling layer
model.add(MaxPooling3D(pool_size=(1, 2, 2)))

#dropout
model.add(Dropout(0.25))
#second convlution layer
model.add(Convolution3D(filters=64 , strides=(1, 1, 1) , kernel_size=(3,3,3) , activation='relu' , padding='same' ))
#second pooling layer
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#third convlution layer
model.add(Convolution3D(filters=128 , kernel_size=(3,3,3) , activation='relu' , padding='same' ))
model.add(BatchNormalization())

#fourth convlution layer
model.add(Convolution3D(filters=128 , kernel_size=(3,3,3) , activation='relu' , padding='same' ))
#third pooling layer
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization())

#fifth convlution layer
model.add(Convolution3D(filters=256 , kernel_size=(3,3,3) , activation='relu' , padding='same'  ))
#sixth convlution layer
model.add(Convolution3D(filters=256 , kernel_size=(3,3,3) , activation='relu' , padding='same' ))
#fourth pooling layer
model.add(MaxPooling3D(pool_size=(2, 2, 2),padding='same' ))
model.add(BatchNormalization())

#seventh convlution layer
model.add(Convolution3D(filters=256 , kernel_size=(3,3,3) , activation='relu' , padding='same'  ))
#8th convlution layer
model.add(Convolution3D(filters=256 , kernel_size=(3,3,3) , activation='relu' , padding='same'  ))
#fifth pooling layer
model.add(MaxPooling3D(pool_size=(2, 2, 2) , padding='same'))
model.add(BatchNormalization())


#dropout layer

# flattening
model.add(Flatten())
#first fully connected layer
model.add(Dense(units=100 , activation='relu'))
#dropout
model.add(Dropout(0.25))

#second fully connected layer
model.add(Dense(units=100 , activation='relu'))
#output layer
model.add(Dense(units=4))
# Activation layer
model.add(Activation('softmax'))
#compile our model
#fit our model


model.summary()


model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=0.0005),metrics=["accuracy"])
#fit our model
checkpoint=ModelCheckpoint("action_model.h5",monitor="val_loss",mode="min",save_best_only=True,verbose=1)
earlystop=EarlyStopping(monitor="val_loss",patience=100,verbose=1,restore_best_weights=True)
callbacks=[earlystop,checkpoint]



with tensorflow.device("/gpu:0") as f:
    history=model.fit(x=X_train, y=y_train,batch_size=10 ,epochs=50 ,callbacks=callbacks, validation_data=(X_test,y_test))
    



model.load_weights("action_model.h5")





pred=model.predict([[X_test[50]]])
print("PRED" ,np.argmax(pred))
print(y_test[50])




testo=X_test[5]

for frame in X_test[50]:
    cv2.imshow("frame",frame)
    cv2.waitKey(9999)
    cv2.destroyAllWindows()



Oframes=[]

sampled_frames,frames=preprocessing("../dataset/HockeyFights/no210_xvid.avi",(112,112),16,1)

for frame_index in sampled_frames:
            frame=frames[frame_index]
            Oframes.append(frame)
Oframes=np.reshape(Oframes,(len(Oframes),img_rows,img_cols,1))

Oframes = Oframes.astype('float32')
Oframes /=np.max(Oframes)

predO=np.argmax(model.predict([[Oframes]]))
print(predO)



import numpy as np
import pyautogui
import imutils

frames=[]
cap=cv2.VideoCapture("rtsp://admin:mohamed2015@192.168.1.6:554/Streaming/Channels/101/")
while(cap.isOpened()):
    _,image=cap.read()
    image_=cv2.resize(image,(112,112))
    image_=cv2.cvtColor(image_,cv2.COLOR_RGB2GRAY)

    frames.append(image_)
    if(len(frames)%16==0 and len(frames)!=0):
        frames=np.array(frames)
        print(frames.shape)
        frames=np.reshape(frames,(len(frames),img_rows,img_cols,1))
        frames = frames.astype('float32')
        frames /=np.max(frames)
        predO=np.argmax(model.predict([[frames]]))
        print(predO)
        #print(np.argmax(pred) if pred[np.argmax(pred)]>0.1  else -1)
        frames=[]
    cv2.imshow("test", image)
    k=cv2.waitKey(1) & 0xFF
    if(k==27):
        break
cv2.destroyAllWindows() 







frames=[]
predT=-1
while True:
    image = pyautogui.screenshot()
    image=np.array(image)
    imgcv,flag,rois,centers=yolo_.draw_pred(image,0.5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image=image[:1000,:1000]
    crop_img=image
    if(len(rois)!=0):
            roi=rois[0]
            y=roi["y"]
            x=roi["x"]
            w=roi["w"]
            h=roi["h"]
            crop_img = image[y:y+h, x:x+w]
    
    image_=cv2.resize(crop_img,(112,112),interpolation=cv2.INTER_NEAREST)
    
            #imgcv=cv2.rectangle(imgcv,(x,y),(x+w,y+h),(0,0,255),6)        
            #cv2.imshow("image",imgcv)
    frames.append(image_)
  
    if(len(frames)%6==0 and len(frames)!=0):
        frames=np.array(frames)
        print(frames.shape)
        frames=np.reshape(frames,(len(frames),img_rows,img_cols,1))
        frames = frames.astype('float32')
        frames /=np.max(frames)
        predicted=model.predict([[frames]])
        predO=np.argmax(predicted)
        if(predicted[0][predO]>0.8):
            predT=predO
        else:
            predT=-1
       
        print(predicted)
        print("Predicted => "+str(predO))
      
            

        #print(np.argmax(pred) if pred[np.argmax(pred)]>0.1  else -1)
        frames=[]
    if(predT==0):
        cv2.putText(imgcv,"jump",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
    elif(predT==1):
        cv2.putText(imgcv,"kick",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
    elif(predT==2):
        cv2.putText(imgcv,"sit",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
    elif(predT==3):
        cv2.putText(imgcv,"squat",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
        
    cv2.imshow("test", imgcv)
    k=cv2.waitKey(1) & 0xFF
    if(k==27):  
        break


cv2.destroyAllWindows()




import numpy as np
import pyautogui

while True:
    image = pyautogui.screenshot()
    image=np.array(image)
    imgcv,flag,rois,centers=yolo_.draw_pred(image,0.5)
    
        
    #if(roi[0]!=0 ):
        #boundry=Tracking().start(roi,imgcv)
        #print(boundry)
    ##cv2.rectangle(imgcv, roi[0], roi[1], (0,0,255), 3)
    print(rois)

        
      
    
    cv2.imshow("image",imgcv)  
    k=cv2.waitKey(1) & 0xFF
    
    if(k==27):
        break


cv2.destroyAllWindows()











import numpy as np
import pyautogui
frames=[]
predT=-1

cap=cv2.VideoCapture("rtsp://admin:mohamed2015@192.168.1.6:554/Streaming/Channels/101/")
while(cap.isOpened()):
    _,image=cap.read()
   
    

    imgcv,flag,rois,centers=yolo_.draw_pred(image,0.5)
    imgcv=cv2.resize(imgcv,(1000,1000))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image=image[:1000,:1000]
    crop_img=image
    if(len(rois)!=0):
            roi=rois[0]
            y=roi["y"]
            x=roi["x"]
            w=roi["w"]
            h=roi["h"]
            crop_img = image[y:y+h, x:x+w]
    
    image_=cv2.resize(crop_img,(112,112),interpolation=cv2.INTER_NEAREST)
    
            #imgcv=cv2.rectangle(imgcv,(x,y),(x+w,y+h),(0,0,255),6)        
            #cv2.imshow("image",imgcv)
    frames.append(image_)
  
    if(len(frames)%6==0 and len(frames)!=0):
        frames=np.array(frames)
        print(frames.shape)
        frames=np.reshape(frames,(len(frames),img_rows,img_cols,1))
        frames = frames.astype('float32')
        frames /=np.max(frames)
        predicted=model.predict([[frames]])
        predO=np.argmax(predicted)
        if(predicted[0][predO]>0.8):
            predT=predO
        else:
            predT=-1
       
        print(predicted)
        print("Predicted => "+str(predO))
      
            

        #print(np.argmax(pred) if pred[np.argmax(pred)]>0.1  else -1)
        frames=[]
    if(predT==0):
        cv2.putText(imgcv,"jump",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
    elif(predT==1):
        cv2.putText(imgcv,"kick",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
    elif(predT==2):
        cv2.putText(imgcv,"sit",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
    elif(predT==3):
        cv2.putText(imgcv,"squat",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2)
        
    cv2.imshow("test", imgcv)
    k=cv2.waitKey(1) & 0xFF
    if(k==27):  
        break
cv2.destroyAllWindows() 







frames=[]
predT=-1
while True:
    image = pyautogui.screenshot()
    


cv2.destroyAllWindows()









