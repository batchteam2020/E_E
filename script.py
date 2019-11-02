
from darkflow.net.build import TFNet
import cv2
from modules.yolo import Yolo



options = {"model": "./cfg/yolo.cfg", "load": "./bin/yolo.weights", "threshold": 0.1,"gpu":0.5}
#create yolo object and intialize it 
yolo_=Yolo()
yolo_.init(options,TFNet)

#example code to read from video
cap=cv2.VideoCapture("top-10-dumb-robbers-video-digest.mp4")

while True:
    
    _,imgcv=cap.read()
    imgcv,flag=yolo_.draw_pred(imgcv,0.3)
    cv2.imshow("image",imgcv)
    k=cv2.waitKey(1) & 0xFF
    if(k==27):
        break
    
cap.release()
cv2.destroyAllWindows()

