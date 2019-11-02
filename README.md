
## Intro

## Dependencies

Python3, tensorflow-gpu(for gpu) or tensorflow (for cpu)  1.14  ,numpy, opencv .
### Getting started
#### install darkflow first
1. Just build the Cython extensions in place. NOTE: If installing this way you will have to use `./flow` in the cloned darkflow directory instead of `flow` as darkflow is not installed globally.
    ```
    python3 setup.py build_ext --inplace
    ```

2. Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
    ```
    pip install -e .
    ```

3. Install with pip globally
    ```
    pip install .
    ```

#### download weights and cfg
    from :  https://pjreddie.com/darknet/yolo/
    download .weight file inside bin & .cfg inside cfg folder
           
#### run the code with default example
1. 
    ```
   python script.py
    ```

## Main file script.py (example : yolo with python using darkflow)
1)
```python
## import external dependencies

from darkflow.net.build import TFNet
import cv2
## import our dependencies

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
```
## modules/dependencies created by us are included in modules folder

 ### (e.g yolo.py)
  #### methods 
1. initialize yolo
 ```
   init(options,TFNet)
 ```

2. start detection and localization on image with choosen confidence and kind values
 ```
    draw_pred(image,confidence,kind)
 ```
3. ensure that detected object is our needed kind 
 ```
    label_is_(detected_result,kind )
    (e.g label_is_(result,"person)" will return 1 if it's person)
 ```

## Options 
```bash
#using gpu , yolov2 weights
options = {"model": "./cfg/yolo.cfg", "load": "./bin/yolo.weights", "threshold": 0.1,"gpu":0.5}
#using cpu , yolov2 weights
options = {"model": "./cfg/yolo.cfg", "load": "./bin/yolo.weights", "threshold": 0.1}

```

@darkflow repo https://github.com/thtrieu/darkflow
@opencv repo https://github.com/opencv/opencv


