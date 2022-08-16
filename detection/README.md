# Pingpong Ball Detection

# 1. Goal of this project
The goal of this project is to detect pingpong ball with YOLOv3.  
![image](https://user-images.githubusercontent.com/58837749/184803562-02528a74-0233-4277-9230-36cb18c147b9.png)


# 2. Developed Environment
OS : ubuntu18.04 ROS melodic  
Processor : Nvidia NX  
Nvidia Kernel : Xavier NX Kernel  
<br>
This program is for Xycar(RC-Car). So a "weights(model)" file is converted to tensorrt sutable for architecture of Xycar. So if you want to lighten your model to fit your hardware, you should make your own tensorrt.  
If you don't want, just use the model(a weights file).  
<br>
pips required for this project are written on "requirements.txt"  
```bash
$ pip install -r requirements.txt
```

# 3. Main Scripts
## 1. trt_detection.py
This script is to subscribe detected objects and control next activation of Xycar. In this script, it prints the color of pingpong ball out.  

## 2. trt_distance.py
This script is to compute distance of detected pinpong ball. It is calculated based on Geometrical Distance Estimation.  
![Screenshot from 2022-08-11 20-39-04](https://user-images.githubusercontent.com/58837749/184804788-f1b1a05f-e7f3-446b-8ec1-b257f5bbd00c.png)  

## 3. trt_inference.py
This script is to inference object if you convert a weights file to tensorrt.  

# 4. How to Run
below is the example code.  
```bash
$ roslaunch detection detection.launch
```
