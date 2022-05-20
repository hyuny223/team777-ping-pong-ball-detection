#!/usr/bin/env python2

import json, rospy, cv2, os
from cv2 import undistort
# import matplotlib.pyplot as plt
import os
import numpy as np
from week14.msg import BoundingBoxes, BoundingBox

json_file_path = os.path.join("data", "camera_matrix.json")

window_name = "Perception"
obj = 0

with open(json_file_path, "r") as json_file:
    labeling_info = json.load(json_file)


camera_matrix = np.asarray(labeling_info["cam_intrinsic"], dtype=np.float32)
# dist_coeff = np.asarray(labeling_info["calib"]["cam01"]["distortion"], dtype=np.float32)


# labeling = labeling_info["frames"][0]["annos"]
# class_names = labeling["names"]
# boxes_2d = labeling["boxes_2d"]["cam01"]

CAMERA_HEIGHT = 1.3

# distance = f * height / img(y)
# 종/횡 방향으로 분리된 거리가 아닌, 직선거리
# FOV 정보를 알면 -> 종/횡 분리가 가능하다.

def cal(bbox):
    global CAMERA_HEIGHT, camera_matrix
    
    width = bbox.xmax - bbox.xmin
    height = bbox.ymax - bbox.ymin

    # Normalized Image Plane
    y_norm = (bbox.ymax - camera_matrix[1][2]) / camera_matrix[1][1]

    distance = 1 * CAMERA_HEIGHT / y_norm

    # cv2.rectangle(undist_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    # cv2.putText(undist_image, f"{index}-{class_name}-{int(distance)}", (xmin, ymin+25), 1, 2, (255, 255, 0), 2)
    # index += 1
    return distance

rospy.init_node('trt_detector')
rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, callback, queue_size=1)

rate = rospy.Rate(10)

def callback(data):
    global obj
    obj = data.bounding_boxes

while not rospy.is_shutdown():
    print(cal(obj))

    rate.sleep()
