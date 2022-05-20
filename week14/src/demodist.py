#!/usr/bin/env python

import json, rospy, os, math, sys
import cv2
import numpy as np
from week14.msg import BoundingBoxes, BoundingBox

json_file_path = "/home/nvidia/xycar_ws/src/week14/src/camera_matrix_aug.json"

bb = 0
pingpong_list = []

with open(json_file_path, "r") as json_file:
    labeling_info = json.load(json_file)

camera_matrix = np.asarray([[342.34044020257693, 0.0, 314.5118577260303], [0.0, 342.9577609580139, 239.73684312070492], [0.0, 0.0, 1.0]], dtype=np.float32)
CAMERA_HEIGHT = 16
fov_x = labeling_info["fov_x"]


def callback(data) :
    global bb
    global pingpong_list

    pingpong_list = []
    for bbox in data.bounding_boxes:
        # print(bbox.ymax)
        bb = bbox
        pingpong_list.append([bb.id, bb.xmin, bb.xmax, bb.ymin, bb.ymax])

    demoDist(pingpong_list)


def demoDist(plist):
    global bb
    global camera_matrix
    global fov_x

    CAMERA_HEIGHT = 16

    # # distance = f * height / img(y)
    # # ?/? ???? ??? ??? ??, ????
    # # FOV ??? ?? -> ?/? ??? ????.

    index = 0

    for ping_bbox in plist:
        id, xmin, ymin, xmax, ymax= ping_bbox

        w = xmax - xmin
        h = ymax - ymin

        cx = (xmax + xmin) // 2
        cy = (ymax + ymin) // 2

        # Normalized Image Plane

        a = camera_matrix[1][2] * 416 / 480
        b = camera_matrix[1][1] * 416 / 480

        y_norm = (ymax - a) / b
        # y_norm = (ymax - camera_matrix[1][2]) / camera_matrix[1][1]

        distance = 1 * CAMERA_HEIGHT / y_norm

        azimuth = (camera_matrix[0][2]*416/640 - cx) * fov_x / 416 
    
        dz = distance * math.cos(azimuth)
        dx = distance * math.sin(azimuth)
        c = distance * math.tan(azimuth)

        print(distance)
        # print("=========================")



        # print(int(distance))
        # print(int(dz))
        # print(int(dx))
        # print("-------------")

        # cv2.rectangle(image, (int(cx - w/2), int(cy - h/2)), (int(cx + w/2), int(cy + h/2)), (0, 0, 255), 3)
        # cv2.putText(image, f"{int(distance)}", (int(cx - w/2), int(cy - h/2+25)), 1, 2, (255, 0, 0), 2)
        index += 1



rospy.init_node('trt_dd')
rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, callback, queue_size=1)

rate = rospy.Rate(10)


while not rospy.is_shutdown():
    rate.sleep()
