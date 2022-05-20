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

camera_matrix = np.asarray(labeling_info["get_optimal_new_matrix"], dtype=np.float32)
CAMERA_HEIGHT = 16
fov_x = labeling_info["fov_x"]

# def callback(data) :
#     global obj
#     obj = data.bounding_boxes

def callback(data) :
    global bb
    global pingpong_list

    pingpong_list = []
    for bbox in data.bounding_boxes:
        # print(bbox.ymax)
        bb = bbox
        pingpong_list.append([bb.id, bb.xmin, bb.xmax, bb.ymin, bb.ymax])


def demoDist(plist):
    global bb
    global camera_matrix
    CAMERA_HEIGHT = 16

    # # distance = f * height / img(y)
    # # ?/? ???? ??? ??? ??, ????
    # # FOV ??? ?? -> ?/? ??? ????.

    index = 0
    for ping_bbox in plist:
        id, xmin, xmax, ymin, ymax= ping_bbox

        w = xmax - xmin
        h = ymax - ymin

        cx = (xmax + xmin) // 2
        cy = (ymax + ymin) // 2


        # Normalized Image Plane
        y_norm = (ymax - camera_matrix[1][2]) / camera_matrix[1][1]
        distance = 1 * CAMERA_HEIGHT / y_norm

        if (cx >= 320):
            azimuth = -85
        else:
            azimuth = 85

        dz = distance * math.cos(azimuth)
        dx = distance * math.sin(azimuth)

        print(int(distance))
        print(int(dz))
        print(int(dx))
        print("-------------")

        # cv2.rectangle(image, (int(cx - w/2), int(cy - h/2)), (int(cx + w/2), int(cy + h/2)), (0, 0, 255), 3)
        # cv2.putText(image, f"{int(distance)}", (int(cx - w/2), int(cy - h/2+25)), 1, 2, (255, 0, 0), 2)
        index += 1




def focal_length_distanc_list():
    global bb
    global camera_matrix
    global pingpong_list
    global CAMERA_HEIGHT
    global fov_x
    adjust_y = 1
    adjust_x = 1

    pingpong_distance = []
    '''
    pingpong list format: xyxy
    '''

    for pingpong in pingpong_list:
        x1 = pingpong[1]
        y1 = pingpong[2]
        x2 = pingpong[3]
        y2 = pingpong[4]
        # print(x1,y1,x2,y2)

        y_norm = (y2 - camera_matrix[1][2]*416.0/480) / (camera_matrix[1][1]*416.0/480)
        # y_norm = (y2 - 216*416.0/480) / (camera_matrix[1][1]*416.0/480)
        y_distance = int(1 * CAMERA_HEIGHT / y_norm) * adjust_y
        x_angle = fov_x * ((float(x1)+x2)/2-camera_matrix[0][2]*416.0/640)/416

        x_distance = y_distance * math.tan(x_angle) * adjust_x

        d = int(math.sqrt(y_distance**2 + x_distance**2))
        print(d)
        
        
        # print(x_angle)
        # print(x_distance,y_distance)
        pingpong_distance.append(
            list(map(int, [bb.id, x_distance, y_distance])))

            # cv2.rectangle(tmp, (int(cx-w/2), int(cy-h/2)),
            #               (int(cx+w/2), int(cy+h/2)), (0, 0, 255), 3)
            # cv2.putText(tmp, f"{int(cls)}-{y_distance}cm",
            #             (int(cx-w/2), int(cy-h/2-10)), 1, 2, (255, 255, 0), 1)
    return pingpong_distance


def visualize_location(pingpong_distance):
    pingpong_distance = np.array(pingpong_distance)
    if len(pingpong_distance) == 0:
        return
    cls = pingpong_distance[:, 0]
    x = pingpong_distance[:, 1]
    y = pingpong_distance[:, 2]

    white_idx = (cls == 0)
    white_x = x[white_idx]*6//2
    white_y = y[white_idx]*6//2
    orange_idx = (cls == 1)
    orange_x = x[orange_idx]*6//2
    orange_y = y[orange_idx]*6//2

    white_board = np.ones((600, 600, 3), dtype=np.uint8)*255
    for idx in range(len(white_x)):
        cv2.circle(
            white_board, (white_x[idx]+300, 600-(white_y[idx]+100)), 5, (50, 50, 50), cv2.FILLED)
        cv2.putText(white_board, str(white_x[idx]//3)+", "+str(white_y[idx]//3),
                    (white_x[idx]+300-30, 600-10-(white_y[idx]+100)), 1, 1, (50, 50, 50), 2)

    for idx in range(len(orange_x)):
        cv2.circle(
            white_board, (orange_x[idx]+300, 600-(orange_y[idx]+100)), 5, (0, 80, 200), cv2.FILLED)
        cv2.putText(white_board, str(orange_x[idx]//3)+", "+str(orange_y[idx]//3),
                    (orange_x[idx]+300-30, 600-10-(orange_y[idx]+100)), 1, 1, (0, 80, 200), 2)
    cv2.imshow("white", white_board)
    cv2.waitKey(1)
    return white_board



rospy.init_node('trt_dd')
rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, callback, queue_size=1)

rate = rospy.Rate(10)

while not rospy.is_shutdown():

    # demoDist()

    # print(distance_out(obj))
    if bb != 0:
        pingpong_distance = focal_length_distanc_list()
        white_board = visualize_location(pingpong_distance)

        
    rate.sleep()