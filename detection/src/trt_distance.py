#!/usr/bin/env python

import json, rospy, os, math, sys
import cv2
import numpy as np
import pandas as pd
from detection.msg import BoundingBoxes, BoundingBox


class Distance:
    def __init__(self):
        json_file_path = "/home/nvidia/xycar_ws/src/detection/info/camera_matrix_aug.json"
        self.csv_export_list = []

        with open(json_file_path, "r") as json_file:
            labeling_info = json.load(json_file)

        self.camera_matrix = np.asarray(labeling_info["get_optimal_new_matrix"], dtype=np.float32)
        self.CAMERA_HEIGHT = 16
        self.fov_x = labeling_info["fov_x"]
        rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, self.callback, queue_size=1)


    def callback(self, data) :
        global csv_export_list
        bb = 0
        pingpong_list = []

        for bbox in data.bounding_boxes:
            bb = bbox
            pingpong_list.append([bb.id, bb.xmin, bb.xmax, bb.ymin, bb.ymax])

        if not bb:
            pingpong_distance = self.focal_length_distanc_list(bb, pingpong_list)
            self.csv_export_list.append(pingpong_distance)


    def focal_length_distanc_list(self, bb, pingpong_list):
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

            y_norm = (y2 - self.camera_matrix[1][2]*416.0/480) / (self.camera_matrix[1][1]*416.0/480)
            # y_norm = (y2 - 216*416.0/480) / (camera_matrix[1][1]*416.0/480)
            y_distance = int(1 * self.CAMERA_HEIGHT / y_norm)
            x_angle = self.fov_x * ((float(x1)+x2)/2-self.camera_matrix[0][2]*416.0/640)/416

            x_distance = y_distance * math.tan(x_angle)
            y_distance = y_distance - 16

            d = int(math.sqrt(y_distance**2 + x_distance**2))

            pingpong_distance.append(
                list(map(int, [x_distance, y_distance])))

        return pingpong_distance


    def visualize_location(self, pingpong_distance):
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


if __name__ == "__main__":
    rospy.init_node('trt_distance')
    ROS = Distance()
    rospy.spin()

    line = pd.DataFrame(ROS.csv_export_list)
    line.to_csv('/home/nvidia/xycar_ws/src/week14/distance.csv',header=False, index=False)
