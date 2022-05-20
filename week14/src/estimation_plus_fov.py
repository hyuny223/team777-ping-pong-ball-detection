import json
import cv2
import os
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import random


def changing_format_nxywh_to_xyxy(pingpong_list):
    pingpong_list_aug = []
    for pingpong in pingpong_list:
        cls, cx, cy, w, h = pingpong
        cx *= 640
        cy *= 480
        w *= 640
        h *= 480
        x1 = cx-w/2
        x2 = cx+w/2
        y1 = cy-h/2
        y2 = cy+h/2
        pingpong_list_aug.append(list(map(int, [cls, x1, y1, x2, y2])))
    return pingpong_list_aug


def distance_estimation(img, pingpong_list):
    tmp = copy.deepcopy(img)
    '''
    pingpong_list format xyxy
    '''
    fov_x = math.atan(45/50)
    fov_y = math.atan(33/50)
    json_file_path = "./camera_matrix.json"
    with open(json_file_path, "r") as json_file:
        labeling_info = json.load(json_file)

    cam_matrix = np.array(labeling_info['cam_intrinsic'])
    dist_coeff = np.array(labeling_info['dist_coeffs'])
    CAMERA_HEIGHT = 16

    pingpong_distance = []
    for pingpong in pingpong_list:
        cls, x1, y1, x2, y2 = pingpong

        y_norm = (y2 - cam_matrix[1][2]) / cam_matrix[1][1]
        y_distance = int(1 * CAMERA_HEIGHT / y_norm)
        x_angle = fov_x * ((x1+x2)/2-320)/320
        x_distance = y_distance * math.tan(x_angle)
        pingpong_distance.append(
            list(map(int, [cls, x_distance, y_distance])))

        cv2.rectangle(tmp, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(tmp, f"{int(cls)},{y_distance}cm",
                    (x1, y1-10), 1, 2, (255, 255, 0), 1)
    return tmp, pingpong_distance


def visualize_location(pingpong_distance):
    pingpong_distance = np.array(pingpong_distance)
    cls = pingpong_distance[:, 0]
    x = pingpong_distance[:, 1]
    y = pingpong_distance[:, 2]

    white_idx = (cls == 0)
    white_x = x[white_idx]*8//2
    white_y = y[white_idx]*8//2
    orange_idx = (cls == 1)
    orange_x = x[orange_idx]*8//2
    orange_y = y[orange_idx]*8//2

    white_board = np.ones((600, 600, 3), dtype=np.uint8)*255
    for idx in range(len(white_x)):
        cv2.circle(
            white_board, (white_x[idx]+300, 600-(white_y[idx]+100)), 3, (50, 50, 50), cv2.FILLED)
        cv2.putText(white_board, f"{white_x[idx]//4}, {white_y[idx]//4}",
                    (white_x[idx]+300-30, 600-10-(white_y[idx]+100)), 1, 1, (50, 50, 50), 2)

    for idx in range(len(orange_x)):
        cv2.circle(
            white_board, (orange_x[idx]+300, 600-(orange_y[idx]+100)), 3, (0, 80, 200), cv2.FILLED)
        cv2.putText(white_board, f"{orange_x[idx]//4}, {orange_y[idx]//4}",
                    (orange_x[idx]+300-30, 600-10-(orange_y[idx]+100)), 1, 1, (0, 80, 200), 2)
    cv2.imshow("white", white_board)
    return white_board


if __name__ == "__main__":
    pingpong_list = []
    with open("./resources/2.txt", "r") as f:
        while True:
            temp = f.readline().strip()
            if temp == "":
                break
            temp = list(map(float, temp.split()))
            pingpong_list.append(temp)
    pingpong_list = changing_format_nxywh_to_xyxy(pingpong_list)
    # print(pingpong_list)
    img = cv2.imread("./resources/2.png")
    img2, pingpong_distance = distance_estimation(img, pingpong_list)

    img_h = cv2.hconcat([img, img2])
    visualize_location(pingpong_distance)
    cv2.imshow("img", img_h)
    cv2.waitKey()
