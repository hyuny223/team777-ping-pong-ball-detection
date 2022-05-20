import json
import cv2
import os
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import random


def focal_length_distance(img, pingpong_list):
    fov_x = math.atan(45/50)
    fov_y = math.atan(33/50)

    pingpong_distance = []
    tmp = copy.deepcopy(img)
    '''
    pingpong list format: Normalized cx, cy, w, h
    '''
    json_file_path = "./camera_matrix.json"
    with open(json_file_path, "r") as json_file:
        labeling_info = json.load(json_file)

    cam_matrix = np.array(labeling_info['cam_intrinsic'])
    dist_coeff = np.array(labeling_info['dist_coeffs'])
    CAMERA_HEIGHT = 16
    for pingpong in pingpong_list:
        cls, cx, cy, w, h = pingpong
        cx *= 640
        cy *= 480
        w *= 640
        h *= 480

        y_norm = (cy+h/2 - cam_matrix[1][2]) / cam_matrix[1][1]
        y_distance = int(1 * CAMERA_HEIGHT / y_norm)
        x_angle = fov_x * (cx-320)/320
        x_distance = y_distance * math.tan(x_angle)
        pingpong_distance.append(
            list(map(int, [cls, x_distance, y_distance])))

        cv2.rectangle(tmp, (int(cx-w/2), int(cy-h/2)),
                      (int(cx+w/2), int(cy+h/2)), (0, 0, 255), 3)
        cv2.putText(tmp, f"{int(cls)}-{y_distance}cm",
                    (int(cx-w/2), int(cy-h/2-10)), 1, 2, (255, 255, 0), 1)
    return tmp, pingpong_distance


def fov_distance(img, pingpong_list):
    tmp = copy.deepcopy(img)
    '''
    pingpong list format: Normalized cx, cy, w, h
    '''
    json_file_path = "./camera_matrix.json"
    with open(json_file_path, "r") as json_file:
        labeling_info = json.load(json_file)

    cam_matrix = np.array(labeling_info['cam_intrinsic'])
    dist_coeff = np.array(labeling_info['dist_coeffs'])
    fov_x = math.atan(45/50)
    fov_y = math.atan(33/50)

    for pingpong in pingpong_list:
        cls, cx, cy, w, h = pingpong
        cx *= 640
        cy *= 480
        w *= 640
        h *= 480

        azimuth_x = (cy-320)/320*fov_x
        dx = cam_matrix[0][0]*w/2/(math.tan(azimuth_x))
        dx /= (w/2/(math.tan(azimuth_x))-cam_matrix[0][0])

        azimuth_y = (cy-240)/240*fov_y
        dy = cam_matrix[1][1]*h/2/(math.tan(azimuth_y))
        dy /= (h/2/(math.tan(azimuth_y))-cam_matrix[1][1])

        dd = math.sqrt(dx**2 + dy**2)
        cv2.rectangle(tmp, (int(cx-w/2), int(cy-h/2)),
                      (int(cx+w/2), int(cy+h/2)), (0, 0, 255), 3)
        cv2.putText(tmp, f"{int(cls)}-{int(dd)}cm",
                    (int(cx-w/2), int(cy-h/2-10)), 1, 2, (255, 255, 0), 1)
    return tmp


def visualize_location(pingpong_distance):
    pingpong_distance = np.array(pingpong_distance)
    cls = pingpong_distance[:, 0]
    x = pingpong_distance[:, 1]
    y = pingpong_distance[:, 2]

    white_idx = (cls == 0)
    white_x = x[white_idx]
    white_y = y[white_idx]
    orange_idx = (cls == 1)
    orange_x = x[orange_idx]
    orange_y = y[orange_idx]
    f = plt.figure()
    plt.ylim([0, max(y)+20])
    plt.scatter(orange_x, orange_y, s=5*5, c='orange')
    plt.scatter(white_x, white_y, s=5*5, c="#C8B7BE")
    img = figure_to_array(f)[:, :, :]

    return img


def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


if __name__ == "__main__":
    pingpong_list = []
    with open("./resources/2.txt", "r") as f:
        while True:
            temp = f.readline().strip()
            if temp == "":
                break
            temp = list(map(float, temp.split()))
            pingpong_list.append(temp)
    img = cv2.imread("./resources/2.png", cv2.IMREAD_ANYCOLOR)
    img1, pingpong_distance = focal_length_distance(img, pingpong_list)
    img2 = fov_distance(img, pingpong_list)

    img_h = cv2.hconcat([img1, img2])
    location_img = visualize_location(pingpong_distance)

    cv2.imshow("img", img_h)
    cv2.imshow("location", location_img)
    cv2.waitKey()
