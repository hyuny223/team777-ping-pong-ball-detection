import cv2
import glob
import numpy as np
import sys, os, time
from matplotlib import pyplot as plt
import json

class calibrate:
    # Chessboard Config
    BOARD_WIDTH = 8
    BOARD_HEIGHT = 6
    SQUARE_SIZE = 0.026

    DISPLAY_IMAGE = True
    INTRINSIC_DIR = "images_past/"
    MATRIX_FILE_NAME = "camera_matrix.json"
    PINGPONG_IMG_DIR = "output_pingpong/"
    UNDISTORTED_IMG_DIR = "undistorted/"

    def __init__(self):
        pass

    def get_camera_matrix(self):
        if os.path.isfile(self.MATRIX_FILE_NAME):
            # print("hi")
            with open(self.MATRIX_FILE_NAME, "r") as f:
                matrix_info = json.load(f)
                camera_matrix = np.array(matrix_info["cam_intrinsic"], dtype=np.float32)
                ret = matrix_info["ret"]
                dist_coeffs = np.array(matrix_info["dist_coeffs"], dtype=np.float32)
                rvecs = matrix_info["rvecs"]
                tvecs = matrix_info["tvecs"]
                #print(camera_matrix)
                return ret, camera_matrix,  dist_coeffs, rvecs, tvecs
        else:
            return self.calculate_camera_matrix()
    def calculate_camera_matrix(self):
        # Get Image Path List
        image_path_list = sorted(glob.glob(self.INTRINSIC_DIR+"*.png"))

        

        # Window-name Config
        window_name = "Intrinsic Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Calibration Config
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )
        pattern_size = (self.BOARD_WIDTH, self.BOARD_HEIGHT)
        counter = 0

        image_points = list()

        for image_path in image_path_list:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            # OpneCV Color Space -> BGR
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(image_gray, pattern_size, flags)
            if ret:
                if self.DISPLAY_IMAGE:
                    image_draw = cv2.drawChessboardCorners(image, pattern_size, corners, ret)
                    for corner in corners:
                        counter_text = str(counter)
                        point = (int(corner[0][0]), int(corner[0][1]))
                        cv2.putText(image_draw, counter_text, point, 2, 0.5, (0, 0, 255), 1)
                        counter += 1

                    counter = 0
                    # cv2.imshow(window_name, image_draw)
                    # cv2.waitKey(0)
        
                image_points.append(corners)
            else:
                print(image_path)
                os.remove(image_path)


        object_points = list()

        for i in range(len(image_path_list)):
            object_point = list()
            height = 0
            for _ in range(0, self.BOARD_HEIGHT):
                # Loop Width -> 9
                width = 0
                for _ in range(0, self.BOARD_WIDTH):
                    # Loop Height -> 6
                    point = [[height, width, 0]]
                    object_point.append(point)
                    width += self.SQUARE_SIZE
                height += self.SQUARE_SIZE
            object_points.append(object_point)



        object_points = np.asarray(object_points, dtype=np.float32)

        tmp_image = cv2.imread(self.INTRINSIC_DIR+"0.png", cv2.IMREAD_ANYCOLOR)
        image_shape = np.shape(tmp_image)

        image_height = image_shape[0]
        image_width = image_shape[1]
        image_size = (image_width, image_height)
        print("start calibration")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

        
        with open(self.MATRIX_FILE_NAME, "w") as f:
            camera_mat_dict = {}
            camera_mat_dict["ret"] = ret
            camera_mat_dict["cam_intrinsic"] = camera_matrix.tolist()
            camera_mat_dict["dist_coeffs"] = dist_coeffs.tolist()
            camera_mat_dict["rvecs"] = 0 #rvecs
            camera_mat_dict["tvecs"] = 0 #tvecs
            camera_matrix_json = json.dumps(camera_mat_dict)
            f.write(camera_matrix_json)
        return ret, camera_matrix, dist_coeffs, rvecs, tvecs

        """
        위에보다 아래가 더 빠르다
        """
    def undistort(self, image_path_list, camera_matrix, dist_coeffs):
        tmp_image = cv2.imread("output_white/"+"50.png", cv2.IMREAD_ANYCOLOR)
        image_shape = np.shape(tmp_image)

        image_height = image_shape[0]
        image_width = image_shape[1]
        image_size = (image_width, image_height)
        mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, None, image_size, cv2.CV_32FC1)
        count = 1

        for image_path in image_path_list:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_undist = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            file_name = self.UNDISTORTED_IMG_DIR +str(count) + ".png"
            cv2.imwrite(file_name, image_undist)
            count+=1
        # cv2.imshow("image", image)
        # cv2.imshow("image_undist", image_undist)
        # cv2.waitKey()
        
### 구부러진 사진을 intrinsic matrix를 통해 펴는 작업
cal = calibrate()
# print(cal.get_camera_matrix())
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cal.get_camera_matrix()
# print(cal.get_camera_matrix())
cal.undistort(sorted(glob.glob("output_white/*.png")), camera_matrix, dist_coeffs)