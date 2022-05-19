import glob
import numpy as np
import cv2
import sys
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="image sharpening")
    parser.add_argument("--img_path", type=str,
                        help="image directory. for example, ./images/",
                        default="./resources/")
    parser.add_argument("--img_ext", type=str,
                        help="image ext. for example, .png",
                        default=".png")
    parser.add_argument("--visual", type=bool,
                        help="visualization of image True/False",
                        default=False)
    parser.add_argument("--save_path", type=str,
                        help="save path ex) ./output/", default="./output/")
    if (len(sys.argv) == 1):
        # python file opened without any argument
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def start():
    args = parse_args()
    image_path = args.img_path + "*" + args.img_ext
    f_list = glob.glob(image_path)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    if len(f_list) == 0:
        print("no file loaded")
        sys.exit(1)

    for f_name in f_list:
        image = cv2.imread(f_name, cv2.IMREAD_ANYCOLOR)
        image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        if args.visual:
            image_hc = cv2.hconcat([image, image_sharp])
            cv2.imshow("images", image_hc)
            cv2.waitKey(5000)
        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)
        cv2.imwrite(f_name.replace(args.img_path, args.save_path), image_sharp)


if __name__ == "__main__":
    start()
