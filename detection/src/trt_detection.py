#!/usr/bin/env python

import rospy, time
from xycar_msgs.msg import xycar_motor
from detection.msg import BoundingBoxes, BoundingBox

class Detection:
    def __init__(self):
        rospy.Subscriber('/yolov3_trt_ros/detections', BoundingBoxes, self.callback, queue_size=1)
        self.pub = rospy.Publisher('xycar_motor',xycar_motor,queue_size=1)

    def detection(self, obj_id):
        motor_msg = xycar_motor()
        if obj_id == 0:
            print("white")
            motor_msg.speed = 0
            motor_msg.angle = 0
            self.pub.publish(motor_msg)
        elif obj_id == 1:
            print("orange")
            motor_msg = xycar_motor()
            motor_msg.speed = 0
            motor_msg.angle = 0
            self.pub.publish(motor_msg)

    def callback(self, data) :
        obj_id = -1
        for bbox in data.bounding_boxes:
            print(bbox.id)
            obj_id = bbox.id
            self.detection(obj_id)

if __name__ == "__main__":
    rospy.init_node('trt_detection')
    ROS = Detection()
    rospy.spin()
