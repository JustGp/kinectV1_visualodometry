#!/usr/bin/env python3

#imports
import freenect
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np

#Nodo
class kinect_depth_publisher(Node):
    def __init__(self):
        super().__init__("kinect_depth_publisher")
# Maybe cambiar por tipo uint8 si marca error
        self.publish = self.create_publisher(
            Image, "/kinect_depth", 10
        )


        ##Loop
        self.timer = self.create_timer(0.5, self.kinect_depth)
    def kinect_depth(self):
        bridge = CvBridge()
        array,_ = freenect.sync_get_depth()        
        #array = array.astype(np.uint8)
        depth_message = bridge.cv2_to_imgmsg(array, encoding="passthrough")
        self.publish.publish(depth_message)



    def cleanup_kinect(self):
        freenect.sync_stop()
        self.get_logger().info("Released")

        
def main(args=None):
    rclpy.init(args=args)
    node = kinect_depth_publisher()
    rclpy.spin(node)

    node.cleanup_kinect()
    node.destroy_node()
    rclpy.shutdown()