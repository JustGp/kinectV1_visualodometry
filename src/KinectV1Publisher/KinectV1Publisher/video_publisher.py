#!/usr/bin/env python3

#imports
import freenect
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import threading 
# Nodo
class kinect_video_publisher(Node):
    def __init__(self):
        super().__init__("kinect_video_publisher")

        # Publisher
        self.publish = self.create_publisher(
            Image, "/kinect_camera", 10 )
        
        # Loop
        self.timer = self.create_timer(0.033, self.kinect_video)


        ## Video from kinect to topic
    def kinect_video(self):
        bridge = CvBridge() # Para las imagenes 
        array, _ = freenect.sync_get_video()
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        video_message = bridge.cv2_to_imgmsg(array, encoding="passthrough")
        self.publish.publish(video_message)


    def cleanup_kinect(self):
        freenect.sync_stop()
        self.get_logger().info("Released")


## Esperar para el escape
def listen_for_q():
    print("Presiona Q para detener el nodo...")
    while True:
        key = cv2.waitKey(100)
        if key == ord('q') or key == ord('Q'):
            rclpy.shutdown()
            break



# Main
def main(args=None):
    rclpy.init(args=args)
    node = kinect_video_publisher()
    rclpy.spin(node)

    esc_thread = threading.Thread(target=listen_for_q, daemon=True)
    esc_thread.start()


    node.cleanup_kinect()
    node.destroy_node()
    rclpy.shutdown()
