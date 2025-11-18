#!/usr/bin/env python3
# imports
import freenect
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np

# Nodo
class KinectVideoPublisher(Node):
    def __init__(self):
        super().__init__("kinect_video_publisher")
        # Publishers
        self.video_publisher = self.create_publisher(Image, "/kinect_camera", 10)
        self.depth_publisher = self.create_publisher(Image, "/kinect_depth", 10)
        # Timers
        self.video_timer = self.create_timer(0.033, self.kinect_video)
        self.depth_timer = self.create_timer(0.033, self.kinect_depth)
        # Para convertir imagen
        self.bridge = CvBridge()
        # Flag para controlar el shutdown
        self.running = True
        # Crear ventana una sola vez
    def kinect_video(self):
        if not self.running:
            return
        try:
            array, _ = freenect.sync_get_video()
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            video_message = self.bridge.cv2_to_imgmsg(array, encoding="bgr8")
            self.video_publisher.publish(video_message)

        # Mostrar ventana vacía (opcional)
        # blank = np.zeros((480, 640, 3), dtype=np.uint8)
        # cv2.imshow("Kinect Video (presiona Q para salir)", blank)

        except Exception as e:
            self.get_logger().error(f"Error en kinect_video: {e}")

            
    def kinect_depth(self):
        if not self.running:
            return
        try:
            array, _ = freenect.sync_get_depth()
            array_8bit = cv2.convertScaleAbs(array, alpha=(255.0 / 2048.0))
            depth_message = self.bridge.cv2_to_imgmsg(array_8bit, encoding="mono8")
            self.depth_publisher.publish(depth_message)
        except Exception as e:
            self.get_logger().error(f"Error en kinect_depth: {e}")
    
    def cleanup_kinect(self):
        self.running = False
        freenect.sync_stop()
        self.get_logger().info("Kinect released")
        cv2.destroyAllWindows()

# Función para escuchar la tecla Q en la ventana

# Main
def main(args=None):
    rclpy.init(args=args)
    node = KinectVideoPublisher()
    
    # Hilo para escuchar la tecla Q
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup_kinect()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

