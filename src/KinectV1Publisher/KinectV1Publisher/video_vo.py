import freenect
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
import matplotlib.pyplot as piplop

from ultralytics import YOLO
# Remember Multisense s21


class video_odometry(Node):
    def __init__(self):
        super().__init__("video_odometry")

        self.subscriber = self.create_subscription(
            Image, "/kinect_camera",self.visual_odometry,10
        )

        # Publisher for publishing trajectory as Path
        self.publisher = self.create_publisher(Path, "/video_vo", 10)
        
        # Store pose history for path
        self.pose_history = []
        
        # Kinect V1 intrinsic parameters
        self.declare_parameter('fx', 525.0)  # focal length x
        self.declare_parameter('fy', 525.0)  # focal length y
        self.declare_parameter('cx', 319.5)  # optical center x
        self.declare_parameter('cy', 239.5)  # optical center y
        
        # Build camera matrix from parameters
        fx = self.get_parameter('fx').value
        fy = self.get_parameter('fy').value
        cx = self.get_parameter('cx').value
        cy = self.get_parameter('cy').value
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        self.get_logger().info(f"Using camera matrix:\n{self.K}")
        
        # Store previous image for visual odometry comparisons
        self.prev_image = None
        # Reuse CvBridge instance
        self.bridge = CvBridge()
        
        # Initialize global pose tracking
        self.global_pose = np.eye(4)  # Start at identity transformation
        self.frame_count = 0  # Track number of processed frames


        self.model = YOLO('yolov8n.pt')



    def visual_odometry(self, imageros: Image):
        # Convert ROS Image to OpenCV and to grayscale
        try:
            cv_image = self.bridge.imgmsg_to_cv2(imageros, desired_encoding="bgr8")
            image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().info(f"error en greyificacion: {e}")
            return

        curr = image.copy()

        if self.prev_image is None:
            self.prev_image = curr
            self.get_logger().info("Stored first image as previous frame")
            return

        # --- YOLO detection (puedes reducir frecuencia si quieres) ---
        results = self.model(cv_image)

        vis = cv_image.copy()
        h, w = curr.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255  # 255 -> permitir, 0 -> excluir

        for result in results:
            for box in result.boxes:
                # Clase robusta
                try:
                    cls_raw = box.cls[0]
                    cls_id = int(cls_raw.item()) if hasattr(cls_raw, "item") else int(cls_raw)
                except Exception:
                    continue
                label = self.model.names.get(cls_id, None)
                if label != 'person':
                    continue

                # Coordenadas robustas
                try:
                    xyxy = box.xyxy[0]
                    if hasattr(xyxy, "cpu"):
                        coords = xyxy.cpu().numpy()
                    else:
                        coords = np.array(xyxy)
                    x1f, y1f, x2f, y2f = coords.astype(float).tolist()
                    x1, y1, x2, y2 = map(int, [x1f, y1f, x2f, y2f])
                except Exception:
                    continue

                # Clamp a límites
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                # Evita rects degenerados
                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                mask[y1:y2+1, x1:x2+1] = 0

        # Dilatar para cubrir bordes y pequeños movimientos
        kernel = np.ones((9, 9), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)

        # --- Feature detection con máscara ---
        orb = cv2.ORB_create(3000)

        if self.prev_image.shape != curr.shape:
            self.prev_image = cv2.resize(self.prev_image, (w, h))


# Anadir mascara 
        kp1, des1 = orb.detectAndCompute(self.prev_image, mask_dilated)
        kp2, des2 = orb.detectAndCompute(curr, mask_dilated) 

        if des1 is None or des2 is None:
            self.get_logger().info("No descriptors en uno de los frames despues de enmascar")
            self.prev_image = curr.copy()
            return

        # --- Feature matching ---
        bruteforce = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bruteforce.match(des1, des2)
        good_matches = [m for m in matches if m.distance < 50]
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        N = min(len(good_matches), 100)
        if N < 8:
            self.get_logger().info("No hay suficientes matches buenos")
            self.prev_image = curr.copy()
            return

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches[:N]])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches[:N]])

        E, mask_e = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            self.get_logger().info("Essential matrix no encontrada")
            self.prev_image = curr.copy()
            return

        retval, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask_e)
        if retval < 15:
            self.get_logger().info(f"Poca confianza en recoverPose ({retval} inliers); se omite acumulación")
            self.prev_image = curr.copy()
            return

        transform_mat = np.eye(4)
        transform_mat[:3, :3] = R
        transform_mat[:3, 3] = t.ravel()
        self.global_pose = self.global_pose @ transform_mat

        global_t = self.global_pose[:3, 3]
        global_R = self.global_pose[:3, :3]

        def rot_to_quat(R):
            tr = R[0, 0] + R[1, 1] + R[2, 2]
            if tr > 0:
                S = np.sqrt(tr + 1.0) * 2.0
                qw = 0.25 * S
                qx = (R[2, 1] - R[1, 2]) / S
                qy = (R[0, 2] - R[2, 0]) / S
                qz = (R[1, 0] - R[0, 1]) / S
            else:
                if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
                    S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
                    qw = (R[2, 1] - R[1, 2]) / S
                    qx = 0.25 * S
                    qy = (R[0, 1] + R[1, 0]) / S
                    qz = (R[0, 2] + R[2, 0]) / S
                elif R[1, 1] > R[2, 2]:
                    S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
                    qw = (R[0, 2] - R[2, 0]) / S
                    qx = (R[0, 1] + R[1, 0]) / S
                    qy = 0.25 * S
                    qz = (R[1, 2] + R[2, 1]) / S
                else:
                    S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
                    qw = (R[1, 0] - R[0, 1]) / S
                    qx = (R[0, 2] + R[2, 0]) / S
                    qy = (R[1, 2] + R[2, 1]) / S
                    qz = 0.25 * S
            return np.array([qx, qy, qz, qw])

        quat = rot_to_quat(global_R)
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 1e-6:
            quat = quat / quat_norm

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(global_t[0])
        pose_msg.pose.position.y = 0.0 # Forzar visualizacion en y
        pose_msg.pose.position.z = float(global_t[2])


        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        self.pose_history.append(pose_msg)
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        path_msg.poses = self.pose_history
        self.publisher.publish(path_msg)

        # Actualiza prev_image al final
        self.prev_image = curr.copy()


def main(args=None):
    rclpy.init(args=args)
    node = video_odometry()
    rclpy.spin(node)
    rclpy.shutdown()