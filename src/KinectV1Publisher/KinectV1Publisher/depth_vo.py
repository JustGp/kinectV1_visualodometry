#!/usr/bin/env python3
# depth_to_path_diagnostic.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math

class DepthToPathMinimal(Node):
    def __init__(self):
        super().__init__('depth_vo')
        self.bridge = CvBridge()

        # topics y parámetros (pueden override desde CLI / launch)
        self.declare_parameter('depth_topic', '/kinect_depth')
        self.declare_parameter('path_topic', '/depth_vo')

        self.declare_parameter('fx', 525.0)
        self.declare_parameter('fy', 525.0)
        self.declare_parameter('cx', 319.5)
        self.declare_parameter('cy', 239.5)
        self.declare_parameter('roi_half', 2)
        self.declare_parameter('min_valid_m', 0.8)
        self.declare_parameter('max_valid_m', 5.0)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('path_max_len', 500)

        depth_topic = self.get_parameter('depth_topic').value
        path_topic = self.get_parameter('path_topic').value

        # subs y pub usando parámetros
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_cb, 10)
        self.path_pub = self.create_publisher(Path, path_topic, 10)

        # estado interno
        self.path = Path()
        self.path.header.frame_id = self.get_parameter('frame_id').value
        
        # contador de frames para diagnóstico
        self.frame_count = 0

        self.get_logger().info(f"Init completado. Subscribed to: {depth_topic}, publishing path to: {path_topic}")
        self.get_logger().info("=" * 60)
        self.get_logger().info("DIAGNOSTIC MODE ENABLED - Detailed logging active")
        self.get_logger().info("=" * 60)

    # --- util: convertir Image -> matriz en metros
    def depth_to_meters(self, img_msg: Image):
        self.get_logger().info("[STEP 1] Converting Image to depth array...")
        
        # Intentar convertir con cv_bridge
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
            self.get_logger().info(f"[STEP 1] ✓ cv_bridge success - shape: {cv_image.shape}, dtype: {cv_image.dtype}")
        except CvBridgeError as e:
            self.get_logger().error(f"[STEP 1] ✗ cv_bridge error: {e}")
            return np.full((0,), np.nan, dtype=np.float32)
        except Exception as e:
            self.get_logger().error(f"[STEP 1] ✗ cv_bridge unexpected error: {e}")
            return np.full((0,), np.nan, dtype=np.float32)

        # detectar encoding
        enc = None
        try:
            enc = img_msg.encoding
            self.get_logger().info(f"[STEP 1] Image encoding: '{enc}'")
        except Exception:
            enc = None
            self.get_logger().warn("[STEP 1] No encoding found in message")

        # Estadísticas del raw image
        try:
            img_min = np.min(cv_image)
            img_max = np.max(cv_image)
            img_mean = np.mean(cv_image)
            self.get_logger().info(f"[STEP 1] Raw image stats - min: {img_min}, max: {img_max}, mean: {img_mean:.2f}")
        except Exception as e:
            self.get_logger().warn(f"[STEP 1] Could not compute image stats: {e}")

        # manejar casos esperados según encoding
        if enc in ('16UC1', '16UC', '16U', '16U;depth'):
            depth_m = cv_image.astype(np.float32) / 1000.0
            self.get_logger().info("[STEP 1] Applied 16UC1 conversion (mm -> m)")
        elif enc in ('32FC1', '32FC', '32F'):
            depth_m = cv_image.astype(np.float32)
            self.get_logger().info("[STEP 1] Using 32FC1 (already in meters)")
        elif enc == 'mono8':
            self.get_logger().info("[STEP 1] Applying MONO8 calibrated conversion...")
            depth_m = 0.800 + (cv_image.astype(np.float32) / 255.0) * 4.200
            self.get_logger().info("[STEP 1] ✓ Mono8 conversion applied: depth_m = 0.800 + (value/255.0) * 4.200")
        else:
            # Heurística
            if cv_image.dtype == np.uint16:
                depth_m = cv_image.astype(np.float32) / 1000.0
                self.get_logger().info("[STEP 1] Heuristic: uint16 -> assuming mm, converting to m")
            elif cv_image.dtype == np.float32 or cv_image.dtype == np.float64:
                depth_m = cv_image.astype(np.float32)
                self.get_logger().info("[STEP 1] Heuristic: float type -> assuming meters")
            else:
                depth_m = cv_image.astype(np.float32)
                try:
                    med = float(np.nanmedian(depth_m))
                except Exception:
                    med = float('nan')
                if not math.isnan(med) and med > 10.0:
                    depth_m = depth_m / 1000.0
                    self.get_logger().info("[STEP 1] Heuristic: large values -> assuming mm")
                else:
                    self.get_logger().info("[STEP 1] Heuristic: treating as meters")

        # Estadísticas post-conversión (simplified to avoid hanging)
        try:
            self.get_logger().info("[STEP 1] Computing depth statistics...")
            # Don't use boolean indexing on large arrays - it can hang
            # Just compute stats directly on the array
            depth_min = float(np.nanmin(depth_m))
            self.get_logger().info(f"[STEP 1] Min computed: {depth_min:.3f}m")
            
            depth_max = float(np.nanmax(depth_m))
            self.get_logger().info(f"[STEP 1] Max computed: {depth_max:.3f}m")
            
            depth_mean = float(np.nanmean(depth_m))
            self.get_logger().info(f"[STEP 1] Mean computed: {depth_mean:.3f}m")
            
            valid_count = int(np.sum(~np.isnan(depth_m)))
            self.get_logger().info(f"[STEP 1] Depth stats - min: {depth_min:.3f}m, max: {depth_max:.3f}m, mean: {depth_mean:.3f}m, valid pixels: {valid_count}/{depth_m.size}")
        except Exception as e:
            self.get_logger().error(f"[STEP 1] ✗ EXCEPTION computing depth stats: {e}")
            import traceback
            self.get_logger().error(f"[STEP 1] Traceback: {traceback.format_exc()}")

        # limpiar valores inválidos (simplified - no need to set 0.0 to nan since conversion doesn't produce zeros)
        try:
            self.get_logger().info("[STEP 1] Cleaning invalid values...")
            # Skip this step for mono8 since the conversion formula can't produce inf or 0
            if enc != 'mono8':
                depth_m[np.isinf(depth_m)] = np.nan
                depth_m[depth_m == 0.0] = np.nan
            self.get_logger().info("[STEP 1] ✓ Cleanup complete")
        except Exception as e:
            self.get_logger().error(f"[STEP 1] ✗ EXCEPTION during cleanup: {e}")

        self.get_logger().info("[STEP 1] ✓ depth_to_meters() complete, returning array")
        return depth_m

    # --- util: extraer Z (mediana en ROI o en toda la imagen si ROI falla) y pixel (u,v)
    def roi_median_depth(self, depth_m):
        self.get_logger().info("[STEP 2] Computing ROI median depth...")
        
        # validar que depth_m tenga forma (h,w)
        if depth_m is None or depth_m.size == 0:
            self.get_logger().error("[STEP 2] ✗ depth_m is None or empty")
            return float('nan'), (0, 0)
        if depth_m.ndim != 2:
            self.get_logger().error(f"[STEP 2] ✗ depth_m.ndim != 2 ({depth_m.ndim})")
            return float('nan'), (0, 0)

        h, w = depth_m.shape
        cx = w // 2
        cy = h // 2
        half = int(self.get_parameter('roi_half').value)
        x0 = max(0, cx - half); x1 = min(w, cx + half + 1)
        y0 = max(0, cy - half); y1 = min(h, cy + half + 1)
        
        self.get_logger().info(f"[STEP 2] Image size: {w}x{h}, center: ({cx},{cy}), ROI: [{x0}:{x1}, {y0}:{y1}]")
        self.get_logger().info(f"[STEP 2] About to extract ROI slice...")
        
        min_m = float(self.get_parameter('min_valid_m').value)
        max_m = float(self.get_parameter('max_valid_m').value)
        
        # Try to extract ROI first
        try:
            roi = depth_m[y0:y1, x0:x1].copy()
            self.get_logger().info(f"[STEP 2] ✓ ROI extracted, shape: {roi.shape}")
            
            # Compute median
            median_roi = float(np.median(roi))
            self.get_logger().info(f"[STEP 2] ROI median: {median_roi:.3f}m")
            
            # Check if median is in valid range
            if median_roi >= min_m and median_roi <= max_m:
                self.get_logger().info(f"[STEP 2] ✓ Valid depth found in ROI: {median_roi:.3f}m")
                return median_roi, (cx, cy)
            else:
                self.get_logger().warn(f"[STEP 2] ROI median {median_roi:.3f}m outside valid range [{min_m}-{max_m}m]")
        except Exception as e:
            self.get_logger().warn(f"[STEP 2] Could not get valid depth from ROI: {e}")
        
        # Fallback: use median of entire valid depth map
        self.get_logger().info("[STEP 2] Trying entire image median as fallback...")
        try:
            # Filter to valid range
            valid_mask = (depth_m >= min_m) & (depth_m <= max_m)
            valid_count = int(np.sum(valid_mask))
            self.get_logger().info(f"[STEP 2] Valid pixels in image: {valid_count}/{depth_m.size}")
            
            if valid_count > 0:
                valid_depths = depth_m[valid_mask]
                median_all = float(np.median(valid_depths))
                self.get_logger().info(f"[STEP 2] ✓ Using image median: {median_all:.3f}m")
                
                # Find a pixel with this approximate depth for (u,v)
                diff = np.abs(depth_m - median_all)
                min_idx = np.argmin(diff)
                py, px = np.unravel_index(min_idx, depth_m.shape)
                self.get_logger().info(f"[STEP 2] Representative pixel: ({px}, {py})")
                
                return median_all, (int(px), int(py))
            else:
                self.get_logger().error("[STEP 2] ✗ No valid depth in entire image")
                return float('nan'), (cx, cy)
        except Exception as e:
            self.get_logger().error(f"[STEP 2] ✗ Failed to compute image median: {e}")
            return float('nan'), (cx, cy)

    # --- util: construir PoseStamped
    def make_pose_stamped(self, x, y, z, frame_id, stamp):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return pose

    # --- callback principal
    def depth_cb(self, msg: Image):
        self.frame_count += 1
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"FRAME #{self.frame_count} - Processing new depth image")
        self.get_logger().info("=" * 60)
        
        # logs basicos
        try:
            enc = msg.encoding
        except Exception:
            enc = "<no encoding>"
        self.get_logger().info(f"[STEP 0] Received Image - stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}, encoding: '{enc}', size: {msg.width}x{msg.height}")

        # 1) convertir a metros y limpiar
        depth_m = self.depth_to_meters(msg)
        if depth_m is None or depth_m.size == 0:
            self.get_logger().error("[STEP 1] ✗ FAILED - No depth data available after conversion")
            return

        # 2) obtener Z representativa y pixel central
        z, (u, v) = self.roi_median_depth(depth_m)
        if math.isnan(z):
            self.get_logger().error('[STEP 2] ✗ FAILED - Invalid depth in ROI')
            return

        # 3) intrínsecos desde parámetros
        self.get_logger().info("[STEP 3] Unprojecting to 3D coordinates...")
        fx = float(self.get_parameter('fx').value)
        fy = float(self.get_parameter('fy').value)
        cx = float(self.get_parameter('cx').value)
        cy = float(self.get_parameter('cy').value)
        
        self.get_logger().info(f"[STEP 3] Camera intrinsics - fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
        
        if fx == 0 or fy == 0:
            self.get_logger().error('[STEP 3] ✗ FAILED - Invalid intrinsics (fx or fy = 0)')
            return

        # 4) pixel -> coordenada en cámara (métrico)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        self.get_logger().info(f"[STEP 3] ✓ 3D point - pixel ({u},{v}) -> camera coords (x={x:.3f}m, y={y:.3f}m, z={z:.3f}m)")

        # 5) stamp
        self.get_logger().info("[STEP 4] Setting timestamp...")
        if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()
            self.get_logger().info(f"[STEP 4] Using current time: {stamp.sec}.{stamp.nanosec}")
        else:
            stamp = msg.header.stamp
            self.get_logger().info(f"[STEP 4] Using message stamp: {stamp.sec}.{stamp.nanosec}")

        # 6) crear PoseStamped y añadir al Path
        self.get_logger().info("[STEP 5] Creating and adding pose to path...")
        frame_id = self.get_parameter('frame_id').value
        pose_stamped = self.make_pose_stamped(x, y, z, frame_id, stamp)
        self.path.poses.append(pose_stamped)
        self.path.header.stamp = stamp
        self.path.header.frame_id = frame_id
        
        self.get_logger().info(f"[STEP 5] ✓ Pose added - path length now: {len(self.path.poses)}")

        # 7) limitar longitud del path
        max_len = int(self.get_parameter('path_max_len').value)
        if len(self.path.poses) > max_len:
            self.path.poses = self.path.poses[-max_len:]
            self.get_logger().info(f"[STEP 5] Path trimmed to max length: {max_len}")

        # 8) publicar
        self.get_logger().info("[STEP 6] Publishing path...")
        self.path_pub.publish(self.path)
        self.get_logger().info(f'[STEP 6] ✓ SUCCESS - Path published! Length: {len(self.path.poses)}, Latest pose: (x={x:.3f}, y={y:.3f}, z={z:.3f})')
        self.get_logger().info("=" * 60)

def main(args=None):
    rclpy.init(args=args)
    node = DepthToPathMinimal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()