import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge

from ultralytics import YOLO
import numpy as np
import cv2


class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")

        self.bridge = CvBridge()
        self.model = YOLO("yolov8s.pt")

        # Kamera sub
        self.sub = self.create_subscription(
            Image,
            "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image",
            self.image_callback,
            10
        )

        # Yayınlayıcılar
        self.pub_img = self.create_publisher(Image, "/yolo/detections_img", 10)
        self.pub_det = self.create_publisher(Detection2DArray, "/yolo/detections", 10)

        self.get_logger().info("YOLO Node Started")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # YOLO inference (resize otomatik)
        results = self.model(frame, conf=0.25, verbose=False)
        boxes = results[0].boxes

        # Annotated görüntü
        annotated = results[0].plot()
        img_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
        img_msg.header = msg.header
        img_msg.header.frame_id = "camera_link"
        self.pub_img.publish(img_msg)

        # BEV uyumlu detection array
        det_array = Detection2DArray()
        det_array.header = msg.header
        det_array.header.frame_id = "camera_link"

        # Kamera gerçek çözünürlüğü
        cam_w = msg.width   # =1280
        cam_h = msg.height  # =960

        # YOLO'nun işlediği çözünürlük (genelde 640x480)
        input_h, input_w = results[0].orig_shape[:2]

        scale_x = cam_w / input_w
        scale_y = cam_h / input_h

        for box in boxes:
            # YOLO coords (resized)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            # *** GERÇEK piksel koordinatlarına ölçekle ***
            x1 *= scale_x
            x2 *= scale_x
            y1 *= scale_y
            y2 *= scale_y

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            det = Detection2D()
            det.header = msg.header
            det.header.frame_id = "camera_link"

            det.bbox.center.position.x = float(cx)
            det.bbox.center.position.y = float(cy)
            det.bbox.center.theta = 0.0

            det.bbox.size_x = float(x2 - x1)
            det.bbox.size_y = float(y2 - y1)

            hypo = ObjectHypothesisWithPose()
            hypo.hypothesis.class_id = str(int(box.cls.cpu().numpy()[0]))
            hypo.hypothesis.score = float(box.conf.cpu().numpy()[0])

            det.results.append(hypo)
            det_array.detections.append(det)

        self.pub_det.publish(det_array)


def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()