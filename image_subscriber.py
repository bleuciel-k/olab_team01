"""ROS 2 カメラ画像から迷路の標識を判別する。"""

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


def detect_sign(frame):
    """白地に黒い矢印またはひし形を探す。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if not 0.001 * height * width <= area <= 0.50 * height * width:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < 12 or x == 0 or y == 0 or x + w == width or y + h == height:
            continue

        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        points = [
            ((px - x) / (w - 1), (py - y) / (h - 1))
            for px, py in polygon.reshape(-1, 2)
        ]
        left = [py for px, py in points if px < 0.12]
        right = [py for px, py in points if px > 0.88]
        box = (x, y, w, h)

        if len(points) == 7 and not cv2.isContourConvex(polygon) and w >= h:
            if len(left) == 1 and len(right) == 2 and 0.25 <= left[0] <= 0.75:
                return 'LEFT', box
            if len(right) == 1 and len(left) == 2 and 0.25 <= right[0] <= 0.75:
                return 'RIGHT', box

        if len(points) == 4 and cv2.isContourConvex(polygon) and 0.65 <= w / h <= 1.55:
            top = min(points, key=lambda p: p[1])
            bottom = max(points, key=lambda p: p[1])
            if (
                len(left) == len(right) == 1
                and all(0.25 <= v <= 0.75 for v in (left[0], right[0], top[0], bottom[0]))
            ):
                return 'DIAMOND', box

    return 'NONE', None


class CameraSigns(Node):
    def __init__(self):
        super().__init__('camera_signs')
        self.topic = self.declare_parameter('image_topic', '/image_raw').value
        self.show_image = self.declare_parameter('show_image', False).value
        self.bridge = CvBridge()
        self.last_frame = None
        self.publisher = self.create_publisher(String, '/sign_direction', 10)
        self.subscription = self.create_subscription(
            Image, self.topic, self.on_image, qos_profile_sensor_data
        )
        self.watchdog = self.create_timer(2.0, self.check_input)
        self.get_logger().info(f'{self.topic} → /sign_direction')

    def check_input(self):
        now = self.get_clock().now()
        if self.last_frame is None or (now - self.last_frame).nanoseconds > 2_000_000_000:
            self.get_logger().warning(f'{self.topic} の画像が届いていません')

    def on_image(self, msg):
        self.last_frame = self.get_clock().now()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except (CvBridgeError, cv2.error) as error:
            self.get_logger().error(f'画像変換エラー ({msg.encoding}): {error}',
                                    throttle_duration_sec=2.0)
            self.publisher.publish(String(data='NONE'))
            return

        label, box = detect_sign(frame)
        self.publisher.publish(String(data=label))
        self.get_logger().info(
            f'画像受信 {msg.width}x{msg.height} / 判定 {label}',
            throttle_duration_sec=1.0
        )
        if self.show_image:
            display = frame.copy()
            if box:
                x, y, w, h = box
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.imshow('Camera Signs', display)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = CameraSigns()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_image:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
