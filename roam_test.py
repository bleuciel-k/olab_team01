"""道があると仮定して、半径5m圏内を15秒間走行する動作確認用テストノード (ROS 2 Jazzy)

配信 (Publish):
  - /cmd_vel (Twist): 走行用の速度指令（終了時は必ず0を送信して停止）

購読 (Subscribe, 任意):
  - /odom (Odometry): 開始位置からの距離が半径5mを超えないよう監視・旋回補正
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class RoamTestNode(Node):
    """Vstone Light Rover の基本走行を15秒間テストするノード"""

    def __init__(self) -> None:
        super().__init__("roam_test_node")

        # パラメータ宣言（デフォルト: 半径5mの円弧を15秒間走行）
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("duration_sec", 15.0)
        self.declare_parameter("max_radius", 5.0)
        self.declare_parameter("linear_speed", 0.25)   # R = v / w = 0.25 / 0.05 = 5.0 m
        self.declare_parameter("angular_speed", 0.05)
        self.declare_parameter("control_hz", 20.0)

        gp = lambda name: self.get_parameter(name).value
        self.cmd_vel_topic = str(gp("cmd_vel_topic"))
        self.odom_topic = str(gp("odom_topic"))
        self.duration_sec = float(gp("duration_sec"))
        self.max_radius = float(gp("max_radius"))
        self.linear_speed = float(gp("linear_speed"))
        self.angular_speed = float(gp("angular_speed"))
        self.control_hz = float(gp("control_hz"))

        # パブリッシャー・サブスクライバー設定
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, qos_profile_sensor_data
        )

        self._start_xy: Optional[Tuple[float, float]] = None
        self._current_xy: Optional[Tuple[float, float]] = None
        self._start_time = self.get_clock().now()
        self._finished = False

        self.timer = self.create_timer(1.0 / self.control_hz, self._control_loop)
        self.get_logger().info(
            f"RoamTestNode started: duration={self.duration_sec:.1f}s, "
            f"max_radius={self.max_radius:.1f}m, v={self.linear_speed:.2f}m/s, "
            f"w={self.angular_speed:.2f}rad/s"
        )

    def _odom_callback(self, msg: Odometry) -> None:
        """オドメトリから開始地点に対する現在位置を記録する"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self._start_xy is None:
            self._start_xy = (x, y)
        self._current_xy = (x, y)

    def _distance_from_start(self) -> float:
        """開始地点からの直線距離 [m] を返す（/odom 未受信時は 0.0）"""
        if self._start_xy is None or self._current_xy is None:
            return 0.0
        dx = self._current_xy[0] - self._start_xy[0]
        dy = self._current_xy[1] - self._start_xy[1]
        return math.hypot(dx, dy)

    def _control_loop(self) -> None:
        """指定時間（15秒）まで走行し、経過後は停止してノードを終了する"""
        if self._finished:
            self.stop_robot()
            return

        elapsed = (self.get_clock().now() - self._start_time).nanoseconds * 1e-9
        if elapsed >= self.duration_sec:
            self._finished = True
            self.stop_robot()
            self.get_logger().info(
                f"Test finished ({elapsed:.1f}s). Robot stopped."
            )
            return

        cmd = Twist()
        dist = self._distance_from_start()

        # オドメトリ上で半径5mの上限に近づいた場合は、前進を緩めて内側へ旋回する
        if dist >= self.max_radius * 0.90:
            cmd.linear.x = self.linear_speed * 0.4
            cmd.angular.z = max(0.35, abs(self.angular_speed))
        else:
            cmd.linear.x = self.linear_speed
            cmd.angular.z = self.angular_speed

        self.cmd_pub.publish(cmd)

    def stop_robot(self) -> None:
        """停止指令（速度0）を送信する"""
        self.cmd_pub.publish(Twist())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RoamTestNode()
    try:
        while rclpy.ok() and not node._finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
