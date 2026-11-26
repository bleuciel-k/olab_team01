"""ROS 2 LaserScan から前・左・右の壁までの距離を測る。"""

import math
from statistics import median

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


def measure_walls(scan, directions, half_width):
    samples = {name: [] for name in directions}
    for index, distance in enumerate(scan.ranges):
        if not math.isfinite(distance) or not scan.range_min <= distance <= scan.range_max:
            continue
        if distance <= 0.0:
            continue

        angle = scan.angle_min + index * scan.angle_increment
        for name, center in directions.items():
            offset = math.remainder(angle - center, math.tau)
            if abs(offset) <= half_width:
                samples[name].append(distance * math.cos(offset))

    return {
        name: float(median(values)) if values else math.nan
        for name, values in samples.items()
    }


class WallDistances(Node):
    def __init__(self):
        super().__init__('wall_distances')
        self.topic = self.declare_parameter('scan_topic', '/scan').value
        front = math.radians(self.declare_parameter('front_angle_deg', 0.0).value)
        width = self.declare_parameter('sector_half_width_deg', 10.0).value
        if not 0.0 < width < 45.0:
            raise ValueError('sector_half_width_deg は 0 より大きく 45 未満にしてください')
        self.half_width = math.radians(width)
        self.directions = {
            'front': front,
            'left': front + math.pi / 2,
            'right': front - math.pi / 2,
        }
        self.last_scan = None
        self.distance_publishers = {
            name: self.create_publisher(Float32, f'/dist_{name}', 10)
            for name in self.directions
        }
        self.subscription = self.create_subscription(
            LaserScan, self.topic, self.on_scan, qos_profile_sensor_data
        )
        self.watchdog = self.create_timer(2.0, self.check_input)
        self.get_logger().info(f'{self.topic} → /dist_front, /dist_left, /dist_right')

    def check_input(self):
        now = self.get_clock().now()
        if self.last_scan is None or (now - self.last_scan).nanoseconds > 2_000_000_000:
            self.get_logger().warning(f'{self.topic} のスキャンが届いていません')

    def on_scan(self, scan):
        self.last_scan = self.get_clock().now()
        distances = measure_walls(scan, self.directions, self.half_width)
        for name, distance in distances.items():
            self.distance_publishers[name].publish(Float32(data=distance))

        summary = ' / '.join(
            f'{name}: {value * 100:.1f} cm' if math.isfinite(value) else f'{name}: 測定なし'
            for name, value in distances.items()
        )
        self.get_logger().info(f'スキャン受信 / {summary}', throttle_duration_sec=1.0)


def main():
    rclpy.init()
    node = WallDistances()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
