"""幅30cm通路の迷路用 Vstone Light Rover 移動制御ノード (ROS 2 Jazzy)

購読 (Subscribe):
  - レーダー/LiDARノードからの壁距離（個別Float32トピック または Vector3）
  - カメラノードからの標識方向 ("LEFT", "RIGHT", "DIAMOND"/"STOP", "NONE")
  - /odom (Odometry): 90度旋回の閉ループ制御用（未受信時は時間ベースで代替）

配信 (Publish):
  - /cmd_vel (Twist): 速度指令
  - /motion_controller/state (String): 現在の状態遷移 (FSM)
"""

from __future__ import annotations

from collections import deque
from enum import Enum
import math
from typing import Deque, Optional

from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32, String


class SignCommand(Enum):
    """カメラノードから受信する正規化済みの方向・停止コマンド"""

    NONE = "NONE"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    GOAL = "GOAL"


class MotionState(Enum):
    """移動制御ノードの状態遷移 (FSM)"""

    CRUISE_STRAIGHT = "CRUISE_STRAIGHT"
    WAIT_SIGN_AT_WALL = "WAIT_SIGN_AT_WALL"
    APPROACH_GOAL = "APPROACH_GOAL"
    PRE_TURN_PAUSE = "PRE_TURN_PAUSE"
    TURNING = "TURNING"
    POST_TURN_SETTLE = "POST_TURN_SETTLE"
    GOAL_REACHED = "GOAL_REACHED"
    ERROR_STOP = "ERROR_STOP"


def normalize_angle(angle_rad: float) -> float:
    """角度を [-pi, pi] の範囲に正規化する"""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """geometry_msgs のクォータニオンから平面ヨー角 (rad) を計算する"""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def parse_sign_command(raw: str) -> SignCommand:
    """カメラノードからの文字列ラベルを SignCommand に変換する"""
    token = raw.strip().upper()
    if token in ("LEFT", "L", "TURN_LEFT"):
        return SignCommand.LEFT
    if token in ("RIGHT", "R", "TURN_RIGHT"):
        return SignCommand.RIGHT
    if token in ("DIAMOND", "GOAL", "STOP", "D", "END"):
        return SignCommand.GOAL
    return SignCommand.NONE


class MotionControllerNode(Node):
    """Light Rover の速度制御、通路中央維持、旋回、ゴール停止を行うノード"""

    def __init__(self) -> None:
        super().__init__("motion_controller")
        self._declare_parameters()
        self._load_parameters()

        # パブリッシャー設定
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.state_pub = self.create_publisher(String, "/motion_controller/state", 10)

        # 距離サブスクライバー設定（レーダーノードの出力形式に合わせて切り替え）
        if self.distance_input_mode == "vector3":
            self.dist_vec_sub = self.create_subscription(
                Vector3,
                self.wall_distances_topic,
                self._wall_distances_callback,
                qos_profile_sensor_data,
            )
        else:
            self.dist_front_sub = self.create_subscription(
                Float32,
                self.dist_front_topic,
                self._dist_front_callback,
                qos_profile_sensor_data,
            )
            self.dist_left_sub = self.create_subscription(
                Float32,
                self.dist_left_topic,
                self._dist_left_callback,
                qos_profile_sensor_data,
            )
            self.dist_right_sub = self.create_subscription(
                Float32,
                self.dist_right_topic,
                self._dist_right_callback,
                qos_profile_sensor_data,
            )

        # 方向サブスクライバー設定（カメラノードから受信）
        self.sign_sub = self.create_subscription(
            String, self.sign_topic, self._sign_callback, 10
        )

        # オドメトリサブスクライバー設定（正確な90度旋回のため）
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, qos_profile_sensor_data
        )

        # センサー状態の初期化
        self._raw_front: Optional[float] = None
        self._raw_left: Optional[float] = None
        self._raw_right: Optional[float] = None
        self._last_front_time = None
        self._last_left_time = None
        self._last_right_time = None

        self._current_yaw: Optional[float] = None
        self._sign_history: Deque[SignCommand] = deque(maxlen=self.sign_history_len)
        self._pending_sign: SignCommand = SignCommand.NONE

        # FSM 状態の初期化
        self.state = MotionState.CRUISE_STRAIGHT
        self._state_enter_time = self.get_clock().now()
        self._turn_direction: SignCommand = SignCommand.NONE
        self._turn_target_yaw: Optional[float] = None
        self._turn_duration_sec: float = 0.0

        self.control_timer = self.create_timer(1.0 / self.control_hz, self._control_loop)
        self.get_logger().info(
            f"MotionController started (corridor={self.corridor_width:.2f}m, "
            f"dist_mode={self.distance_input_mode}, sign_topic={self.sign_topic}, "
            f"cmd_vel={self.cmd_vel_topic})"
        )

    def _declare_parameters(self) -> None:
        # トピック名および入力モード設定
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("sign_topic", "/sign_direction")
        self.declare_parameter("distance_input_mode", "separate")  # "separate" または "vector3"
        self.declare_parameter("dist_front_topic", "/dist_front")
        self.declare_parameter("dist_left_topic", "/dist_left")
        self.declare_parameter("dist_right_topic", "/dist_right")
        self.declare_parameter("wall_distances_topic", "/wall_distances")
        self.declare_parameter("distance_unit_scale", 1.0)  # メートルなら 1.0、cm なら 0.01

        # ロボットおよび30cm通路の寸法 (Vstone Light Rover: W176 x D194 mm)
        self.declare_parameter("corridor_width", 0.30)
        self.declare_parameter("robot_width", 0.176)
        self.declare_parameter("lidar_to_front_offset", 0.097)  # バンパー基準の距離なら 0.0
        self.declare_parameter("lidar_to_side_offset", 0.088)   # 車体側面基準の距離なら 0.0

        # 距離しきい値（ロボット前端・側面からの距離 [m]）
        self.declare_parameter("slowdown_start_distance", 0.45)
        self.declare_parameter("turn_trigger_distance", 0.06)   # 前端6cm（中心約15.7cm = 30cm角の中央）
        self.declare_parameter("goal_stop_distance", 0.10)      # ひし形標識の壁の手前10cmで停止
        self.declare_parameter("emergency_stop_distance", 0.04)
        self.declare_parameter("corridor_wall_max_dist", 0.20)  # 通路の壁とみなす側面からの最大距離
        self.declare_parameter("side_center_deadband", 0.008)   # 振動防止用の不感帯 (8mm)

        # 30cm通路向けの低速・安定走行パラメータ
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("cruise_linear_speed", 0.10)
        self.declare_parameter("approach_min_speed", 0.045)
        self.declare_parameter("k_wall_center", 1.8)
        self.declare_parameter("max_center_angular", 0.35)

        # 低速コーナリング（旋回）パラメータ
        self.declare_parameter("turn_angle_deg", 90.0)
        self.declare_parameter("k_yaw", 1.4)
        self.declare_parameter("min_turn_speed", 0.28)
        self.declare_parameter("max_turn_speed", 0.48)
        self.declare_parameter("fallback_turn_speed", 0.40)
        self.declare_parameter("yaw_tolerance_deg", 2.0)
        self.declare_parameter("turn_timeout_sec", 8.0)

        # タイミング・標識確定・安全設定
        self.declare_parameter("sign_history_len", 3)
        self.declare_parameter("sign_confirm_count", 2)
        self.declare_parameter("pre_turn_pause_sec", 0.15)
        self.declare_parameter("post_turn_settle_sec", 0.25)
        self.declare_parameter("wait_sign_timeout_sec", 1.5)
        self.declare_parameter("sensor_timeout_sec", 1.0)
        self.declare_parameter("require_front_distance", False)
        self.declare_parameter("treat_nan_front_as_clear", True)
        self.declare_parameter("lidar_fallback_on_unknown_wall", False)

    def _load_parameters(self) -> None:
        gp = lambda name: self.get_parameter(name).value

        self.cmd_vel_topic = str(gp("cmd_vel_topic"))
        self.odom_topic = str(gp("odom_topic"))
        self.sign_topic = str(gp("sign_topic"))
        self.distance_input_mode = str(gp("distance_input_mode")).lower()
        self.dist_front_topic = str(gp("dist_front_topic"))
        self.dist_left_topic = str(gp("dist_left_topic"))
        self.dist_right_topic = str(gp("dist_right_topic"))
        self.wall_distances_topic = str(gp("wall_distances_topic"))
        self.distance_unit_scale = float(gp("distance_unit_scale"))

        self.corridor_width = float(gp("corridor_width"))
        self.robot_width = float(gp("robot_width"))
        self.lidar_to_front_offset = float(gp("lidar_to_front_offset"))
        self.lidar_to_side_offset = float(gp("lidar_to_side_offset"))
        self.target_side_clearance = max(0.02, (self.corridor_width - self.robot_width) * 0.5)

        self.slowdown_start_distance = float(gp("slowdown_start_distance"))
        self.turn_trigger_distance = float(gp("turn_trigger_distance"))
        self.goal_stop_distance = float(gp("goal_stop_distance"))
        self.emergency_stop_distance = float(gp("emergency_stop_distance"))
        self.corridor_wall_max_dist = float(gp("corridor_wall_max_dist"))
        self.side_center_deadband = float(gp("side_center_deadband"))

        self.control_hz = float(gp("control_hz"))
        self.cruise_linear_speed = float(gp("cruise_linear_speed"))
        self.approach_min_speed = float(gp("approach_min_speed"))
        self.k_wall_center = float(gp("k_wall_center"))
        self.max_center_angular = float(gp("max_center_angular"))

        self.turn_angle_rad = math.radians(float(gp("turn_angle_deg")))
        self.k_yaw = float(gp("k_yaw"))
        self.min_turn_speed = float(gp("min_turn_speed"))
        self.max_turn_speed = float(gp("max_turn_speed"))
        self.fallback_turn_speed = float(gp("fallback_turn_speed"))
        self.yaw_tolerance_rad = math.radians(float(gp("yaw_tolerance_deg")))
        self.turn_timeout_sec = float(gp("turn_timeout_sec"))

        self.sign_history_len = int(gp("sign_history_len"))
        self.sign_confirm_count = int(gp("sign_confirm_count"))
        self.pre_turn_pause_sec = float(gp("pre_turn_pause_sec"))
        self.post_turn_settle_sec = float(gp("post_turn_settle_sec"))
        self.wait_sign_timeout_sec = float(gp("wait_sign_timeout_sec"))
        self.sensor_timeout_sec = float(gp("sensor_timeout_sec"))
        self.require_front_distance = bool(gp("require_front_distance"))
        self.treat_nan_front_as_clear = bool(gp("treat_nan_front_as_clear"))
        self.lidar_fallback_on_unknown_wall = bool(gp("lidar_fallback_on_unknown_wall"))

    # ------------------------------------------------------------------
    # サブスクライバー コールバック
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_distance(val: float, scale: float) -> Optional[float]:
        if not math.isfinite(val) or val <= 0.0:
            return None
        return val * scale

    def _dist_front_callback(self, msg: Float32) -> None:
        self._raw_front = self._sanitize_distance(msg.data, self.distance_unit_scale)
        self._last_front_time = self.get_clock().now()

    def _dist_left_callback(self, msg: Float32) -> None:
        self._raw_left = self._sanitize_distance(msg.data, self.distance_unit_scale)
        self._last_left_time = self.get_clock().now()

    def _dist_right_callback(self, msg: Float32) -> None:
        self._raw_right = self._sanitize_distance(msg.data, self.distance_unit_scale)
        self._last_right_time = self.get_clock().now()

    def _wall_distances_callback(self, msg: Vector3) -> None:
        now = self.get_clock().now()
        self._raw_front = self._sanitize_distance(msg.x, self.distance_unit_scale)
        self._raw_left = self._sanitize_distance(msg.y, self.distance_unit_scale)
        self._raw_right = self._sanitize_distance(msg.z, self.distance_unit_scale)
        self._last_front_time = now
        self._last_left_time = now
        self._last_right_time = now

    def _sign_callback(self, msg: String) -> None:
        if self.state in (
            MotionState.PRE_TURN_PAUSE,
            MotionState.TURNING,
            MotionState.POST_TURN_SETTLE,
            MotionState.GOAL_REACHED,
        ):
            return

        cmd = parse_sign_command(msg.data)
        self._sign_history.append(cmd)

        for candidate in (SignCommand.GOAL, SignCommand.LEFT, SignCommand.RIGHT):
            if sum(1 for s in self._sign_history if s == candidate) >= self.sign_confirm_count:
                if candidate != self._pending_sign:
                    self._pending_sign = candidate
                    self.get_logger().info(f"Confirmed camera command: {candidate.value}")
                break

    def _odom_callback(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        self._current_yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)

    # ------------------------------------------------------------------
    # ヘルパー関数
    # ------------------------------------------------------------------
    def _transition(self, new_state: MotionState) -> None:
        if new_state != self.state:
            self.get_logger().info(f"State: {self.state.value} -> {new_state.value}")
            self.state = new_state
            self._state_enter_time = self.get_clock().now()

    def _elapsed_in_state(self) -> float:
        return (self.get_clock().now() - self._state_enter_time).nanoseconds * 1e-9

    def _clear_sign_memory(self) -> None:
        self._sign_history.clear()
        self._pending_sign = SignCommand.NONE

    def _prepare_turn(self, direction: SignCommand) -> None:
        self._turn_direction = direction
        self._transition(MotionState.PRE_TURN_PAUSE)

    def _begin_turn(self) -> None:
        delta = self.turn_angle_rad if self._turn_direction == SignCommand.LEFT else -self.turn_angle_rad
        if self._current_yaw is not None:
            self._turn_target_yaw = normalize_angle(self._current_yaw + delta)
            self._turn_duration_sec = 0.0
        else:
            self._turn_target_yaw = None
            self._turn_duration_sec = abs(delta) / max(0.1, self.fallback_turn_speed)
            self.get_logger().warning("Odometry unavailable; using timed fallback turn.")
        self._transition(MotionState.TURNING)

    def _compute_forward_speed(self, d_front: float, stop_dist: float) -> float:
        """正面壁との距離に応じて前進速度を滑らかに減速させる"""
        if d_front >= self.slowdown_start_distance:
            return self.cruise_linear_speed
        span = max(0.01, self.slowdown_start_distance - stop_dist)
        ratio = max(0.0, min(1.0, (d_front - stop_dist) / span))
        return self.approach_min_speed + ratio * (
            self.cruise_linear_speed - self.approach_min_speed
        )

    def _compute_centering_angular(
        self, d_left: Optional[float], d_right: Optional[float]
    ) -> float:
        """左右の壁距離を用いて30cm通路の中央を維持する角速度を計算する"""
        left_valid = d_left is not None and d_left < self.corridor_wall_max_dist
        right_valid = d_right is not None and d_right < self.corridor_wall_max_dist

        if left_valid and right_valid:
            # d_left > d_right なら右寄りのため左旋回 (+z) で中央へ戻す
            err = d_left - d_right
        elif left_valid:
            # 右側が開いている場合（分岐点など）：左壁との基準間隔を維持
            err = d_left - self.target_side_clearance
        elif right_valid:
            # 左側が開いている場合：右壁との基準間隔を維持
            err = self.target_side_clearance - d_right
        else:
            return 0.0

        if abs(err) < self.side_center_deadband:
            return 0.0

        angular = self.k_wall_center * err
        return max(-self.max_center_angular, min(self.max_center_angular, angular))

    # ------------------------------------------------------------------
    # メイン制御ループ
    # ------------------------------------------------------------------
    def _control_loop(self) -> None:
        state_msg = String()
        state_msg.data = self.state.value
        self.state_pub.publish(state_msg)

        if self.state in (MotionState.GOAL_REACHED, MotionState.ERROR_STOP):
            self.stop_robot()
            return

        now = self.get_clock().now()
        front_msg_fresh = (
            self._last_front_time is not None
            and (now - self._last_front_time).nanoseconds * 1e-9 <= self.sensor_timeout_sec
        )
        if front_msg_fresh and self._raw_front is not None:
            d_front = max(0.0, self._raw_front - self.lidar_to_front_offset)
        elif (front_msg_fresh and self.treat_nan_front_as_clear) or (
            not self.require_front_distance
        ):
            # 壁がない（NaN）または壁なしテストモード時は前方クリア（10m）として走行
            d_front = 10.0
        else:
            self.stop_robot()
            self.get_logger().warning(
                "Waiting for valid front distance data...", throttle_duration_sec=2.0
            )
            return
        d_left = (
            max(0.0, self._raw_left - self.lidar_to_side_offset)
            if self._raw_left is not None
            and self._last_left_time is not None
            and (now - self._last_left_time).nanoseconds * 1e-9 <= self.sensor_timeout_sec
            else None
        )
        d_right = (
            max(0.0, self._raw_right - self.lidar_to_side_offset)
            if self._raw_right is not None
            and self._last_right_time is not None
            and (now - self._last_right_time).nanoseconds * 1e-9 <= self.sensor_timeout_sec
            else None
        )

        cmd = Twist()

        if self.state == MotionState.CRUISE_STRAIGHT:
            if self._pending_sign == SignCommand.GOAL:
                self._transition(MotionState.APPROACH_GOAL)
            elif d_front <= self.turn_trigger_distance:
                if self._pending_sign in (SignCommand.LEFT, SignCommand.RIGHT):
                    self._prepare_turn(self._pending_sign)
                else:
                    self._transition(MotionState.WAIT_SIGN_AT_WALL)
            elif d_front <= self.emergency_stop_distance:
                self.get_logger().warning(
                    f"Emergency close wall ({d_front:.3f}m); waiting for command."
                )
                self._transition(MotionState.WAIT_SIGN_AT_WALL)
            else:
                cmd.linear.x = self._compute_forward_speed(
                    d_front, self.turn_trigger_distance
                )
                cmd.angular.z = self._compute_centering_angular(d_left, d_right)

        elif self.state == MotionState.WAIT_SIGN_AT_WALL:
            if self._pending_sign == SignCommand.GOAL:
                self._transition(MotionState.APPROACH_GOAL)
            elif self._pending_sign in (SignCommand.LEFT, SignCommand.RIGHT):
                self._prepare_turn(self._pending_sign)
            elif self._elapsed_in_state() >= self.wait_sign_timeout_sec:
                if self.lidar_fallback_on_unknown_wall and (
                    d_left is not None or d_right is not None
                ):
                    left_val = d_left if d_left is not None else 0.0
                    right_val = d_right if d_right is not None else 0.0
                    fallback = SignCommand.LEFT if left_val >= right_val else SignCommand.RIGHT
                    self.get_logger().warning(
                        f"No camera command at wall; fallback turn {fallback.value}."
                    )
                    self._prepare_turn(fallback)
                else:
                    self.get_logger().error("No camera command received at front wall; stopping.")
                    self._transition(MotionState.ERROR_STOP)

        elif self.state == MotionState.APPROACH_GOAL:
            if d_front <= self.goal_stop_distance:
                self.stop_robot()
                self.get_logger().info(
                    f"GOAL REACHED! Stopped {d_front * 100.0:.1f} cm before front wall."
                )
                self._transition(MotionState.GOAL_REACHED)
                return
            cmd.linear.x = self._compute_forward_speed(d_front, self.goal_stop_distance)
            cmd.angular.z = self._compute_centering_angular(d_left, d_right)

        elif self.state == MotionState.PRE_TURN_PAUSE:
            if self._elapsed_in_state() >= self.pre_turn_pause_sec:
                self._begin_turn()

        elif self.state == MotionState.TURNING:
            elapsed = self._elapsed_in_state()
            if elapsed >= self.turn_timeout_sec:
                self.get_logger().warning("Turn timeout reached; settling.")
                self._clear_sign_memory()
                self._transition(MotionState.POST_TURN_SETTLE)
            elif self._turn_target_yaw is not None and self._current_yaw is not None:
                err = normalize_angle(self._turn_target_yaw - self._current_yaw)
                if abs(err) <= self.yaw_tolerance_rad:
                    self.stop_robot()
                    self._clear_sign_memory()
                    self._transition(MotionState.POST_TURN_SETTLE)
                    return
                mag = max(
                    self.min_turn_speed, min(self.max_turn_speed, self.k_yaw * abs(err))
                )
                cmd.angular.z = mag if err > 0.0 else -mag
            else:
                if elapsed >= self._turn_duration_sec:
                    self.stop_robot()
                    self._clear_sign_memory()
                    self._transition(MotionState.POST_TURN_SETTLE)
                    return
                cmd.angular.z = (
                    self.fallback_turn_speed
                    if self._turn_direction == SignCommand.LEFT
                    else -self.fallback_turn_speed
                )

        elif self.state == MotionState.POST_TURN_SETTLE:
            if self._elapsed_in_state() >= self.post_turn_settle_sec:
                self._clear_sign_memory()
                self._transition(MotionState.CRUISE_STRAIGHT)

        self.cmd_pub.publish(cmd)

    def stop_robot(self) -> None:
        """停止指令（速度0）を送信する"""
        self.cmd_pub.publish(Twist())


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MotionControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
