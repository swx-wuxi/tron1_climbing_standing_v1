import os
import sys
import copy
import numpy as np
import yaml
import time
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from functools import partial
import limxsdk
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes

class WheelfootController:
    def __init__(self, model_dir, robot, robot_type, rl_type, start_controller):
        # Initialize robot and type information
        self.robot = robot
        self.robot_type = robot_type
        self.rl_type = rl_type
        print("======")
        # Load configuration and model file paths based on robot type
        self.config_file = f'{model_dir}/{self.robot_type}/params.yaml'
        self.model_policy = f'{model_dir}/{self.robot_type}/policy/{self.rl_type}/policy.onnx'
        self.model_encoder = f'{model_dir}/{self.robot_type}/policy/{self.rl_type}/encoder.onnx'

        # Load configuration settings from the YAML file
        self.load_config(self.config_file)
        
        # Load the ONNX model
        self.initialize_onnx_models()
        self.validate_deployment_config()

        # Prepare robot command structure with default values for mode, q, dq, tau, Kp, Kd
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.q = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for x in range(0, self.joint_num)]
        self.robot_cmd.Kp = [self.control_cfg['stiffness'] for x in range(0, self.joint_num)]
        self.robot_cmd.Kd = [self.control_cfg['damping'] for x in range(0, self.joint_num)]

        # Prepare robot state structure
        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0. for x in range(0, self.joint_num)]
        self.robot_state.q = [0. for x in range(0, self.joint_num)]
        self.robot_state.dq = [0. for x in range(0, self.joint_num)]
        self.robot_state_tmp = copy.deepcopy(self.robot_state)

        # Initialize IMU (Inertial Measurement Unit) data structure
        self.imu_data = datatypes.ImuData()
        self.imu_data.quat[0] = 0
        self.imu_data.quat[1] = 0
        self.imu_data.quat[2] = 0
        self.imu_data.quat[3] = 1
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        # Set up a callback to receive updated robot state data
        self.robot_state_callback_partial = partial(self.robot_state_callback)
        self.robot.subscribeRobotState(self.robot_state_callback_partial)

        # Set up a callback to receive updated IMU data
        self.imu_data_callback_partial = partial(self.imu_data_callback)
        self.robot.subscribeImuData(self.imu_data_callback_partial)

        # Set up a callback to receive updated SensorJoy
        self.sensor_joy_callback_partial = partial(self.sensor_joy_callback)
        self.robot.subscribeSensorJoy(self.sensor_joy_callback_partial)

        # Set up a callback to receive diagnostic data
        self.robot_diagnostic_callback_partial = partial(self.robot_diagnostic_callback)
        self.robot.subscribeDiagnosticValue(self.robot_diagnostic_callback_partial)

        # Initialize the calibration state to -1, indicating no calibration has occurred.
        self.calibration_state = -1

        # Flag to start the controller
        self.start_controller = start_controller

        # Gait index
        self.gait_index = 0

        # Flag indicating first received observation
        self.is_first_rec_obs = True

        # >>> S2S: minimal fixed-kneel -> stand -> RL handoff state.
        # This intentionally does not change the 28-D observation or WALK policy path.
        self.has_robot_state = False
        self.has_imu_data = False
        self.phase_timer = 0.0
        self.stand_start_q = None
        self.stand_mid_q = None
        self.stand_start_pitch = 0.0
        self.stand_handoff_pitch = 0.0
        self.balance_wheel_velocity = 0.0
        self.rl_blend_start_q = None
        self.rl_blend_started = False

        # Fixed-kneel stand-up FSM state (stand-up only; no kneel-down/RL logic copied).
        self.phase_start_angles = None
        self.stand_target_angles = None
        self.wheel_start_angles = None
        self.fsm_last_leg_target = None
        self.abort_hold_angles = None

    def initialize_onnx_models(self):
        # Configure ONNX Runtime session options to optimize CPU usage
        session_options = ort.SessionOptions()
        # Limit the number of threads used for parallel computation within individual operators
        session_options.intra_op_num_threads = 1
        # Limit the number of threads used for parallel execution of different operators
        session_options.inter_op_num_threads = 1
        # Enable all possible graph optimizations to improve inference performance
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Disable CPU memory arena to reduce memory fragmentation
        session_options.enable_cpu_mem_arena = False
        # Disable memory pattern optimization to have more control over memory allocation
        session_options.enable_mem_pattern = False

        # Define execution providers to use CPU only, ensuring no GPU inference
        cpu_providers = ['CPUExecutionProvider']
        
        # Load the ONNX model and set up input and output names
        self.policy_session = ort.InferenceSession(self.model_policy, sess_options=session_options, providers=cpu_providers)
        self.policy_input_names = [self.policy_session.get_inputs()[i].name for i in range(self.policy_session.get_inputs().__len__())]
        self.policy_output_names = [self.policy_session.get_outputs()[i].name for i in range(self.policy_session.get_outputs().__len__())]
        self.policy_input_shapes = [self.policy_session.get_inputs()[i].shape for i in range(self.policy_session.get_inputs().__len__())]
        self.policy_output_shapes = [self.policy_session.get_outputs()[i].shape for i in range(self.policy_session.get_outputs().__len__())]

        self.encoder_session = ort.InferenceSession(self.model_encoder, sess_options=session_options, providers=cpu_providers)
        self.encoder_input_names = [self.encoder_session.get_inputs()[i].name for i in range(self.encoder_session.get_inputs().__len__())]
        self.encoder_output_names = [self.encoder_session.get_outputs()[i].name for i in range(self.encoder_session.get_outputs().__len__())]
        self.encoder_input_shapes = [self.encoder_session.get_inputs()[i].shape for i in range(self.encoder_session.get_inputs().__len__())]
        self.encoder_output_shapes = [self.encoder_session.get_outputs()[i].shape for i in range(self.encoder_session.get_outputs().__len__())]

    def validate_deployment_config(self):
        """Fail before commanding hardware if an export differs from this training interface."""
        expected_policy_input = (
            self.encoder_output_size + self.observations_size + self.commands_size
        )
        checks = (
            ("encoder input", self.encoder_input_shapes[0][-1], self.encoder_input_size),
            ("encoder output", self.encoder_output_shapes[0][-1], self.encoder_output_size),
            ("policy input", self.policy_input_shapes[0][-1], expected_policy_input),
            ("policy output", self.policy_output_shapes[0][-1], self.actions_size),
        )
        for name, actual, expected in checks:
            if isinstance(actual, int) and actual != expected:
                raise ValueError(
                    f"{name} dimension mismatch: ONNX has {actual}, deployment expects "
                    f"{expected}. Export both ONNX files from the current training checkpoint."
                )

    # Load the configuration from a YAML file
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Assign configuration parameters to controller variables
        self.joint_names = config['PointfootCfg']['joint_names']
        self.init_state = config['PointfootCfg']['init_state']['default_joint_angle']
        stand_cfg = config['PointfootCfg'].get('stand_mode', {})
        self.stand_duration = float(stand_cfg.get('stand_duration', 1.0))  # legacy only

        # >>> S2S: independent defaults so the existing stand_duration: 1.0 does
        # not force a one-second squat-to-stand. All keys below are optional.
        self.s2s_kneel_hold_duration = float(stand_cfg.get('kneel_hold_duration', 0.35))
        self.s2s_duration = float(stand_cfg.get('squat_to_stand_duration', 3.0))
        self.s2s_mid_fraction = float(stand_cfg.get('mid_phase_fraction', 0.45))
        self.s2s_handoff_pitch_ratio = float(stand_cfg.get('handoff_pitch_ratio', 0.15))
        self.s2s_handoff_pitch_max_deg = float(stand_cfg.get('handoff_pitch_max_deg', 8.0))
        self.s2s_balance_kp = float(stand_cfg.get('balance_kp', 1.2))
        self.s2s_balance_kd = float(stand_cfg.get('balance_kd', 0.25))
        self.s2s_balance_direction = float(stand_cfg.get('balance_direction', -1.0))
        self.s2s_balance_max_speed = float(stand_cfg.get('max_balance_wheel_speed', 1.0))
        self.s2s_balance_accel = float(stand_cfg.get('balance_wheel_accel_limit', 4.0))
        self.s2s_rl_blend_duration = float(stand_cfg.get('rl_blend_duration', 1.0))

        # ---- Fixed-kneel stand-up FSM parameters ----
        # These are the small subset required by the colleague's stand-up path.
        fsm_cfg = stand_cfg.get('fsm', {})
        self.balance_target_pitch_deg = float(stand_cfg.get('target_pitch_deg', 0.0))
        self.support_pitch_deg = float(stand_cfg.get('support_pitch_deg', 20.0))
        self.balance_kp = float(stand_cfg.get('balance_kp', 4.0))
        self.balance_kd = float(stand_cfg.get('balance_kd', 0.6))
        self.balance_direction = float(stand_cfg.get('balance_direction', -1.0))
        self.balance_left_wheel_sign = float(stand_cfg.get('left_wheel_sign', 1.0))
        self.balance_right_wheel_sign = float(stand_cfg.get('right_wheel_sign', 1.0))
        self.balance_max_wheel_speed = float(stand_cfg.get('max_balance_wheel_speed', 3.0))
        self.balance_wheel_accel_limit = float(stand_cfg.get('balance_wheel_accel_limit', 10.0))

        self.prepare_duration = float(fsm_cfg.get('prepare_duration', 1.0))
        self.shift_duration = float(fsm_cfg.get('shift_duration', 2.0))
        self.lift_duration = float(fsm_cfg.get('lift_duration', self.s2s_duration))
        self.fsm_target_fraction = float(fsm_cfg.get(
            'support_fraction', fsm_cfg.get('target_fraction', 1.0)
        ))
        self.knee_delay_fraction = float(fsm_cfg.get('knee_delay_fraction', 0.18))
        self.knee_release_duration = float(fsm_cfg.get('knee_release_duration', 5.0))
        self.knee_release_timeout = float(fsm_cfg.get('knee_release_timeout', 6.0))
        self.rl_handoff_knee_angle = float(fsm_cfg.get('rl_handoff_knee_angle', 1.20))
        self.knee_delay_fraction = float(fsm_cfg.get('knee_delay_fraction', 0.18))
        self.fsm_low_stiffness = float(fsm_cfg.get('low_stiffness', 18.0))
        self.fsm_low_damping = float(fsm_cfg.get('low_damping', 2.0))
        self.hip_pitch_kp = float(fsm_cfg.get('hip_pitch_kp', 0.18))
        self.hip_pitch_kd = float(fsm_cfg.get('hip_pitch_kd', 0.04))
        self.max_hip_compensation = float(fsm_cfg.get('max_hip_compensation', 0.18))
        self.roll_knee_kp = float(fsm_cfg.get('roll_knee_kp', 0.08))
        self.roll_knee_kd = float(fsm_cfg.get('roll_knee_kd', 0.02))
        self.roll_compensation_direction = float(fsm_cfg.get('roll_compensation_direction', 1.0))
        self.max_roll_compensation = float(fsm_cfg.get('max_roll_compensation', 0.10))
        self.wheel_position_kp = float(fsm_cfg.get('wheel_position_kp', 0.35))
        self.wheel_travel_limit = float(fsm_cfg.get('wheel_travel_limit', 4.0))
        self.fsm_max_joint_speed = float(fsm_cfg.get('max_joint_speed', 6.0))
        self.fsm_max_tracking_error = float(fsm_cfg.get('max_tracking_error', 0.75))
        self.abort_stiffness = float(fsm_cfg.get('abort_stiffness', 8.0))
        self.abort_damping = float(fsm_cfg.get('abort_damping', 3.0))

        self.control_cfg = config['PointfootCfg']['control']
        self.rl_cfg = config['PointfootCfg']['normalization']
        self.obs_scales = config['PointfootCfg']['normalization']['obs_scales']
        self.actions_size = config['PointfootCfg']['size']['actions_size']
        self.commands_size = config['PointfootCfg']['size']['commands_size']
        self.observations_size = config['PointfootCfg']['size']['observations_size']
        self.obs_history_length = config['PointfootCfg']['size']['obs_history_length']
        self.encoder_output_size = config['PointfootCfg']['size']['encoder_output_size']
        self.imu_orientation_offset = np.array(list(config['PointfootCfg']['imu_orientation_offset'].values()))
        self.user_cmd_cfg = config['PointfootCfg']['user_cmd_scales']
        self.loop_frequency = config['PointfootCfg']['loop_frequency']
        self.encoder_input_size = self.obs_history_length * self.observations_size

        # Initialize variables for actions, observations, and commands
        self.proprio_history_vector = np.zeros(self.obs_history_length * self.observations_size)
        self.encoder_out = np.zeros(self.encoder_output_size)
        self.actions = np.zeros(self.actions_size)
        self.observations = np.zeros(self.observations_size)
        self.last_actions = np.zeros(self.actions_size)
        self.commands = np.zeros(self.commands_size)  # command to the robot (e.g., velocity, rotation)
        self.scaled_commands = np.zeros(self.commands_size)
        self.base_lin_vel = np.zeros(3)  # base linear velocity
        self.base_position = np.zeros(3)  # robot base position
        self.loop_count = 0  # loop iteration count
        self.stand_percent = 0  # percentage of time the robot has spent in stand mode
        self.policy_session = None  # ONNX model session for policy inference
        self.joint_num = len(self.joint_names)  # number of joints

        self.joint_pos_idxs = config['PointfootCfg']['size']['jointpos_idxs']
        self.wheel_joint_damping = config['PointfootCfg']['control']['wheel_joint_damping']
        self.wheel_joint_torque_limit = config['PointfootCfg']['control']['wheel_joint_torque_limit']

        # Initialize joint angles based on the initial configuration
        self.init_joint_angles = np.zeros(len(self.joint_names))
        for i in range(len(self.joint_names)):
            self.init_joint_angles[i] = self.init_state[self.joint_names[i]]
        
        # Controller starts from the fixed kneeling pose.
        self.mode = "KNEEL_HOLD"
    
    # Main control loop
    def run(self):
        while not self.start_controller:
            time.sleep(0.1)

        # >>> S2S: never capture the zero-initialized placeholder RobotState.
        print("Waiting for the first RobotState and IMU packets...")
        deadline = time.monotonic() + 5.0
        while self.start_controller and not (self.has_robot_state and self.has_imu_data):
            if time.monotonic() >= deadline:
                raise TimeoutError("RobotState/IMU data not received within 5 seconds")
            time.sleep(0.01)
        if not self.start_controller:
            return

        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)
        self.stand_start_q = np.asarray(self.robot_state.q, dtype=float)[:self.joint_num].copy()
        if self.stand_start_q.size != self.joint_num or not np.all(np.isfinite(self.stand_start_q)):
            raise ValueError("Invalid initial joint state for squat-to-stand")

        self.stand_start_pitch, _ = self.get_pitch_state()
        max_pitch = np.radians(self.s2s_handoff_pitch_max_deg)
        self.stand_handoff_pitch = float(np.clip(
            self.s2s_handoff_pitch_ratio * self.stand_start_pitch,
            -max_pitch,
            max_pitch,
        ))
        self.stand_mid_q = self.build_stand_mid_pose()

        # FSM starts from the measured fixed kneel and targets the existing nominal stand pose.

        # self.stand_target_angles = self.init_joint_angles.copy()
        target_fraction = float(np.clip(self.fsm_target_fraction, 0.0, 1.0))
        self.stand_target_angles = (
            self.stand_start_q * (1.0 - target_fraction)
           + self.init_joint_angles * target_fraction
        )
        self.phase_start_angles = self.stand_start_q.copy()
        self.wheel_start_angles = self.stand_start_q.copy()
        self.fsm_last_leg_target = self.stand_start_q.copy()
        self.abort_hold_angles = None

        self.phase_timer = 0.0
        self.balance_wheel_velocity = 0.0
        self.commands.fill(0.0)
        self.actions.fill(0.0)
        self.last_actions.fill(0.0)
        self.is_first_rec_obs = True
        self.mode = "KNEEL_HOLD"
        self.loop_count = 0

        print(
            "Squat-to-stand initialized: "
            f"pitch={np.degrees(self.stand_start_pitch):.1f} deg -> "
            f"handoff={np.degrees(self.stand_handoff_pitch):.1f} deg"
        )

        rate = Rate(self.loop_frequency)
        while self.start_controller:
            self.update()
            rate.sleep()

        self.robot_cmd.q = [0. for _ in range(self.joint_num)]
        self.robot_cmd.dq = [0. for _ in range(self.joint_num)]
        self.robot_cmd.tau = [0. for _ in range(self.joint_num)]
        self.robot_cmd.Kp = [0. for _ in range(self.joint_num)]
        self.robot_cmd.Kd = [1.0 for _ in range(self.joint_num)]
        self.robot.publishRobotCmd(self.robot_cmd)
        time.sleep(1)

    @staticmethod
    def smoothstep(value):
        value = float(np.clip(value, 0.0, 1.0))
        return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5

    def get_pitch_state(self):
        """Return pitch and pitch rate using the same IMU convention as observations."""
        q = np.asarray(self.imu_data_tmp.quat, dtype=float)
        gyro = np.asarray(self.imu_data_tmp.gyro, dtype=float)
        if q.size != 4 or gyro.size < 3 or not np.all(np.isfinite(q)):
            return 0.0, 0.0
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-6:
            return 0.0, 0.0
        q = q / q_norm

        inverse_rot = R.from_quat(q).inv().as_matrix()
        gravity = inverse_rot @ np.array([0.0, 0.0, -1.0])
        offset_rot = R.from_euler('zyx', self.imu_orientation_offset).as_matrix()
        gravity = offset_rot @ gravity
        gyro = offset_rot @ gyro[:3]
        pitch = np.arctan2(gravity[0], -gravity[2])
        return float(pitch), float(gyro[1])

    def build_stand_mid_pose(self):
        """First move the hips/COM, then finish knee extension toward RL nominal."""
        q = self.stand_start_q.copy()
        # Joint order is [abadL, hipL, kneeL, wheelL, abadR, hipR, kneeR, wheelR].
        fractions = {0: 0.50, 1: 0.55, 2: 0.20, 4: 0.50, 5: 0.55, 6: 0.20}
        for index, fraction in fractions.items():
            q[index] = self.stand_start_q[index] + fraction * (
                self.init_joint_angles[index] - self.stand_start_q[index]
            )
        return q

    def get_s2s_wheel_velocity(self, target_pitch, fade=1.0):
        """Small rate-limited pitch correction used only during STAND_UP."""
        pitch, pitch_rate = self.get_pitch_state()
        target = self.s2s_balance_direction * (
            self.s2s_balance_kp * (pitch - target_pitch)
            + self.s2s_balance_kd * pitch_rate
        )
        target = float(np.clip(
            target,
            -self.s2s_balance_max_speed,
            self.s2s_balance_max_speed,
        )) * float(np.clip(fade, 0.0, 1.0))

        max_step = self.s2s_balance_accel / self.loop_frequency
        step = np.clip(
            target - self.balance_wheel_velocity,
            -max_step,
            max_step,
        )
        self.balance_wheel_velocity += float(step)
        return self.balance_wheel_velocity, pitch, pitch_rate

    def handle_kneel_hold(self):
        dt = 1.0 / self.loop_frequency
        self.commands.fill(0.0)
        # 固定蹲姿时就开始维持当前 pitch
        pitch_target = self.stand_start_pitch
        wheel_dq, pitch, pitch_rate = self.get_s2s_wheel_velocity(
            pitch_target,
            fade=1.0
        )
        for i in range(self.joint_num):
            if (i + 1) % 4 == 0:
                self.set_joint_command(
                    i,
                    0,
                    wheel_dq,
                    0,
                    0,
                    self.wheel_joint_damping
                )
            else:
                self.set_joint_command(
                    i,
                    self.stand_start_q[i],
                    0,
                    0,
                    self.control_cfg['stiffness'],
                    self.control_cfg['damping']
                )
        self.phase_timer += dt

        if self.phase_timer >= self.s2s_kneel_hold_duration:
            self.phase_timer = 0.0
            # 不要重新清零！
            # self.balance_wheel_velocity = 0.0
            self.mode = "PREPARE_SUPPORT"
            self.phase_start_angles = np.asarray(
                self.robot_state.q, dtype=float
            )[:self.joint_num].copy()
            self.wheel_start_angles = self.phase_start_angles.copy()
            self.fsm_last_leg_target = self.phase_start_angles.copy()
            print("KNEEL_HOLD -> PREPARE_SUPPORT")

    @staticmethod
    def quintic_blend(value):
        """Smooth 0->1 blend used by the colleague stand-up trajectory."""
        value = float(np.clip(value, 0.0, 1.0))
        return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5

    def get_full_orientation_state(self):
        """Return roll, pitch, roll-rate, pitch-rate and total tilt."""
        quat = np.asarray(self.imu_data_tmp.quat, dtype=float)
        gyro = np.asarray(self.imu_data_tmp.gyro, dtype=float)
        if (quat.size != 4 or gyro.size < 3 or not np.all(np.isfinite(quat))
                or not np.all(np.isfinite(gyro[:3])) or np.linalg.norm(quat) < 1e-6):
            return (float("inf"),) * 5

        inverse_rot = R.from_quat(quat).inv().as_matrix()
        offset_rot = R.from_euler('xyz', self.imu_orientation_offset).as_matrix()
        gravity = offset_rot @ (inverse_rot @ np.array([0.0, 0.0, -1.0]))
        norm = np.linalg.norm(gravity)
        if norm < 1e-6:
            return (float("inf"),) * 5
        gravity /= norm

        roll = float(np.arctan2(-gravity[1], -gravity[2]))
        pitch = float(np.arctan2(gravity[0], -gravity[2]))
        tilt = float(np.arccos(np.clip(-gravity[2], -1.0, 1.0)))
        corrected_gyro = offset_rot @ gyro[:3]
        return roll, pitch, float(corrected_gyro[0]), float(corrected_gyro[1]), tilt

    def get_wheel_indices(self):
        return [i for i in range(self.joint_num) if (i + 1) % 4 == 0]

    def get_leg_indices(self):
        return [i for i in range(self.joint_num) if (i + 1) % 4 != 0]

    def get_wheel_displacement(self):
        """Signed average wheel displacement relative to stand-up start."""
        if self.wheel_start_angles is None:
            return 0.0
        q = np.asarray(self.robot_state_tmp.q, dtype=float)[:self.joint_num]
        wheels = self.get_wheel_indices()
        if q.size != self.joint_num or len(wheels) < 2:
            return float("inf")
        left = (q[wheels[0]] - self.wheel_start_angles[wheels[0]]) * self.balance_left_wheel_sign
        right = (q[wheels[1]] - self.wheel_start_angles[wheels[1]]) * self.balance_right_wheel_sign
        return float(0.5 * (left + right))

    def get_balance_wheel_commands(self, target_pitch_rad):
        """Colleague-style rate-limited pitch controller for the two wheels."""
        _, pitch, _, pitch_rate, _ = self.get_full_orientation_state()
        if not np.all(np.isfinite([pitch, pitch_rate])):
            self.balance_wheel_velocity = 0.0
            return 0.0, 0.0, pitch, pitch_rate

        error = pitch - float(target_pitch_rad)
        target_velocity = self.balance_direction * (self.balance_kp * error + self.balance_kd * pitch_rate)
        target_velocity = float(np.clip(target_velocity, -self.balance_max_wheel_speed, self.balance_max_wheel_speed))
        max_step = self.balance_wheel_accel_limit / self.loop_frequency
        self.balance_wheel_velocity += float(np.clip(
            target_velocity - self.balance_wheel_velocity, -max_step, max_step
        ))
        return (self.balance_left_wheel_sign * self.balance_wheel_velocity,
                self.balance_right_wheel_sign * self.balance_wheel_velocity,
                pitch, pitch_rate)

    def transition_standup_phase(self, new_mode, reason=""):
        """Every phase starts from the current measured pose, not an old reference."""
        old_mode = self.mode
        self.mode = new_mode
        self.phase_timer = 0.0
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.phase_start_angles = np.asarray(
            self.robot_state_tmp.q, dtype=float
        )[:self.joint_num].copy()
        if new_mode == "KNEE_RELEASE":
            _, self.knee_release_start_pitch, _, _, _ = (
                self.get_full_orientation_state())
        if new_mode == "ABORT_HOLD":
            self.abort_hold_angles = self.phase_start_angles.copy()
            print(f"Stand-up ABORT: {reason or 'safety condition'}")
        else:
            print(f"Stand-up phase: {old_mode} -> {new_mode}")

    def begin_rl_direct(self):
            """Start the existing zero-command RL controller without RL_BLEND."""
            self.commands.fill(0.0)
            self.actions.fill(0.0)
            self.last_actions.fill(0.0)
            self.encoder_out.fill(0.0)
            self.proprio_history_vector.fill(0.0)
            self.is_first_rec_obs = True
            # Force policy evaluation on this control frame.
            self.loop_count = 0
            self.phase_timer = 0.0
            self.balance_wheel_velocity = 0.0
            self.mode = "WALK"
            print("KNEE_RELEASE -> WALK: direct zero-command RL handoff")

    def knee_handoff_ready(self):
        q = np.asarray(self.robot_state_tmp.q, dtype=float)[:self.joint_num]
        knee_l = self.joint_names.index('knee_L_Joint')
        knee_r = self.joint_names.index('knee_R_Joint')
        return max(abs(q[knee_l]), abs(q[knee_r])) <= self.rl_handoff_knee_angle
    
    def fsm_wheel_commands(self, target_pitch_rad):
        """Pitch feedback plus wheel-position anchor, copied from the useful stand-up idea."""
        left, right, pitch, pitch_rate = self.get_balance_wheel_commands(target_pitch_rad)
        displacement = self.get_wheel_displacement()
        if not np.isfinite(displacement):
            return 0.0, 0.0, pitch, pitch_rate, displacement

        base_velocity = self.balance_wheel_velocity - self.wheel_position_kp * displacement
        base_velocity = float(np.clip(
            base_velocity, -self.balance_max_wheel_speed, self.balance_max_wheel_speed
        ))
        left = self.balance_left_wheel_sign * base_velocity
        right = self.balance_right_wheel_sign * base_velocity
        return left, right, pitch, pitch_rate, displacement
    
    def build_knee_release_target(self, progress):
        """Move from the knee-supported pose toward the training nominal pose."""
        blend = self.quintic_blend(progress)
        target = self.phase_start_angles.copy()
        for i in self.get_leg_indices():
            target[i] = (
                self.phase_start_angles[i] * (1.0 - blend)
                + self.init_joint_angles[i] * blend
        )
        return target

    def apply_fsm_commands(self, leg_target, pitch_target_rad, stiffness, damping,
                           enable_feedback=True):
        """Apply leg target + pitch->hip + roll->knee + wheel feedback."""
        target = np.asarray(leg_target, dtype=float).copy()
        roll, pitch, roll_rate, pitch_rate, _ = self.get_full_orientation_state()
        if not np.all(np.isfinite([roll, pitch, roll_rate, pitch_rate])):
            self.transition_standup_phase("ABORT_HOLD", "invalid IMU orientation")
            return

        if enable_feedback:
            pitch_error = pitch - pitch_target_rad
            hip_comp = np.clip(
                -self.hip_pitch_kp * pitch_error - self.hip_pitch_kd * pitch_rate,
                -self.max_hip_compensation, self.max_hip_compensation
            )
            roll_comp = self.roll_compensation_direction * np.clip(
                -self.roll_knee_kp * roll - self.roll_knee_kd * roll_rate,
                -self.max_roll_compensation, self.max_roll_compensation
            )

            hip_l = self.joint_names.index('hip_L_Joint')
            hip_r = self.joint_names.index('hip_R_Joint')
            knee_l = self.joint_names.index('knee_L_Joint')
            knee_r = self.joint_names.index('knee_R_Joint')
            target[hip_l] += hip_comp
            target[hip_r] -= hip_comp
            target[knee_l] += roll_comp
            target[knee_r] += roll_comp

            left_wheel, right_wheel, _, _, _ = self.fsm_wheel_commands(pitch_target_rad)
        else:
            self.balance_wheel_velocity = 0.0
            left_wheel = right_wheel = 0.0

        wheels = self.get_wheel_indices()
        wheel_cmd = {wheels[0]: left_wheel, wheels[1]: right_wheel}
        for i in range(self.joint_num):
            if i in wheel_cmd:
                self.set_joint_command(i, 0, wheel_cmd[i], 0, 0, self.wheel_joint_damping)
            else:
                self.set_joint_command(i, target[i], 0, 0, stiffness, damping)
        self.fsm_last_leg_target = target.copy()

    def build_shift_target(self, progress):
        """COM-shift nominal target: only abad moves halfway; hip/knee stay at kneel."""
        blend = self.quintic_blend(progress)
        target = self.phase_start_angles.copy()
        fractions = {
            'abad_L_Joint': 0.50, 'abad_R_Joint': 0.50,
            'hip_L_Joint': 0.0, 'hip_R_Joint': 0.0,
            'knee_L_Joint': 0.0, 'knee_R_Joint': 0.0,
        }
        for name, fraction in fractions.items():
            i = self.joint_names.index(name)
            intermediate = self.stand_start_q[i] * (1.0 - fraction) + self.stand_target_angles[i] * fraction
            target[i] = self.phase_start_angles[i] * (1.0 - blend) + intermediate * blend
        return target

    def build_lift_target(self, progress):
        """Main body lift: Give knee progress a delay"""
        target = self.phase_start_angles.copy()
        hip_blend = self.quintic_blend(progress)
        # knee_progress = np.clip((progress - 0.18) / 0.82, 0.0, 1.0)
        delay = float(np.clip(self.knee_delay_fraction, 0.0, 0.9))
        knee_progress = np.clip((progress - delay) / (1.0 - delay), 0.0, 1.0)
        knee_blend = self.quintic_blend(knee_progress)
        for i, name in enumerate(self.joint_names):
            if (i + 1) % 4 == 0:
                continue
            blend = knee_blend if 'knee' in name else hip_blend
            target[i] = self.phase_start_angles[i] * (1.0 - blend) + self.stand_target_angles[i] * blend
        return target

    def continuous_standup_pitch_target(self, elapsed):
        """One continuous pitch reference across SHIFT + LIFT."""
        total = self.shift_duration + self.lift_duration
        p = np.clip(elapsed / max(total, 1e-6), 0.0, 1.0)
        blend = self.quintic_blend(p)
        final_pitch = np.radians(self.support_pitch_deg)
        return self.stand_start_pitch * (1.0 - blend) + final_pitch * blend

    def standup_safety_reason(self, expected_pitch_rad=None):
        """Small stand-up-only safety guard; no fall-recovery logic is imported."""
        roll, pitch, _, _, _ = self.get_full_orientation_state()
        q = np.asarray(self.robot_state_tmp.q, dtype=float)[:self.joint_num]
        dq = np.asarray(self.robot_state_tmp.dq, dtype=float)[:self.joint_num]
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
            return "non-finite joint state"
        if np.max(np.abs(dq)) > self.fsm_max_joint_speed:
            return "joint speed limit exceeded"
        if abs(np.degrees(roll)) > 25.0:
            return "roll safety limit exceeded"
        if abs(self.get_wheel_displacement()) > self.wheel_travel_limit:
            return "wheel travel limit exceeded"
        if self.fsm_last_leg_target is not None and self.phase_timer > 0.4:
            legs = self.get_leg_indices()
            err = np.max(np.abs(q[legs] - self.fsm_last_leg_target[legs]))
            if err > self.fsm_max_tracking_error:
                return "joint tracking error limit exceeded"
        return ""

    def handle_abort_hold(self):
        """Freeze the current pose after a stand-up safety abort."""
        if self.abort_hold_angles is None:
            self.abort_hold_angles = np.asarray(
                self.robot_state.q, dtype=float
            )[:self.joint_num].copy()
        for i in range(self.joint_num):
            if (i + 1) % 4 == 0:
                self.set_joint_command(i, 0, 0, 0, 0, self.wheel_joint_damping)
            else:
                self.set_joint_command(
                    i, self.abort_hold_angles[i], 0, 0,
                    self.abort_stiffness, self.abort_damping
                )

    def handle_standup_fsm(self):
        """Fixed kneel -> prepare support -> shift COM -> lift -> hold stand."""
        dt = 1.0 / self.loop_frequency
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)
        roll, pitch, roll_rate, pitch_rate, _ = self.get_full_orientation_state()

        if self.mode == "ABORT_HOLD":
            self.handle_abort_hold()
            return

        if self.mode == "PREPARE_SUPPORT":
            self.apply_fsm_commands(
                self.stand_start_q,
                self.stand_start_pitch,
                self.fsm_low_stiffness,
                self.fsm_low_damping,
                enable_feedback=True
            )
            reason = self.standup_safety_reason(self.stand_start_pitch)
            if reason:
                self.transition_standup_phase("ABORT_HOLD", reason)
            elif self.phase_timer >= self.prepare_duration:
                self.transition_standup_phase("SHIFT_CENTER_OF_MASS")

        elif self.mode == "SHIFT_CENTER_OF_MASS":
            progress = self.phase_timer / max(self.shift_duration, 1e-6)
            target = self.build_shift_target(progress)
            pitch_target = self.continuous_standup_pitch_target(
                self.phase_timer
            )
            self.apply_fsm_commands(
                target,
                pitch_target,
                min(
                    self.control_cfg['stiffness'],
                    self.fsm_low_stiffness * 1.4
                ),
                self.control_cfg['damping'],
                enable_feedback=True
            )
            reason = self.standup_safety_reason(pitch_target)
            if reason:
                self.transition_standup_phase("ABORT_HOLD", reason)
            elif self.phase_timer >= self.shift_duration:
                self.transition_standup_phase("KNEE_RELEASE")

        elif self.mode == "LIFT_BODY":
            progress = self.phase_timer / max(self.lift_duration, 1e-6)
            target = self.build_lift_target(progress)
            pitch_target = self.continuous_standup_pitch_target(
                self.shift_duration + self.phase_timer
            )
            self.apply_fsm_commands(
                target,
                pitch_target,
                self.control_cfg['stiffness'],
                self.control_cfg['damping'],
                enable_feedback=True
            )
            reason = self.standup_safety_reason(pitch_target)
            if reason:
                self.transition_standup_phase("ABORT_HOLD", reason)
            elif self.phase_timer >= self.lift_duration:
                self.transition_standup_phase("STAND_HOLD")

        elif self.mode == "KNEE_RELEASE":
            progress = self.phase_timer / max(
                self.knee_release_duration, 1e-6
            )
            target = self.build_knee_release_target(progress)
            pitch_blend = self.quintic_blend(progress)
            pitch_target = (
                self.knee_release_start_pitch * (1.0 - pitch_blend)
                + np.radians(self.balance_target_pitch_deg) * pitch_blend
            )

            self.apply_fsm_commands(
                target,
                pitch_target,
                self.control_cfg['stiffness'],
                self.control_cfg['damping'],
                enable_feedback=True
            )

            if self.knee_handoff_ready():
                fsm_q = np.asarray(
                    self.robot_cmd.q, dtype=float
                )[:self.joint_num].copy()
                fsm_dq = np.asarray(
                    self.robot_cmd.dq, dtype=float
                )[:self.joint_num].copy()

                handoff_q = np.asarray(
                    self.robot_state_tmp.q, dtype=float
                )[:self.joint_num].copy()

                self.begin_rl_direct()
                self.handle_walk_mode()

                rl_q = np.asarray(
                    self.robot_cmd.q, dtype=float
                )[:self.joint_num].copy()
                rl_dq = np.asarray(
                    self.robot_cmd.dq, dtype=float
                )[:self.joint_num].copy()

                wheels = self.get_wheel_indices()
                legs = self.get_leg_indices()
                knee_l = self.joint_names.index('knee_L_Joint')
                knee_r = self.joint_names.index('knee_R_Joint')
                max_leg_jump = float(np.max(
                    np.abs(rl_q[legs] - fsm_q[legs])
                ))

                print(
                    "RL_HANDOFF | "
                    f"progress={progress:.3f} | "
                    f"pitch={np.degrees(pitch):+.1f}deg | "
                    f"pitch_target={np.degrees(pitch_target):+.1f}deg | "
                    f"pitch_rate={pitch_rate:+.3f}rad/s | "
                    f"roll={np.degrees(roll):+.1f}deg | "
                    f"roll_rate={roll_rate:+.3f}rad/s | "
                    f"knees=[{handoff_q[knee_l]:+.3f},"
                    f"{handoff_q[knee_r]:+.3f}] | "
                    f"fsm_wheel=[{fsm_dq[wheels[0]]:+.3f},"
                    f"{fsm_dq[wheels[1]]:+.3f}] | "
                    f"rl_wheel=[{rl_dq[wheels[0]]:+.3f},"
                    f"{rl_dq[wheels[1]]:+.3f}] | "
                    f"max_leg_q_jump={max_leg_jump:.3f}"
                )

            elif self.phase_timer >= self.knee_release_timeout:
                self.transition_standup_phase(
                    "ABORT_HOLD",
                    "RL handoff knee angle not reached"
                )

        period = max(1, int(self.loop_frequency * 0.5))
        q_now = np.asarray(self.robot_state_tmp.q, dtype=float)
        knee_l = self.joint_names.index('knee_L_Joint')
        knee_r = self.joint_names.index('knee_R_Joint')

        if self.loop_count % period == 0:
            print(
                f"FSM={self.mode}, t={self.phase_timer:.2f}s, "
                f"roll={np.degrees(roll):+.1f}deg, "
                f"pitch={np.degrees(pitch):+.1f}deg, "
                f"wheel_dx={self.get_wheel_displacement():+.3f}rad,"
                f"knees=[{q_now[knee_l]:+.3f},{q_now[knee_r]:+.3f}]"
            )

        if self.mode not in {"ABORT_HOLD", "STAND_HOLD"}:
            self.phase_timer += dt

    def begin_rl_blend(self):
        """Reset recurrent/history state and start a zero-command RL handoff."""
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.rl_blend_start_q = np.asarray(
            self.robot_state_tmp.q, dtype=float
        )[:self.joint_num].copy()

        self.phase_timer = 0.0
        self.balance_wheel_velocity = 0.0
        self.commands.fill(0.0)
        self.actions.fill(0.0)
        self.last_actions.fill(0.0)
        self.encoder_out.fill(0.0)
        self.is_first_rec_obs = True

        self.loop_count = -1
        self.mode = "RL_BLEND"
        print("STAND_UP -> RL_BLEND (zero command)")

    def handle_rl_blend(self):
        """Blend posture into the original RL controller, then leave WALK untouched."""
        dt = 1.0 / self.loop_frequency
        self.commands.fill(0.0)

        self.handle_walk_mode()
        policy_q = np.asarray(self.robot_cmd.q, dtype=float).copy()
        policy_dq = np.asarray(self.robot_cmd.dq, dtype=float).copy()

        t = float(np.clip(
            self.phase_timer / max(
                self.s2s_rl_blend_duration, 1e-6
            ),
            0.0,
            1.0
        ))
        alpha = self.smoothstep(t)

        for i in range(self.joint_num):
            if (i + 1) % 4 == 0:
                self.set_joint_command(
                    i,
                    0,
                    alpha * policy_dq[i],
                    0,
                    0,
                    self.wheel_joint_damping
                )
            else:
                q_des = (
                    (1.0 - alpha) * self.rl_blend_start_q[i]
                    + alpha * policy_q[i]
                )
                self.set_joint_command(
                    i,
                    q_des,
                    0,
                    0,
                    self.control_cfg['stiffness'],
                    self.control_cfg['damping']
                )

        period = max(1, int(self.loop_frequency * 0.25))
        if self.loop_count % period == 0:
            pitch, pitch_rate = self.get_pitch_state()
            print(
                f"RL_BLEND {100.0 * alpha:5.1f}% | "
                f"pitch={np.degrees(pitch):6.1f} deg | "
                f"rate={pitch_rate:5.2f}"
            )

        self.phase_timer += dt
        if t >= 1.0:
            self.mode = "WALK"
            self.phase_timer = 0.0
            print("RL_BLEND -> WALK: original RL controller fully active")

    # Handle the walk mode where the robot moves based on computed actions
    def handle_walk_mode(self):
        # Update the temporary robot state and IMU data
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        # Execute actions every 'decimation' iterations
        if self.loop_count % self.control_cfg['decimation'] == 0:
            self.compute_observation()
            self.compute_encoder()
            self.compute_actions()
            # Clip the actions within predefined limits
            action_min = -self.rl_cfg['clip_scales']['clip_actions']
            action_max = self.rl_cfg['clip_scales']['clip_actions']
            self.actions = np.clip(self.actions, action_min, action_max)

            # swap actions positions back to deep first, only when action updated
            if self.rl_type == "isaaclab":
                self.actions = self.swap_positions(self.actions, reverse=True)

            # Training observations contain the raw previous policy action,
            # before deployment-only torque safety limiting below.
            self.last_actions = self.actions.copy()

        # Iterate over the joints and set commands based on actions
        joint_pos = np.array(self.robot_state_tmp.q)
        joint_vel = np.array(self.robot_state_tmp.dq)

        for i in range(len(joint_pos)):
            if (i + 1) % 4 != 0:
                # Compute the limits for the action based on joint position and velocity
                action_min = (joint_pos[i] - self.init_joint_angles[i] +
                              (self.control_cfg['damping'] * joint_vel[i] - self.control_cfg['user_torque_limit']) /
                              self.control_cfg['stiffness'])
                action_max = (joint_pos[i] - self.init_joint_angles[i] +
                              (self.control_cfg['damping'] * joint_vel[i] + self.control_cfg['user_torque_limit']) /
                              self.control_cfg['stiffness'])

                # Clip action within limits
                self.actions[i] = max(action_min / self.control_cfg['action_scale_pos'],
                                      min(action_max / self.control_cfg['action_scale_pos'], self.actions[i]))

                # Compute the desired joint position and set it
                pos_des = self.actions[i] * self.control_cfg['action_scale_pos'] + self.init_joint_angles[i]
                self.set_joint_command(i, pos_des, 0, 0, self.control_cfg['stiffness'], self.control_cfg['damping'])

            else:
                action_min = (
                    joint_vel[i] - self.wheel_joint_torque_limit / self.wheel_joint_damping
                ) / self.control_cfg['action_scale_vel']
                action_max = (
                    joint_vel[i] + self.wheel_joint_torque_limit / self.wheel_joint_damping
                ) / self.control_cfg['action_scale_vel']
                self.actions[i] = max(action_min, min(action_max, self.actions[i]))
                velocity_des = self.actions[i] * self.control_cfg['action_scale_vel']
                self.set_joint_command(i, 0, velocity_des, 0, 0, self.wheel_joint_damping)

    def swap_positions(self, initial_array, reverse=False, exclude_wheel=False):
        if not exclude_wheel:
            joint_idx_lab = [0, 4, 1, 5, 2, 6, 3, 7]
        else:
            joint_idx_lab = [0, 3, 1, 4, 2, 5]
        new_array = np.zeros(initial_array.shape)
        for i in range(len(joint_idx_lab)):
            if not reverse:
                new_array[i] = initial_array[joint_idx_lab[i]]
            else:
                new_array[joint_idx_lab[i]] = initial_array[i]
        return new_array
    
    def compute_observation(self):
        # Convert IMU orientation from quaternion to Euler angles (ZYX convention)
        imu_orientation = np.array(self.imu_data_tmp.quat)
        q_wi = R.from_quat(imu_orientation).as_euler('zyx')  # Quaternion to Euler ZYX conversion
        inverse_rot = R.from_euler('zyx', q_wi).inv().as_matrix()  # Get the inverse rotation matrix

        # Project the gravity vector (pointing downwards) into the body frame
        gravity_vector = np.array([0, 0, -1])  # Gravity in world frame (z-axis down)
        projected_gravity = np.dot(inverse_rot, gravity_vector)  # Transform gravity into body frame

        # Retrieve base angular velocity from the IMU data
        base_ang_vel = np.array(self.imu_data_tmp.gyro)
        # Apply IMU orientation offset correction (using Euler angles)
        rot = R.from_euler('zyx', self.imu_orientation_offset).as_matrix()  # Rotation matrix for offset correction
        base_ang_vel = np.dot(rot, base_ang_vel)  # Apply correction to angular velocity
        projected_gravity = np.dot(rot, projected_gravity)  # Apply correction to projected gravity

        # Retrieve joint positions and velocities from the robot state
        joint_positions = np.array(self.robot_state_tmp.q)
        joint_velocities = np.array(self.robot_state_tmp.dq)

        # Retrieve the last actions that were applied to the robot
        actions = np.array(self.last_actions)

        # Create a command scaler matrix for linear and angular velocities
        command_scaler = np.diag([
            self.user_cmd_cfg['lin_vel_x'],  # Scale factor for linear velocity in x direction
            self.user_cmd_cfg['lin_vel_y'],  # Scale factor for linear velocity in y direction
            self.user_cmd_cfg['ang_vel_yaw']  # Scale factor for yaw (angular velocity)
        ])

        # Apply scaling to the command inputs (velocity commands)
        self.scaled_commands = np.dot(command_scaler, self.commands)

        # Populate observation vector
        joint_pos_value = (joint_positions - self.init_joint_angles) * self.obs_scales['dof_pos']

        # In WF, joint pos does not include wheel speed, index(3, 7) needs to be removed
        joint_pos_input = np.array([joint_pos_value[idx] for idx in self.joint_pos_idxs])
        # swap positions in joint_pos, joint_vel and actions if mode is isaaclab
        if self.rl_type == "isaaclab":
            joint_pos_input = self.swap_positions(joint_pos_input, exclude_wheel=True)
            joint_velocities = self.swap_positions(joint_velocities)
            actions = self.swap_positions(actions)

        # Create the observation vector by concatenating various state variables:
        # - Base angular velocity (scaled)
        # - Projected gravity vector
        # - Joint positions (difference from initial angles, scaled)
        # - Joint velocities (scaled)
        # - Last actions applied to the robot
        # - Scaled command inputs
        obs = np.concatenate([
            base_ang_vel * self.obs_scales['ang_vel'],  # Scaled base angular velocity
            projected_gravity,  # Projected gravity vector in body frame
            joint_pos_input,  # Scaled joint positions
            joint_velocities * self.obs_scales['dof_vel'],  # Scaled joint velocities
            actions  # Last actions taken by the robot
        ])

        # Check if this is the first recorded observation
        if self.is_first_rec_obs:
            # Calculate the total size of the encoder input
            input_size = self.encoder_input_size
            
            # Initialize the proprioceptive history buffer with zeros
            self.proprio_history_buffer = np.zeros(input_size)

            # Fill the proprioceptive history buffer with the current observation for the entire history length
            for i in range(self.obs_history_length):
                self.proprio_history_buffer[i * self.observations_size:(i + 1) * self.observations_size] = obs

            # Update the flag to indicate that the first observation has been processed
            self.is_first_rec_obs = False
        
        # Shift the existing proprioceptive history buffer to the left
        self.proprio_history_buffer[:-self.observations_size] = self.proprio_history_buffer[self.observations_size:]

        # Add the current observation to the end of the proprioceptive history buffer
        self.proprio_history_buffer[-self.observations_size:] = obs

        # Convert the proprioceptive history buffer to a numpy array
        self.proprio_history_vector = np.array(self.proprio_history_buffer)

        # Clip the observation values to within the specified limits for stability
        self.observations = np.clip(
            obs, 
            -self.rl_cfg['clip_scales']['clip_observations'],  # Lower limit for clipping
            self.rl_cfg['clip_scales']['clip_observations']  # Upper limit for clipping
        )

    def compute_actions(self):
        """
        Computes the actions based on the current observations using the policy session.
        """
        # Concatenate observations into a single tensor and convert to float32
        input_tensor = np.concatenate([self.encoder_out, self.observations, self.scaled_commands], axis=0)
        input_tensor = input_tensor.astype(np.float32)
        
        # Create a dictionary of inputs for the policy session
        inputs = {self.policy_input_names[0]: input_tensor}
        
        # Run the policy session and get the output
        output = self.policy_session.run(self.policy_output_names, inputs)
        
        # Flatten the output and store it as actions
        self.actions = np.array(output).flatten()

    def compute_encoder(self):
        """
        Computes the encoder output based on the proprioceptive history buffer.

        This method first concatenates the proprioceptive history buffer into a single input tensor.
        Then it converts the input tensor to the float32 data type. After that, it creates a dictionary
        of inputs for the encoder session and runs the encoder session to get the output. Finally,
        it flattens the output and stores it as the encoder output.
        """
        # Concatenate the proprioceptive history buffer into a single tensor and convert to float32
        input_tensor = np.concatenate([self.proprio_history_buffer], axis=0)
        input_tensor = input_tensor.astype(np.float32)

        # Create a dictionary of inputs for the encoder session
        inputs = {self.encoder_input_names[0]: input_tensor}

        # Run the encoder session and get the output
        output = self.encoder_session.run(self.encoder_output_names, inputs)

        # Flatten the output and store it as the encoder output
        self.encoder_out = np.array(output).flatten()
 
    def set_joint_command(self, joint_index, q, dq, tau, kp, kd):
        """
        Sends a command to configure the state of a specific joint.
        This method updates the joint's desired position, velocity, torque, and control gains.
        Replace this implementation with the actual communication logic for your hardware.

        Parameters:
        joint_index (int): The index of the joint to be controlled.
        q (float): The desired joint position, typically in radians or degrees.
        dq (float): The desired joint velocity, typically in radians/second or degrees/second.
        tau (float): The desired joint torque, typically in Newton-meters (Nm).
        kp (float): The proportional gain for position control.
        kd (float): The derivative gain for velocity control.
        """
        self.robot_cmd.q[joint_index] = q
        self.robot_cmd.dq[joint_index] = dq
        self.robot_cmd.tau[joint_index] = tau
        self.robot_cmd.Kp[joint_index] = kp
        self.robot_cmd.Kd[joint_index] = kd

    def update(self):
        """Run S2S only at startup; once WALK is reached, use the original RL path."""
        self.robot_state_tmp = copy.deepcopy(self.robot_state)
        self.imu_data_tmp = copy.deepcopy(self.imu_data)

        if self.mode == "KNEEL_HOLD":
            self.handle_kneel_hold()
        elif self.mode in {
            "PREPARE_SUPPORT", "SHIFT_CENTER_OF_MASS",
            "LIFT_BODY", "KNEE_RELEASE", "STAND_HOLD", "ABORT_HOLD"
        }:
            self.handle_standup_fsm()
        elif self.mode == "RL_BLEND":
            self.handle_rl_blend()
        elif self.mode == "WALK":
            self.handle_walk_mode()

        self.loop_count += 1
        self.robot.publishRobotCmd(self.robot_cmd)
        
    # Callback function for receiving robot command data
    def robot_state_callback(self, robot_state: datatypes.RobotState):
        """
        Callback function to update the robot state from incoming data.
        
        Parameters:
        robot_state (datatypes.RobotState): The current state of the robot.
        """
        self.robot_state = robot_state
        self.has_robot_state = True

    # Callback function for receiving imu data
    def imu_data_callback(self, imu_data: datatypes.ImuData):
        """
        Callback function to update IMU data from incoming data.
        
        Parameters:
        imu_data (datatypes.ImuData): The IMU data containing stamp, acceleration, gyro, and quaternion.
        """
        self.imu_data.stamp = imu_data.stamp
        self.imu_data.acc = imu_data.acc
        self.imu_data.gyro = imu_data.gyro
        
        # Rotate quaternion values
        self.imu_data.quat[0] = imu_data.quat[1]
        self.imu_data.quat[1] = imu_data.quat[2]
        self.imu_data.quat[2] = imu_data.quat[3]
        self.imu_data.quat[3] = imu_data.quat[0]
        self.has_imu_data = True

    # Callback function for receiving sensor joy data
    def sensor_joy_callback(self, sensor_joy: datatypes.SensorJoy):
        # Check if the robot is in the calibration state and both L1 (button index 4) and Y (button index 3) buttons are pressed.
        if not self.start_controller and self.calibration_state == 0 and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[3] == 1:
          print(f"L1 + Y: start_controller...")
          self.start_controller = True

        # Check if both L1 (button index 4) and X (button index 2) are pressed to stop the controller
        if self.start_controller and sensor_joy.buttons[4] == 1 and sensor_joy.buttons[2] == 1:
          print(f"L1 + X: stop_controller...")
          self.start_controller = False

        linear_x  = sensor_joy.axes[1]
        linear_y  = sensor_joy.axes[0]
        angular_z = sensor_joy.axes[2]

        linear_x  = 1.0 if linear_x > 1.0 else (-1.0 if linear_x < -1.0 else linear_x)
        linear_y  = 1.0 if linear_y > 1.0 else (-1.0 if linear_y < -1.0 else linear_y)
        angular_z = 1.0 if angular_z > 1.0 else (-1.0 if angular_z < -1.0 else angular_z)

        # >>> S2S: do not inject a walking command while standing up or blending.
        if self.mode == "WALK":
            self.commands[0] = linear_x * 1.0
            self.commands[1] = 0.0
            self.commands[2] = angular_z * 0.8
        else:
            self.commands[:] = 0.0

    # Callback function for receiving diagnostic data
    def robot_diagnostic_callback(self, diagnostic_value: datatypes.DiagnosticValue):
      # Check if the received diagnostic data is related to calibration.
      if diagnostic_value.name == "calibration":
        print("=====0817 night debug ======")
        print(f"Calibration state: {diagnostic_value.code}")
        self.calibration_state = diagnostic_value.code