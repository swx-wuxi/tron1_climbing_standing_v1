import os
import sys
import time
import math
import argparse
import json
import mujoco
import mujoco.viewer as viewer
from functools import partial
import limxsdk
import limxsdk.robot.Rate as Rate
import limxsdk.robot.Robot as Robot
import limxsdk.robot.RobotType as RobotType
import limxsdk.datatypes as datatypes
import numpy as np

# ============================================================================
# 1. 实心楼梯参数
# ============================================================================
# 方向：沿世界坐标 +X 方向向前上楼梯。
STAIR_ENABLED = True
STAIR_STEP_HEIGHT = 0.16   # 每级高度 16 cm；例如 8 cm 请改成 0.08
STAIR_STEP_DEPTH = 0.26    # 每级踏面深度 26 cm
STAIR_STEP_WIDTH = 2.00    # 实心楼梯宽度 2 m
STAIR_STEP_NUM = 10
STAIR_START_X = 0.80
STAIR_CENTER_Y = 0.0

# 实心楼梯顶部平台
PLATFORM_ENABLED = True
PLATFORM_LENGTH = 2.00

# ============================================================================
# 2. 镂空楼梯参数：独立薄踏板，没有完整竖直踢面
# ============================================================================
OPEN_STAIR_ENABLED = True
OPEN_STAIR_SIDE = 1        # 1：实心楼梯 +Y 侧；-1：实心楼梯 -Y 侧
OPEN_STAIR_GAP = 0.20      # 与实心楼梯边缘之间的横向间隔
OPEN_STAIR_WIDTH = 2.00
OPEN_STAIR_TREAD_THICKNESS = 0.05

# 自动计算镂空楼梯中心 Y，避免和实心楼梯重叠。
OPEN_STAIR_CENTER_Y = (
    STAIR_CENTER_Y
    + OPEN_STAIR_SIDE
    * (STAIR_STEP_WIDTH / 2.0 + OPEN_STAIR_GAP + OPEN_STAIR_WIDTH / 2.0)
)

# 镂空楼梯顶部平台也使用薄板，而不是从地面填充到平台顶面。
OPEN_PLATFORM_ENABLED = True
OPEN_PLATFORM_THICKNESS = 0.02

# ============================================================================
# 3. 斜坡参数
# ============================================================================
RAMP_ENABLED = True

# 默认把斜坡放在实心楼梯的 -Y 侧，与 +Y 侧的镂空楼梯分开。
RAMP_SIDE = -1             # 1：实心楼梯 +Y 侧；-1：实心楼梯 -Y 侧
RAMP_GAP = 0.20            # 斜坡与实心楼梯边缘的横向间隔
RAMP_WIDTH = 2.00          # 斜坡宽度
RAMP_START_X = 0.80        # 斜坡起点 X

# RAMP_RUN_LENGTH 是斜坡在水平 X 方向上的投影长度，不是斜面本身长度。
RAMP_RUN_LENGTH = 8.00

# 默认斜坡顶部与楼梯顶部等高；改变楼梯级数/级高时会自动同步。
RAMP_HEIGHT = STAIR_STEP_NUM * STAIR_STEP_HEIGHT
RAMP_THICKNESS = 0.08      # 斜坡板厚度
RAMP_FRICTION = "1.0 0.005 0.0001"

# 自动计算斜坡中心 Y，避免与实心楼梯重叠。
RAMP_CENTER_Y = (
    STAIR_CENTER_Y
    + RAMP_SIDE
    * (STAIR_STEP_WIDTH / 2.0 + RAMP_GAP + RAMP_WIDTH / 2.0)
)

# 斜坡顶部平台
RAMP_PLATFORM_ENABLED = True
RAMP_PLATFORM_LENGTH = 2.00
RAMP_PLATFORM_THICKNESS = 0.08

def _validate_positive(name, value):
    """检查地形尺寸参数必须为正数。"""
    if value <= 0.0:
        raise ValueError(f"{name} 必须大于 0，当前值为 {value}。")

def _make_model_xml_with_terrains(model_path):
    """在 MuJoCo worldbody 中插入实心楼梯、镂空楼梯、斜坡及平台。"""
    if not STAIR_ENABLED and not OPEN_STAIR_ENABLED and not RAMP_ENABLED:
        return model_path

    with open(model_path, "r", encoding="utf-8") as f:
        xml_text = f.read()

    # 避免把已经生成过的 XML 再次作为输入，造成同名 geom 重复。
    generated_geom_names = (
        'name="stair_step_01"',
        'name="open_stair_step_01"',
        'name="training_ramp"',
    )
    if any(name in xml_text for name in generated_geom_names):
        print("*** Existing generated terrain detected; use this XML directly. ***")
        return model_path

    insert_pos = xml_text.rfind("</worldbody>")
    if insert_pos < 0:
        raise RuntimeError("robot.xml 中找不到 </worldbody>，无法自动插入地形。")

    terrain_geoms = []

    # ------------------------------------------------------------------
    # 1) 实心楼梯：每一级 box 从地面填充到当前踏面高度。
    # ------------------------------------------------------------------
    if STAIR_ENABLED:
        _validate_positive("STAIR_STEP_HEIGHT", STAIR_STEP_HEIGHT)
        _validate_positive("STAIR_STEP_DEPTH", STAIR_STEP_DEPTH)
        _validate_positive("STAIR_STEP_WIDTH", STAIR_STEP_WIDTH)
        if STAIR_STEP_NUM <= 0:
            raise ValueError("STAIR_STEP_NUM 必须是大于 0 的整数。")

        for i in range(STAIR_STEP_NUM):
            step_id = i + 1
            top_h = step_id * STAIR_STEP_HEIGHT
            center_x = (
                STAIR_START_X
                + i * STAIR_STEP_DEPTH
                + STAIR_STEP_DEPTH / 2.0
            )
            center_z = top_h / 2.0

            terrain_geoms.append(
                f'        <geom name="stair_step_{step_id:02d}" '
                f'type="box" '
                f'pos="{center_x:.6f} {STAIR_CENTER_Y:.6f} {center_z:.6f}" '
                f'size="{STAIR_STEP_DEPTH / 2.0:.6f} '
                f'{STAIR_STEP_WIDTH / 2.0:.6f} {center_z:.6f}" '
                f'rgba="0.45 0.45 0.45 1" '
                f'contype="1" conaffinity="1" '
                f'friction="1.0 0.005 0.0001"/>\n'
            )

        if PLATFORM_ENABLED:
            _validate_positive("PLATFORM_LENGTH", PLATFORM_LENGTH)
            platform_height = STAIR_STEP_NUM * STAIR_STEP_HEIGHT
            stair_end_x = STAIR_START_X + STAIR_STEP_NUM * STAIR_STEP_DEPTH
            platform_center_x = stair_end_x + PLATFORM_LENGTH / 2.0
            platform_center_z = platform_height / 2.0

            terrain_geoms.append(
                f'        <geom name="stair_platform" '
                f'type="box" '
                f'pos="{platform_center_x:.6f} {STAIR_CENTER_Y:.6f} '
                f'{platform_center_z:.6f}" '
                f'size="{PLATFORM_LENGTH / 2.0:.6f} '
                f'{STAIR_STEP_WIDTH / 2.0:.6f} {platform_center_z:.6f}" '
                f'rgba="0.50 0.50 0.50 1" '
                f'contype="1" conaffinity="1" '
                f'friction="1.0 0.005 0.0001"/>\n'
            )

    # ------------------------------------------------------------------
    # 2) 镂空楼梯：每一级仅为独立薄踏板。
    # ------------------------------------------------------------------
    if OPEN_STAIR_ENABLED:
        _validate_positive("OPEN_STAIR_WIDTH", OPEN_STAIR_WIDTH)
        _validate_positive(
            "OPEN_STAIR_TREAD_THICKNESS", OPEN_STAIR_TREAD_THICKNESS
        )
        if OPEN_STAIR_TREAD_THICKNESS >= STAIR_STEP_HEIGHT:
            raise ValueError(
                "OPEN_STAIR_TREAD_THICKNESS 必须小于 STAIR_STEP_HEIGHT，"
                "否则相邻台阶会在竖直方向连成实心结构。"
            )

        half_tread_thickness = OPEN_STAIR_TREAD_THICKNESS / 2.0

        for i in range(STAIR_STEP_NUM):
            step_id = i + 1
            top_h = step_id * STAIR_STEP_HEIGHT
            center_x = (
                STAIR_START_X
                + i * STAIR_STEP_DEPTH
                + STAIR_STEP_DEPTH / 2.0
            )

            # 使踏板上表面位于 top_h。
            center_z = top_h - half_tread_thickness

            terrain_geoms.append(
                f'        <geom name="open_stair_step_{step_id:02d}" '
                f'type="box" '
                f'pos="{center_x:.6f} {OPEN_STAIR_CENTER_Y:.6f} '
                f'{center_z:.6f}" '
                f'size="{STAIR_STEP_DEPTH / 2.0:.6f} '
                f'{OPEN_STAIR_WIDTH / 2.0:.6f} '
                f'{half_tread_thickness:.6f}" '
                f'rgba="0.20 0.55 0.85 1" '
                f'contype="1" conaffinity="1" '
                f'friction="1.0 0.005 0.0001"/>\n'
            )

        if OPEN_PLATFORM_ENABLED:
            _validate_positive("OPEN_PLATFORM_THICKNESS", OPEN_PLATFORM_THICKNESS)
            _validate_positive("PLATFORM_LENGTH", PLATFORM_LENGTH)
            platform_height = STAIR_STEP_NUM * STAIR_STEP_HEIGHT
            stair_end_x = STAIR_START_X + STAIR_STEP_NUM * STAIR_STEP_DEPTH
            platform_center_x = stair_end_x + PLATFORM_LENGTH / 2.0
            half_platform_thickness = OPEN_PLATFORM_THICKNESS / 2.0
            platform_center_z = platform_height - half_platform_thickness

            terrain_geoms.append(
                f'        <geom name="open_stair_platform" '
                f'type="box" '
                f'pos="{platform_center_x:.6f} {OPEN_STAIR_CENTER_Y:.6f} '
                f'{platform_center_z:.6f}" '
                f'size="{PLATFORM_LENGTH / 2.0:.6f} '
                f'{OPEN_STAIR_WIDTH / 2.0:.6f} '
                f'{half_platform_thickness:.6f}" '
                f'rgba="0.25 0.60 0.90 1" '
                f'contype="1" conaffinity="1" '
                f'friction="1.0 0.005 0.0001"/>\n'
            )

    # ------------------------------------------------------------------
    # 3) 斜坡：使用一个绕世界 Y 轴旋转的 box。
    # ------------------------------------------------------------------
    ramp_angle_rad = None
    ramp_surface_length = None
    ramp_end_x = None

    if RAMP_ENABLED:
        _validate_positive("RAMP_RUN_LENGTH", RAMP_RUN_LENGTH)
        _validate_positive("RAMP_HEIGHT", RAMP_HEIGHT)
        _validate_positive("RAMP_WIDTH", RAMP_WIDTH)
        _validate_positive("RAMP_THICKNESS", RAMP_THICKNESS)

        # 水平投影长度、目标高度和斜面实际长度的关系。
        ramp_angle_rad = math.atan2(RAMP_HEIGHT, RAMP_RUN_LENGTH)
        ramp_surface_length = math.hypot(RAMP_RUN_LENGTH, RAMP_HEIGHT)
        half_surface_length = ramp_surface_length / 2.0
        half_thickness = RAMP_THICKNESS / 2.0

        # MuJoCo 四元数顺序为 w x y z。
        # 绕 Y 轴旋转负角度，使斜坡沿 +X 方向升高。
        half_angle = -ramp_angle_rad / 2.0
        quat_w = math.cos(half_angle)
        quat_y = math.sin(half_angle)

        ramp_center_x = RAMP_START_X + RAMP_RUN_LENGTH / 2.0

        # 让斜坡“上表面”的低端恰好位于 z=0，高端位于 z=RAMP_HEIGHT。
        ramp_center_z = (
            RAMP_HEIGHT / 2.0
            - half_thickness * math.cos(ramp_angle_rad)
        )
        ramp_end_x = RAMP_START_X + RAMP_RUN_LENGTH

        terrain_geoms.append(
            f'        <geom name="training_ramp" '
            f'type="box" '
            f'pos="{ramp_center_x:.6f} {RAMP_CENTER_Y:.6f} '
            f'{ramp_center_z:.6f}" '
            f'quat="{quat_w:.9f} 0 {quat_y:.9f} 0" '
            f'size="{half_surface_length:.6f} '
            f'{RAMP_WIDTH / 2.0:.6f} {half_thickness:.6f}" '
            f'rgba="0.75 0.45 0.15 1" '
            f'contype="1" conaffinity="1" '
            f'friction="{RAMP_FRICTION}"/>\n'
        )

        if RAMP_PLATFORM_ENABLED:
            _validate_positive("RAMP_PLATFORM_LENGTH", RAMP_PLATFORM_LENGTH)
            _validate_positive("RAMP_PLATFORM_THICKNESS", RAMP_PLATFORM_THICKNESS)

            half_platform_thickness = RAMP_PLATFORM_THICKNESS / 2.0
            platform_center_x = ramp_end_x + RAMP_PLATFORM_LENGTH / 2.0
            platform_center_z = RAMP_HEIGHT - half_platform_thickness

            terrain_geoms.append(
                f'        <geom name="ramp_platform" '
                f'type="box" '
                f'pos="{platform_center_x:.6f} {RAMP_CENTER_Y:.6f} '
                f'{platform_center_z:.6f}" '
                f'size="{RAMP_PLATFORM_LENGTH / 2.0:.6f} '
                f'{RAMP_WIDTH / 2.0:.6f} '
                f'{half_platform_thickness:.6f}" '
                f'rgba="0.82 0.52 0.20 1" '
                f'contype="1" conaffinity="1" '
                f'friction="{RAMP_FRICTION}"/>\n'
            )

    terrain_xml = (
        "\n"
        "        <!-- Auto-added solid stairs, open stairs and ramp -->\n"
        + "".join(terrain_geoms)
        + "\n"
    )

    terrain_model_path = os.path.join(
        os.path.dirname(model_path),
        "robot_with_stairs_and_ramp.xml",
    )
    with open(terrain_model_path, "w", encoding="utf-8") as f:
        f.write(xml_text[:insert_pos] + terrain_xml + xml_text[insert_pos:])

    print(f"*** Terrain Model Generated: {terrain_model_path}")

    if STAIR_ENABLED:
        stair_total_height = STAIR_STEP_NUM * STAIR_STEP_HEIGHT
        stair_total_depth = STAIR_STEP_NUM * STAIR_STEP_DEPTH
        stair_end_x = STAIR_START_X + stair_total_depth
        platform_end_x = stair_end_x + PLATFORM_LENGTH
        print(
            f"    solid stair: center_y={STAIR_CENTER_Y:.3f}, "
            f"width={STAIR_STEP_WIDTH:.3f}, "
            f"step_height={STAIR_STEP_HEIGHT:.3f}, "
            f"step_depth={STAIR_STEP_DEPTH:.3f}, "
            f"total_height={stair_total_height:.3f}\n"
            f"    solid stair x=[{STAIR_START_X:.3f}, {stair_end_x:.3f}], "
            f"platform x=[{stair_end_x:.3f}, {platform_end_x:.3f}]"
        )

    if OPEN_STAIR_ENABLED:
        print(
            f"    open stair: center_y={OPEN_STAIR_CENTER_Y:.3f}, "
            f"width={OPEN_STAIR_WIDTH:.3f}, "
            f"tread_thickness={OPEN_STAIR_TREAD_THICKNESS:.3f}"
        )

    if RAMP_ENABLED:
        ramp_angle_deg = math.degrees(ramp_angle_rad)
        ramp_platform_end_x = ramp_end_x + RAMP_PLATFORM_LENGTH
        print(
            f"    ramp: center_y={RAMP_CENTER_Y:.3f}, "
            f"width={RAMP_WIDTH:.3f}, height={RAMP_HEIGHT:.3f}, "
            f"run={RAMP_RUN_LENGTH:.3f}, "
            f"surface_length={ramp_surface_length:.3f}, "
            f"angle={ramp_angle_deg:.2f} deg\n"
            f"    ramp x=[{RAMP_START_X:.3f}, {ramp_end_x:.3f}], "
            f"platform x=[{ramp_end_x:.3f}, {ramp_platform_end_x:.3f}]"
        )

    return terrain_model_path

class SimulatorMujoco:
    def __init__(self, asset_path, joint_sensor_names, robot, initial_qpos=None):
        self.robot = robot
        self.joint_sensor_names = joint_sensor_names
        self.joint_num = len(joint_sensor_names)

        # Load the MuJoCo model and data from the specified XML asset path
        self.mujoco_model = mujoco.MjModel.from_xml_path(asset_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        print("\n========== MUJOCO ACTUATOR MAP ==========")
        for i in range(self.mujoco_model.nu):
            aname = mujoco.mj_id2name(
                self.mujoco_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
            )

            joint_id = self.mujoco_model.actuator_trnid[i, 0]
            jname = mujoco.mj_id2name(
                self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )

            limited = self.mujoco_model.actuator_forcelimited[i]
            frange = self.mujoco_model.actuator_forcerange[i]

            print(
                f"[{i}] actuator={aname} -> joint={jname} | "
                f"force_limited={limited} | "
                f"force_range=[{frange[0]:.1f}, {frange[1]:.1f}]"
            )
        print("=========================================\n")
        # Optionally start from a pose saved in manual-edit mode. Applying the
        # pose before opening the passive viewer prevents one simulation step
        # from occurring at the model's default pose.
        if initial_qpos is not None:
            if len(initial_qpos) != self.mujoco_model.nq:
                raise ValueError(
                    "Loaded pose has "
                    f"{len(initial_qpos)} qpos values, but this model needs "
                    f"{self.mujoco_model.nq}."
                )
            self.mujoco_data.qpos[:] = initial_qpos
            self.mujoco_data.qvel[:] = 0.0
            self.mujoco_data.ctrl[:] = 0.0
            mujoco.mj_forward(self.mujoco_model, self.mujoco_data)

        # Launch the MuJoCo viewer in passive mode with custom settings
        self.viewer = viewer.launch_passive(
            self.mujoco_model,
            self.mujoco_data,
            key_callback=self.key_callback,
            show_left_ui=True,
            show_right_ui=True
        )
        self.viewer.cam.distance = 10   # Set camera distance
        self.viewer.cam.elevation = -20 # Set camera elevation

        self.dt = self.mujoco_model.opt.timestep  # Get simulation timestep
        self.fps = 1 / self.dt                    # Calculate frames per second (FPS)

        # Initialize robot command data with default values
        self.robot_cmd = datatypes.RobotCmd()
        self.robot_cmd.mode = [0. for _ in range(0, self.joint_num)]
        self.robot_cmd.q = [0. for _ in range(0, self.joint_num)]
        self.robot_cmd.dq = [0. for _ in range(0, self.joint_num)]
        self.robot_cmd.tau = [0. for _ in range(0, self.joint_num)]
        self.robot_cmd.Kp = [0. for _ in range(0, self.joint_num)]
        self.robot_cmd.Kd = [0. for _ in range(0, self.joint_num)]

        # Initialize robot state data with default values
        self.robot_state = datatypes.RobotState()
        self.robot_state.tau = [0. for _ in range(0, self.joint_num)]
        self.robot_state.q = [0. for _ in range(0, self.joint_num)]
        self.robot_state.dq = [0. for _ in range(0, self.joint_num)]

        # Initialize IMU data structure
        self.imu_data = datatypes.ImuData()

        # Set up callback for receiving robot commands in simulation mode
        self.robotCmdCallbackPartial = partial(self.robotCmdCallback)
        self.robot.subscribeRobotCmdForSim(self.robotCmdCallbackPartial)

    # Callback function for receiving robot command data
    def robotCmdCallback(self, robot_cmd: datatypes.RobotCmd):
        self.robot_cmd = robot_cmd

    # Callback for keypress events in the MuJoCo viewer
    def key_callback(self, keycode):
        pass

    def run(self):
        frame_count = 0
        self.rate = Rate(self.fps)  # Set the update rate according to FPS

        while self.viewer.is_running():
            # Step the MuJoCo physics simulation
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

            # 打印机器人当前坐标
            if frame_count % 500 == 0:
                x = self.mujoco_data.qpos[0]
                y = self.mujoco_data.qpos[1]
                z = self.mujoco_data.qpos[2]
                #print(f"\033[1;32m[机器人坐标] X: {x:7.3f} | Y: {y:7.3f} | Z: {z:7.3f}\033[0m")

            # Update robot state data from simulation
            for i in range(self.joint_num):
                self.robot_state.q[i] = self.mujoco_data.qpos[i + 7]
                self.robot_state.dq[i] = self.mujoco_data.qvel[i + 6]
                self.robot_state.tau[i] = self.mujoco_data.ctrl[i]

                # Apply control commands to the robot based on the received robot command data
                self.mujoco_data.ctrl[i] = (
                    self.robot_cmd.Kp[i] * (self.robot_cmd.q[i] - self.robot_state.q[i]) +
                    self.robot_cmd.Kd[i] * (self.robot_cmd.dq[i] - self.robot_state.dq[i]) +
                    self.robot_cmd.tau[i]
                )
            
            # Set the timestamp for the current robot state and publish it
            self.robot_state.stamp = time.time_ns()
            self.robot.publishRobotStateForSim(self.robot_state)

            # Extract IMU data
            imu_quat_id = mujoco.mj_name2id(
                self.mujoco_model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                "quat"
            )
            self.imu_data.quat[0] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 0]
            self.imu_data.quat[1] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 1]
            self.imu_data.quat[2] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 2]
            self.imu_data.quat[3] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + 3]

            imu_gyro_id = mujoco.mj_name2id(
                self.mujoco_model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                "gyro"
            )
            self.imu_data.gyro[0] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_gyro_id] + 0]
            self.imu_data.gyro[1] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_gyro_id] + 1]
            self.imu_data.gyro[2] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_gyro_id] + 2]

            imu_acc_id = mujoco.mj_name2id(
                self.mujoco_model,
                mujoco.mjtObj.mjOBJ_SENSOR,
                "acc"
            )
            self.imu_data.acc[0] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_acc_id] + 0]
            self.imu_data.acc[1] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_acc_id] + 1]
            self.imu_data.acc[2] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_acc_id] + 2]

            # Set the timestamp for the current IMU data and publish it
            self.imu_data.stamp = time.time_ns()
            self.robot.publishImuDataForSim(self.imu_data)

            # Sync the viewer every 20 frames for smoother visualization
            if frame_count % 20 == 0:
                self.viewer.sync()

            frame_count += 1
            self.rate.sleep()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="TRON1 MuJoCo simulator"
    )
    parser.add_argument(
        "robot_ip",
        nargs="?",
        default="127.0.0.1",
        help="simulation SDK IP address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--manual-edit",
        action="store_true",
        help=(
            "use MuJoCo managed mode so Pause/Run and Joint sliders work; "
            "SDK publishing is disabled"
        ),
    )
    parser.add_argument(
        "--save-pose",
        metavar="FILE",
        help=(
            "save the final qpos to FILE after closing --manual-edit mode "
            "(JSON format)"
        ),
    )
    parser.add_argument(
        "--load-pose",
        metavar="FILE",
        help="load qpos from a pose JSON file before normal simulation starts",
    )
    args = parser.parse_args()

    if args.save_pose and not args.manual_edit:
        parser.error("--save-pose must be used together with --manual-edit")
    if args.manual_edit and args.load_pose:
        parser.error("--load-pose is for normal simulation mode, not --manual-edit")

    robot_type = os.getenv("ROBOT_TYPE")

    # Check if the ROBOT_TYPE environment variable is set, otherwise exit with an error
    if not robot_type:
        print("Error: Please set the ROBOT_TYPE using 'export ROBOT_TYPE=<robot_type>'.")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the robot model XML file based on the robot type
    model_path = f'{script_dir}/robot-description/pointfoot/{robot_type}/xml/robot.xml'

    # Check if the model file exists, otherwise exit with an error
    if not os.path.exists(model_path):
        print(f"Error: The file {model_path} does not exist. Please ensure the ROBOT_TYPE is set correctly.")
        sys.exit(1)

    print(f"*** Base Model Loaded: robot-description/pointfoot/{robot_type}/xml/robot.xml ***")

    # 自动生成“实心楼梯 + 镂空楼梯 + 斜坡 + 顶部平台”的 MuJoCo XML
    model_path = _make_model_xml_with_terrains(model_path)

    # In manual mode MuJoCo owns the stepping loop, so its Pause/Run buttons
    # and Joint sliders work. This mode intentionally does not connect to the
    # SDK or publish RobotState/IMU data.
    if args.manual_edit:
        print("*** Manual joint-edit mode ***")
        print("Pause the simulation, then adjust the right-side Joint sliders.")
        if args.save_pose:
            print(f"The final pose will be saved to: {args.save_pose}")
        else:
            print("No --save-pose file was given; this pose will not be saved.")
        print("Close the viewer or choose File -> Quit to finish manual mode.")
        manual_model = mujoco.MjModel.from_xml_path(model_path)
        manual_data = mujoco.MjData(manual_model)
        viewer.launch(manual_model, manual_data)

        if args.save_pose:
            pose_path = os.path.abspath(os.path.expanduser(args.save_pose))
            pose_dir = os.path.dirname(pose_path)
            if pose_dir:
                os.makedirs(pose_dir, exist_ok=True)
            pose_data = {
                "robot_type": robot_type,
                "nq": int(manual_model.nq),
                "qpos": [float(value) for value in manual_data.qpos],
            }
            with open(pose_path, "w", encoding="utf-8") as pose_file:
                json.dump(pose_data, pose_file, indent=2)
            print(f"Pose saved: {pose_path}")
        sys.exit(0)

    initial_qpos = None
    if args.load_pose:
        pose_path = os.path.abspath(os.path.expanduser(args.load_pose))
        if not os.path.exists(pose_path):
            parser.error(f"pose file does not exist: {pose_path}")
        with open(pose_path, "r", encoding="utf-8") as pose_file:
            pose_data = json.load(pose_file)
        if "qpos" not in pose_data or not isinstance(pose_data["qpos"], list):
            parser.error(f"pose file has no valid qpos list: {pose_path}")
        saved_robot_type = pose_data.get("robot_type")
        if saved_robot_type and saved_robot_type != robot_type:
            parser.error(
                f"pose is for ROBOT_TYPE={saved_robot_type}, "
                f"but current ROBOT_TYPE={robot_type}"
            )
        initial_qpos = [float(value) for value in pose_data["qpos"]]
        print(f"*** Initial pose loaded: {pose_path} ***")

    # Normal SDK simulation mode.
    robot = Robot(RobotType.PointFoot, True)
    if not robot.init(args.robot_ip):
        sys.exit(1)

    # Define the names of the joint sensors used in the robot
    if robot_type.startswith("WF"):
        joint_sensor_names = [
            "abad_L_Joint",
            "hip_L_Joint",
            "knee_L_Joint",
            "wheel_L_Joint",
            "abad_R_Joint",
            "hip_R_Joint",
            "knee_R_Joint",
            "wheel_R_Joint"
        ]
    elif robot_type.startswith("SF"):
        joint_sensor_names = [
            "abad_L_Joint",
            "hip_L_Joint",
            "knee_L_Joint",
            "ankle_L_Joint",
            "abad_R_Joint",
            "hip_R_Joint",
            "knee_R_Joint",
            "ankle_R_Joint"
        ]
    else:
        joint_sensor_names = [
            "abad_L_Joint",
            "hip_L_Joint",
            "knee_L_Joint",
            "abad_R_Joint",
            "hip_R_Joint",
            "knee_R_Joint"
        ]

    # Create and run the MuJoCo simulator instance
    simulator = SimulatorMujoco(
        model_path,
        joint_sensor_names,
        robot,
        initial_qpos=initial_qpos,
    )
    simulator.run()
