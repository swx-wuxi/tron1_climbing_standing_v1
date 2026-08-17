# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from legged_gym.envs.base.base_config import BaseConfig
from legged_gym import LEGGED_GYM_ROOT_DIR

import os
robot_type = os.getenv("ROBOT_TYPE")

class BipedCfgPF(BaseConfig):
    class env:
        num_envs = 8192
        # Actor observation: ang vel, gravity, joint pos/vel, last actions, clock, gait params.
        num_observations = 30
        num_height_samples = 77
        # Critic: actor obs + base lin vel + foot contact forces + height scan.
        num_critic_observations = num_observations + 3 + 6 + num_height_samples
        num_actions = 6
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        obs_history_length = 10  # number of observations stacked together
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 0.5

    class terrain:
        mesh_type = "trimesh"  # none, plane, heightfield or trimesh
        horizontal_scale = 0.052  # [m], 0.26m stair tread = 5 cells
        vertical_scale = 0.005  # [m]
        border_size = 5  # [m]
        curriculum = True
        static_friction = 0.6
        dynamic_friction = 0.6
        restitution = 0.0
        # rough terrain only:
        measure_heights = False
        critic_measure_heights = True
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # start from the easiest stair level
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.10, 0.00, 0.90, 0.00, 0.00]
        stair_up_only = True
        # Stair curriculum: low rows start flat and deep, high rows reach 16 cm / 26 cm.
        stair_step_height = None
        min_stair_step_height = 0.0  # [m]
        stair_step_width = 0.40  # [m], fallback / easiest tread depth
        min_stair_step_width = 0.26  # [m]
        max_stair_step_width = 0.40  # [m]
        stair_approach_size = 1.2  # [m]
        stair_step_width_scales = [1.0]
        max_stair_step_height = 0.16  # [m]
        randomize_init_xy = True
        random_init_xy_range = 0.15  # [m]
        # trimesh only:
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )

    class commands:
        curriculum = False
        smooth_max_lin_vel_x = 1.7
        smooth_max_lin_vel_y = 1.7
        non_smooth_max_lin_vel_x = 1.7
        non_smooth_max_lin_vel_y = 1.7
        max_ang_vel_yaw = 1.0
        curriculum_threshold = 0.75
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = False  # keep yaw command fixed for stair-up-only training
        min_norm = 0.1
        static_command_prob = 0.07
        yaw_only_command_prob = 0.07
        straight_command_prob = 0.20
        stair_only_commands = False
        rough_lin_vel_x = [-0.6, 0.8]
        rough_lin_vel_y = [-0.3, 0.3]
        rough_ang_vel_yaw = [-0.4, 0.4]
        discrete_lin_vel_x = [-0.5, 0.7]
        discrete_lin_vel_y = [-0.25, 0.25]
        discrete_ang_vel_yaw = [-0.35, 0.35]
        stair_lin_vel_x = [0.05, 0.65]
        stair_ang_vel_yaw = [-0.35, 0.35]

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]  # min max [m/s]
            # lin_vel_x = [-1.7, 1.7]  # min max [m/s]
            # lin_vel_y = [-1.7, 1.7]  # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]  # min max [rad/s]
            heading = [-3.14159, 3.14159]

    class gait:
        num_gait_params = 4
        resampling_time = 5  # time before command are changed[s]
        randomize_phase_on_reset = True
        reset_phase_range = [0.0, 1.0]

        class ranges:
            frequencies = [1.4, 2.0]
            offsets = [0.45, 0.55]  # keep near alternating gait
            # durations = [0.3, 0.8]  # small durations(<0.4) is hard to learn
            # frequencies = [2, 2]
            # offsets = [0.5, 0.5]
            durations = [0.5, 0.5]
            swing_height = [0.24, 0.34]

    class init_state:
        pos = [0.0, 0.0, 0.8]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        randomize_reset_yaw = True
        reset_yaw_range = [-0.10, 0.10]  # rad
        randomize_reset_base_height = True
        reset_base_height_range = [-0.03, 0.03]  # m, additive around init z
        randomize_reset_root_velocity = True
        reset_lin_vel_range = [-0.15, 0.15]  # m/s, xyz
        reset_ang_vel_range = [-0.15, 0.15]  # rad/s, xyz
        randomize_reset_dof_pos = True
        reset_dof_pos_range = [-0.03, 0.03]  # rad, additive around default pose
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_R_Joint": 0.0,
        }

    class control:
        action_scale = 0.25

        control_type = "P"
        stiffness = {
            "abad_L_Joint": 42,
            "hip_L_Joint": 42,
            "knee_L_Joint": 42,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 42,
            "hip_R_Joint": 42,
            "knee_R_Joint": 42,
            "foot_R_Joint": 0.0,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 2.5,
            "hip_L_Joint": 2.5,
            "knee_L_Joint": 2.5,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 2.5,
            "hip_R_Joint": 2.5,
            "knee_R_Joint": 2.5,
            "foot_R_Joint": 0.0,
        }  # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        user_torque_limit = 80.0
        max_power = 1000.0  # [W]

    class asset:
        file = "{}/resources/robots/{}/urdf/robot.urdf".format(LEGGED_GYM_ROOT_DIR, robot_type)
        name = "pointfoot_flat"
        foot_name = "foot"
        foot_radius = 0.03
        penalize_contacts_on = ["knee", "hip"]
        terminate_after_contacts_on = ["abad", "base"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = (
            False  # Some .obj meshes must be flipped from y-up to z-up
        )

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.25, 1.4]
        randomize_restitution = True
        restitution_range = [0.0, 0.5]
        randomize_base_mass = True
        added_mass_range = [-0.5, 3.0]
        randomize_base_com = True
        rand_com_vec = [0.025, 0.015, 0.025]
        randomize_inertia = True
        randomize_inertia_range = [0.85, 1.15]
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.6
        rand_force = True
        force_resampling_time_s = 8
        max_force = 15.0
        rand_force_curriculum_level = 0
        randomize_Kp = True
        randomize_Kp_range = [0.85, 1.15]
        randomize_Kd = True
        randomize_Kd_range = [0.85, 1.15]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.85, 1.15]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        randomize_imu_offset = True
        randomize_imu_offset_range = [-0.8, 0.8]
        delay_ms_range = [0, 15]

    class rewards:
        class scales:
            # Survival / tracking / command-relative motion
            keep_balance = 1.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.6
            command_progress = 0.8
            heading_alignment = -1.0
            command_lateral_drift = 0.0
            no_yaw_drift = 0.0
            command_pose_error = 0.0
            expected_position_error = -0.20

            # Foot posture / mirror constraints
            feet_x_separation = -0.55
            feet_x_body_position = -0.35
            human_gait_mirror = -0.25
            feet_y_distance = -0.35
            feet_y_symmetry = -0.20
            feet_distance = -5.0

            # Swing shaping, kept moderate to avoid forcing high steps on flat ground
            swing_foot_clearance = 0.6
            swing_foot_forward = 0.3

            # Regularization
            base_height = -2.0
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            torques = -0.00008
            dof_acc = -2.5e-7
            action_rate = -0.01
            action_smooth = -0.01
            dof_pos_limits = -2.0
            collision = -1.0
            orientation = -10.0

            # Disabled noisy contact/landing terms for this stage
            feet_regulation = 0.0
            foot_landing_vel = 0.0
            tracking_contacts_shaped_force = 0.0
            tracking_contacts_shaped_vel = 0.0

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        ang_tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.75 # 0.58
        # Adaptive swing-clearance target. S22 reaches 22 cm stairs, so the cap
        # must leave room for the stair height plus the safety margin.
        feet_height_target = 0.34  # maximum adaptive clearance [m]
        feet_height_target_flat = 0.10  # nominal flat-ground clearance [m]
        feet_height_safety_margin = 0.07  # clearance above an upcoming step [m]
        feet_height_lookahead = [0.08, 0.16, 0.24, 0.32]  # samples ahead of each foot [m]
        feet_height_phase_power = 1.0  # exponent of the half-sine swing trajectory
        feet_clearance_sigma = 0.015
        feet_height_upper_margin = 0.06
        feet_clearance_high_sigma = 0.04
        swing_forward_vel_target = 0.5
        cmd_min_speed = 0.03
        cmd_min_yaw = 0.05
        command_pose_min_speed = 0.05
        command_pose_min_yaw = 0.05
        command_pose_lateral_sigma = 0.35
        command_pose_forward_sigma = 0.60
        command_pose_yaw_sigma = 0.35
        command_pose_position_sigma = 0.40
        command_pose_max_error = 1.0
        expected_lateral_deadband = 0.0
        expected_lateral_sigma = 0.04
        expected_angle_min_step = 0.02
        expected_angle_deadband = 0.0
        expected_angle_sigma = 0.45
        expected_angle_weight = 0.25
        expected_yaw_deadband = 0.0
        expected_yaw_sigma = 0.02
        expected_yaw_weight = 0.20
        command_pose_grace_time = 2.0
        command_pose_time_decay_power = 0.5
        command_pose_forward_weight = 0.15
        command_pose_yaw_weight = 0.5
        min_feet_distance = 0.115

        # Body-frame foot posture limits. These prevent front-probe / rear-prop
        # shortcuts and encourage a compact, alternating biped gait.
        max_feet_x_separation = 0.32
        swing_feet_x_separation_weight = 0.30
        max_foot_x_forward = 0.38
        max_foot_x_backward = 0.30
        min_feet_y_distance = 0.14
        feet_y_symmetry_sigma = 0.04
        human_step_length_min = 0.06
        human_step_length_max = 0.22
        human_step_length_sigma = 0.06
        human_gait_center_sigma = 0.06
        human_gait_late_swing_height = 0.12
        human_gait_phase_sigma = 0.08
        human_gait_center_weight = 0.6
        human_gait_step_length_weight = 0.6
        human_gait_phase_weight = 1.0

        about_landing_threshold = 0.08
        max_contact_force = 100.0  # forces above this value are penalized
        kappa_gait_probs = 0.05
        gait_force_sigma = 25.0
        gait_vel_sigma = 0.25
        gait_height_sigma = 0.005

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            contact_forces = 1.0
            torque = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [5, -5, 3]  # [m]
        # lookat = [11.0, 5, 3.0]  # [m]
        lookat = [0, 0, 0]  # [m]
        realtime_plot = True

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )


class BipedCfgPPOPF(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class MLP_Encoder:
        output_detach = True
        num_input_dim = BipedCfgPF.env.num_observations * BipedCfgPF.env.obs_history_length
        num_output_dim = 3
        hidden_dims = [256, 128]
        activation = "elu"
        orthogonal_init = False

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        orthogonal_init = False

    class algorithm:
        # PPO training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

        # Extra training params
        est_learning_rate = 1.0e-3
        ts_learning_rate = 1.0e-4
        critic_take_latent = True

    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 15000  # number of policy updates

        # logging
        logger = "tensorboard"
        exptid = ""
        wandb_project = "legged_gym_PF"
        save_interval = 500  # check for potential saves every this many iterations
        experiment_name = robot_type
        run_name = "S37_from_S36_45000"
        # load and resume
        resume = True
        reset_iteration_on_resume = True
        load_run = "Jul10_20-33-50_S36_from_S34_30000"
        checkpoint = 45000
        resume_path = "/mnt/sdb1/wuzeyu128/pointfoot_logs/PF_TRON1A/Jul10_20-33-50_S36_from_S34_30000/model_45000.pt"
