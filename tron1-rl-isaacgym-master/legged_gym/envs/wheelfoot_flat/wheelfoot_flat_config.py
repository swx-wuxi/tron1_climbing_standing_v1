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


class BipedCfgWF(BaseConfig):
    class env:
        num_envs = 8192
        # Actor and Critic share the same 28-D proprioceptive observation.
        num_observations = 28
        num_height_samples = 117
        # Critic additionally sees base velocity + 13 x 9 terrain heights.
        num_critic_observations = 3 + num_observations + num_height_samples
        num_actions = 8
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        obs_history_length = 10  # number of observations stacked together
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 0.5

    class terrain:
        # S21 triangle-mesh terrain training from the original MLP3 Plane policy.
        mesh_type = "trimesh"   # trimesh, heightfield, plane
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 0.8
        dynamic_friction = 0.6
        restitution = 0.0
        # rough terrain only:
        measure_heights = False
        critic_measure_heights = True
        measured_points_x = [
            -0.4,
            -0.2,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            1.0,
            1.2,
        ]  # 13 points: dense near-field plus forward stair look-ahead
        measured_points_y = [
            -0.5,
            -0.35,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.35,
            0.5,
        ]  # 9 points: wheel tracks plus wider lateral context
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # S21 curriculum starts entirely at Level 0
        terrain_length = 8.0
        terrain_width = 8.0
        stair_width_range = [0.26, 0.26]  # level 0--10: fixed 26 cm tread depth
        stair_height_range = [0.00, 0.30]  # level 0--10: 0--30 cm riser height
        stair_step_width_scales = [1.0]  # no per-column depth/height scaling
        max_stair_step_height = 0.30
        stair_platform_size = 3.0  # [m], flat center approach before first riser
        num_rows = 11  # terrain levels 0--10
        num_cols = 20  # number of terrain cols (types)
        # Scene 6 is a centered, 2 m wide straight staircase.
        # Change only this value to configure its parallel-environment share.
        scene6_enabled = True
        scene6_probability = 0.00
        scene6_stair_width = 2.0
        # Reset before a wheel's lateral contact region can leave the open
        # stair edge. The wheel collision width is 0.05 m, so this also
        # includes a small numerical/contact safety allowance.
        scene6_wheel_edge_margin = 0.05
        # Same 3 m central platform as ordinary stairs: the first riser is
        # 1.5 m ahead of the nominal spawn.
        scene6_spawn_length = 1.5
        # Keep the complete robot footprint away from the terrain-tile seam
        # so a reset cannot repeatedly place it in an invalid boundary region.
        scene6_spawn_edge_margin = 1.0
        scene6_trailing_ground_length = 0.40
        scene6_descent_finish_margin = 0.05
        scene6_descent_confirm_time = 0.10
        scene6_descent_max_vertical_speed = 0.30
        # Keep longitudinal spawn variation, but place the base exactly on the
        # central stair line in the lateral direction.
        scene6_spawn_xy_jitter = [0.15, 0.00]
        scene6_success_distance = 4.0
        scene6_episode_length_s = 20.0
        # terrain types: [smooth, rough, stairs up, stairs down, discrete, scene 6]
        # Category 0 is forced to be truly flat for balance retention.
        smooth_slope_as_flat = True
        terrain_proportions = [
            0.30,  # triangle-mesh flat terrain, all commands
            0.00,  # rough slope
            0.60,  # stairs up: 0.40 straight + 0.20 forward-and-turn
            0.00,  # stairs down
            0.10,  # discrete obstacles
            0.00,  # scene 6, forward only
        ]
        # Absolute terrain share assigned to forward-plus-turn ordinary stairs.
        # The remaining 0.40 of the 0.60 stair share stays straight-only.
        stair_full_command_probability = 0.20
        # trimesh only:
        slope_treshold = 0.20  # every nonzero curriculum step has a vertical riser
        terrain_upgrade_min_steps = 920
        terrain_upgrade_forward_distance = 4.0
        # Ordinary straight stairs promote by measured climbing ability:
        # reached steps >= min(max_steps, ratio * total forward stair steps).
        # The flat approach and commanded speed/distance do not count.
        stair_straight_upgrade_max_steps = 6.0
        stair_straight_upgrade_total_step_ratio = 0.60
        # Forward-plus-turn stairs use command tracking for curriculum success
        # because deliberate yaw makes pure +X travel an unsuitable criterion.
        stair_full_command_upgrade_tracking_ratio = 0.75
        stair_full_command_downgrade_tracking_ratio = 0.50
        # Ordinary straight-command stairs use their own 2 m-wide virtual
        # corridor. Its reset boundary is tied to the corridor half-width;
        # Scene 6 independently retains scene6_stair_width above.
        stair_straight_centerline_width = 2.0
        stair_straight_centerline_reset_distance = 1.0
        # Randomize the straight-stair starting lane while keeping enough
        # margin to the fixed +/-1 m terrain-center reset boundary.
        stair_straight_spawn_y_jitter = 0.70

    class commands:
        curriculum = False
        # Both stair groups remain forward-only. The 0.10 command-diverse
        # group additionally samples yaw from the normal command range.
        stair_lin_vel_x = [0.35, 0.80]
        stair_ang_vel_yaw = [0.00, 0.00]
        stair_full_command_lin_vel_x = [0.10, 0.80]
        stair_full_command_ang_vel_yaw = [-0.80, 0.80]
        stair_full_command_standstill_probability = 0.10
        full_command_standstill_probability = 0.05
        full_command_in_place_turn_probability = 0.05
        full_command_in_place_turn_min_abs_yaw = 0.30
        stair_command_standstill_probability = 0.10
        stair_command_in_place_turn_probability = 0.00
        stair_command_in_place_turn_min_abs_yaw = 0.30
        smooth_max_lin_vel_x = 2.0
        smooth_max_lin_vel_y = 0.0
        non_smooth_max_lin_vel_x = 1.2
        non_smooth_max_lin_vel_y = 0.0
        max_ang_vel_yaw = 3.0
        curriculum_threshold = 0.75
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error, only work on adaptive group
        min_norm = 0.1

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]  # no lateral translation command
            # lin_vel_x = [-1.7, 1.7]  # min max [m/s]
            ang_vel_yaw = [-0.8, 0.8]  # min max [rad/s]
            heading = [-3.14159, 3.14159]

    # Compatibility schema required by the shared BaseTask. WheelFoot never
    # resamples it, steps contact targets, observes it, or attaches rewards to it.
    class gait:
        num_gait_params = 4
        resampling_time = 5  # time before command are changed[s]

        class ranges:
            frequencies = [1.5, 2.5]
            offsets = [0, 1]  # offset is hard to learn
            # durations = [0.3, 0.8]  # small durations(<0.4) is hard to learn
            # frequencies = [2, 2]
            # offsets = [0.5, 0.5]
            durations = [0.5, 0.5]
            swing_height = [0.0, 0.1]

    class init_state:
        # Keep the reset height consistent with the crouched nominal posture
        # below (the bent legs raise the wheel center by about 7.7 cm relative
        # to the all-zero leg pose).
        pos = [0.0, 0.0, 0.8 + 0.0898]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # Strictly mirrored zero-action posture.  The right-leg signs follow
        # the mirrored MuJoCo joint axes, so both wheels have the same height.
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.12,
            "knee_L_Joint": 0.635,
            "foot_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": -0.12,
            "knee_R_Joint": -0.635,
            "foot_R_Joint": 0.0,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        }

    class control:
        action_scale_pos = 0.25
        action_scale_vel = 0.5
        control_type = "P"
        stiffness = {
            "abad_L_Joint": 42,
            "hip_L_Joint": 42,
            "knee_L_Joint": 42,
            "abad_R_Joint": 42,
            "hip_R_Joint": 42,
            "knee_R_Joint": 42,
            "wheel_L_Joint": 0.0,
            "wheel_R_Joint": 0.0,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 2.5,
            "hip_L_Joint": 2.5,
            "knee_L_Joint": 2.5,
            "abad_R_Joint": 2.5,
            "hip_R_Joint": 2.5,
            "knee_R_Joint": 2.5,
            "wheel_L_Joint": 1.2,
            "wheel_R_Joint": 1.2,
        }  # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        user_torque_limit = 80.0
        max_power = 1000.0  # [W]

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/WF_TRON1A/urdf/robot.urdf"
        name = "wheelfoot_flat"
        foot_name = "wheel"
        foot_radius = 0.127
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
        # Domain-randomization ranges from Table III. Length values are stored
        # in SI units here, so the COM ranges in centimetres are converted to m.
        randomize_friction = True
        friction_range = [0.20, 1.60]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-0.5, 2.0]
        randomize_base_com = True
        rand_com_vec = [0.03, 0.02, 0.03]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        push_robots = True
        push_interval_s = 8
        rand_force = False
        force_resampling_time_s = 15
        max_force = 50.0
        rand_force_curriculum_level = 0
        randomize_Kp = True
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = True
        randomize_Kd_range = [0.8, 1.2]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        randomize_imu_offset = True
        randomize_imu_offset_range = [-1.2, 1.2]
        delay_ms_range = [0, 20]
        max_push_vel_xy = 1.0

    class rewards:
        class scales:
            # termination related rewards
            keep_balance = 1.2

            # tracking related rewards
            tracking_lin_vel = 4.0
            tracking_ang_vel = 2.0
            tracking_lin_vel_pb = 1.0
            tracking_ang_vel_pb = 0.2

            # regulation related rewards
            nominal_foot_position = 4.0
            nominal_joint_posture = 1.0
            leg_symmetry = 0.5
            # Allow the two wheels to negotiate adjacent stair treads instead
            # of forcing them to remain perfectly aligned.
            same_foot_x_position = -1.0
            same_foot_z_position = 0.0
            lin_vel_z = -0.3
            ang_vel_xy = -0.3
            torques = -0.00016
            dof_acc = -1.5e-7
            action_rate = -0.03
            dof_pos_limits = -2.0
            collision = -50
            action_smooth = -0.03
            orientation = -8.0
            feet_distance = -20.0
            base_height = -8.0
            feet_contact_forces = -0.01
            wheel_slip = -1.0
            scene6_centerline = -2.0
            scene6_heading = -2.0
            stair_rollback = -10.0
            wheel_axis_reversal = -0.2
            stair_stall = -2.0
            command_reverse_motion = -4.0
            reflection_feet_air_time = 2.0
            reflection_landing_forward_progress = 2.0
            reflection_contact_number = 2.0
            reflection_feet_clearance = 2.0
            reflection_joint_ratio = -0.2
            reflection_load_transfer = 1.0

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_reward = 100
        clip_single_reward = 5
        # Scene 6 has open stair sides and can otherwise produce a large
        # one-step sum while falling. Clip only its final per-step reward;
        # ordinary flat/up-stair rewards and every individual reward formula
        # remain unchanged.
        scene6_total_reward_clip = 2.0
        tracking_sigma = 0.2  # tracking reward = exp(-error^2/sigma)
        ang_tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        nominal_foot_position_tracking_sigma = 0.005
        nominal_foot_position_tracking_sigma_wrt_v = 0.5
        # Symmetric crouched reference used for both flat and stair motion.
        nominal_joint_posture_target = [
            0.0, 0.12, 0.635,
            0.0, -0.12, -0.635,
        ]
        nominal_joint_posture_sigma = 0.15  # [rad]
        leg_symmetry_tracking_sigma = 0.001
        foot_x_position_sigma = 0.001
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.62 + 0.1664
        stair_horizontal_force_threshold = 20.0  # [N], elevated-impact detector
        # Estimated horizontal-force application point must be this far above
        # the wheel bottom, filtering horizontal tread-friction forces.
        stair_contact_height_filter = 0.01  # [m]
        stair_contact_probe_margin = 0.01  # [m], probe just beyond wheel rim
        stair_impact_rollback_hold_time = 0.30  # [s]
        stair_rollback_distance_threshold = 0.03  # [m], cumulative retreat
        stair_rollback_latched_penalty = 0.5  # persistent raw penalty after latch
        wheel_contact_on_force = 5.0  # [N], enter contact state
        wheel_contact_off_force = 1.0  # [N], leave contact state
        # Reward-only lift reflection. These signals never enter Actor/Critic
        # observations and only gate the three reflection rewards above.
        reflection_min_forward_command = 0.10  # [m/s]
        reflection_block_force_threshold = 20.0  # [N], opposing forward motion
        reflection_confirm_frames = 3  # strictly consecutive policy frames
        reflection_timeout = 0.80  # [s], prevents a latched reward window
        reflection_cooldown = 0.15  # [s], reject immediate re-trigger chatter
        reflection_airborne_height = 0.01  # [m], true wheel-bottom clearance
        reflection_air_time_cap = 0.10  # [s]
        reflection_landing_min_forward_progress = 0.08  # [m]
        reflection_landing_full_forward_progress = 0.15  # [m], reward cap
        reflection_landing_min_height_gain = 0.02  # [m], reject same-level hops
        reflection_landing_min_vertical_force = 20.0  # [N], tread support
        reflection_landing_max_vertical_speed = 0.25  # [m/s]
        reflection_landing_confirm_frames = 3  # strictly consecutive
        reflection_contact_mismatch_penalty = 1.30
        reflection_clearance_min = 0.05  # [m]
        reflection_clearance_max = 0.35  # [m]
        reflection_clearance_resolution = 0.01  # [m], downward quantization
        reflection_clearance_margin = 0.02  # [m], clearance above riser
        reflection_clearance_sigma_low = 0.02  # [m], undershoot falls faster
        reflection_clearance_sigma_high = 0.04  # [m], allow safe overshoot
        reflection_support_load_ratio_min = 0.60  # level 0 / 0 cm stair
        reflection_support_load_ratio_max = 0.70  # level 10 / 30 cm stair
        reflection_support_load_ratio_fallback = 0.65
        reflection_load_transfer_orientation_sigma = 0.10
        reflection_nominal_foot_min_scale = 0.25
        reflection_nominal_foot_relax_time = 0.10  # [s], entry and recovery
        wheel_action_rate_multiplier = 1.25
        stair_stall_grace_time = 3.0  # [s], no penalty before continuous stall
        stair_stall_ramp_time = 2.0  # [s], linearly reach full penalty
        # Allow slow stair negotiation, but require at least 1 cm/s of average
        # net progress; forward/backward rocking cannot satisfy this threshold.
        stair_stall_speed_threshold = 0.01  # [m/s] along command direction
        command_reverse_speed_tolerance = 0.05  # [m/s], ignore velocity noise
        wheel_axis_reversal_speed_tolerance = 0.05  # [m/s]
        wheel_axis_reversal_max_speed_change = 2.0  # [m/s], bound one wheel
        wheel_slip_tolerance = 0.05  # [m/s], ignore contact/velocity noise
        stair_straight_centerline_deadband = 0.05  # [m], ordinary straight stairs
        scene6_centerline_deadband = 0.15  # [m], Scene 6 only
        scene6_heading_deadband = 0.0872665  # [rad], 5 degrees
        scene6_heading_normalization = 0.50  # [rad], unit heading error
        scene6_direction_bias_time_constant = 2.0  # [s], long-term EMA
        scene6_direction_bias_threshold = 0.10  # normalized signed bias
        scene6_direction_bias_max_multiplier = 2.0
        # Additional shaping only for the straight-command half of ordinary
        # up stairs. Scene 6 retains the base -2.0 reward scales.
        stair_straight_centerline_penalty_multiplier = 1.00
        stair_straight_heading_penalty_multiplier = 1.30
        min_feet_distance = 0.32
        max_feet_distance = 0.35
        # The 22.27 kg robot statically loads each wheel by about 109 N; keep
        # normal support unpenalized and suppress only hard stair impacts.
        max_contact_force = 180.0

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            height_measurements = 5.0
            contact_forces = 0.01
            torque = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 0.25  # light-noise stair specialization stage

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
            num_threads = 0
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


class BipedCfgPPOWF(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class MLP_Encoder:
        num_input_dim = BipedCfgWF.env.num_observations * BipedCfgWF.env.obs_history_length
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
        learning_rate = 2.0e-4  # conservative plane -> S21 stair fine-tuning
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
        max_iterations = 20000

        # logging
        logger = "tensorboard"
        exptid = ""
        wandb_project = "legged_gym_WF"
        save_interval = 500  # check for potential saves every this many iterations
        experiment_name = "WF_TRON1A"
        run_name = "S2_stair_from_model3000"

        resume = True
        load_run = "Aug11_11-40-32_S1_plane_3000_"
        checkpoint = 3000
        resume_path = "/home/pc/tron1-rl-isaacgym-master/logs/wheelfoot_flat/WF_TRON1A/Aug11_11-40-32_S1_plane_3000_/model_3000.pt"
        # load_run = (
        #     "Aug09_17-54-33_"
        #     "S21_continuous_clearance_from_model54000"
        # )
        # checkpoint = 66500
        # resume_path = (
        #     "/home/pc/wuzeyu128/tongshuo/pointfoot-legged-gym/logs/"
        #     "pointfoot_flat/WF_TRON1A/"
        #     "Aug09_17-54-33_S21_continuous_clearance_from_model54000/"
        #     "model_66500.pt"
        # )
        resume_iteration = 3000   # 66500
        # Keep the checkpoint network/optimizer loading path unchanged.
        zero_critic_height_input_on_resume = False
