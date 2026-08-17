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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys

import isaacgym
from isaacgym.torch_utils import *
from legged_gym.envs import *
from legged_gym.utils import (
    get_args,
    get_load_path,
    export_policy_as_jit,
    export_mlp_as_onnx,
    task_registry,
    Logger,
)

import numpy as np
import torch
import matplotlib.pyplot as plt

def parse_record_frames_flag():
    """Parse frame-recording flags before Isaac Gym parses the remaining CLI."""
    record_frames = False
    enable_flags = ("--record_frames", "--record-frames")
    disable_flags = ("--no_record_frames", "--no-record-frames")

    for flag in enable_flags + disable_flags:
        while flag in sys.argv:
            sys.argv.remove(flag)
            record_frames = flag in enable_flags

    return record_frames


def parse_viewer_flag():
    """Parse viewer override flags before Isaac Gym parses the remaining CLI."""
    viewer = None
    enable_flags = ("--viewer",)
    disable_flags = ("--no_viewer", "--no-viewer")

    for flag in enable_flags + disable_flags:
        while flag in sys.argv:
            sys.argv.remove(flag)
            viewer = flag in enable_flags

    return viewer


def parse_small_terrain_flag():
    """Default wheel playback to the full course; allow a quick small test."""
    small_terrain = False
    enable_flags = ("--small_terrain", "--small-terrain")
    disable_flags = ("--full_terrain", "--full-terrain")

    for flag in enable_flags + disable_flags:
        while flag in sys.argv:
            sys.argv.remove(flag)
            small_terrain = flag in enable_flags

    return small_terrain


def configure_demo_terrain(env_cfg):
    """Build one deterministic level-5 stair column matching training."""
    env_cfg.terrain.num_rows = 11
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = 5
    # WF terrain order: flat, rough, stairs up, stairs down, discrete, Scene 6.
    env_cfg.terrain.terrain_proportions = [
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    env_cfg.terrain.curriculum = True
    difficulty = 5.0 / max(env_cfg.terrain.num_rows - 1, 1)
    min_height, max_height = env_cfg.terrain.stair_height_range
    max_depth, min_depth = env_cfg.terrain.stair_width_range
    stair_height = min_height + difficulty * (max_height - min_height)
    stair_depth = max_depth + difficulty * (min_depth - max_depth)
    print(
        "Using fixed level-5 training stair: "
        f"height={stair_height:.3f} m, depth={stair_depth:.3f} m"
    )


def build_command(robot_type, num_cmd, device):
    """Build a command vector matching the environment command dimension."""
    if robot_type == "WF_TRON1A":
        base_cmd = [0.6, 0.0, 0.0]
    elif robot_type is not None and robot_type.startswith("PF"):
        # Stair demos are more stable at 0.4 m/s than the older 0.5 m/s default.
        base_cmd = [0.4, 0.0, 0.0, 0.0]
    else:
        base_cmd = [0.5, 0.0, 0.0, 0.75, 0.0]

    if len(base_cmd) < num_cmd:
        base_cmd = base_cmd + [0.0] * (num_cmd - len(base_cmd))

    return to_torch(base_cmd[:num_cmd], device=device)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    robot_type = os.getenv("ROBOT_TYPE")
    print(f"ROBOT_TYPE={robot_type}")
    # override some parameters for testing
    env_cfg.env.episode_length_s = 120  # record about 2 minutes
    env_cfg.env.num_envs = 1  # record only one robot

    configure_demo_terrain(env_cfg)
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    # Playback should be deterministic enough for policy and video comparison.
    env_cfg.noise.add_noise = False
    env_cfg.noise.noise_level = 0.0
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_inertia = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.rand_force = False
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.domain_rand.randomize_action_delay = False
    env_cfg.domain_rand.randomize_imu_offset = False

    if robot_type == "SF_TRON1A" and args.load_run is not None:
        log_root = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            args.task,
            train_cfg.runner.experiment_name,
        )
        load_path = get_load_path(
            log_root, load_run=args.load_run, checkpoint=args.checkpoint
        )
        loaded_dict = torch.load(load_path, map_location="cpu")
        actor_input_dim = loaded_dict["model_state_dict"]["actor.0.weight"].shape[1]
        legacy_actor_input_dim = 36 + env_cfg.commands.num_commands + 3
        if actor_input_dim == legacy_actor_input_dim:
            print("Detected legacy SF_TRON1A checkpoint without height observations.")
            env_cfg.env.num_observations = 36
            env_cfg.env.num_critic_observations = 3 + env_cfg.env.num_observations
            env_cfg.terrain.mesh_type = "plane"
            env_cfg.terrain.measure_heights = False
            env_cfg.terrain.critic_measure_heights = False
            train_cfg.MLP_Encoder.num_input_dim = (
                env_cfg.env.num_observations * env_cfg.env.obs_history_length
            )

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if not args.headless and env.viewer is None:
        display = os.getenv("DISPLAY", "<unset>")
        raise RuntimeError(
            "Isaac Gym viewer creation failed. DISPLAY="
            f"{display}. Enable MobaXterm X11 forwarding (or connect with "
            "ssh -Y), verify `glxinfo -B`, then rerun with --viewer. "
            "Use --no_viewer for a headless policy check."
        )
    env.cfg.terrain.curriculum = False
    env.terrain_levels.fill_(5)
    env.env_origins[:] = env.terrain_origins[
        env.terrain_levels, env.terrain_types
    ]
    env.reset()
    print(f"Current stair level: {int(env.terrain_levels[0].item())}")
    robot_index = 0  # the playback environment contains exactly one robot
    num_cmd = env.commands.shape[1]
    commands_val = build_command(robot_type, num_cmd, env.device)
    print(f"Command dim: env.commands={num_cmd}, commands_val={commands_val.tolist()}")
    env.commands[:, :] = commands_val.view(1, num_cmd).expand_as(env.commands)
    action_scale = (
        getattr(env.cfg.control, "residual_joint_scale", env.cfg.control.action_scale_pos)
        if robot_type == "WF_TRON1A"
        else env.cfg.control.action_scale
    )
    raw_default_dof_pos = getattr(env, "raw_default_dof_pos", None)
    if raw_default_dof_pos is None:
        raw_default_dof_pos = (
            env.default_dof_pos[robot_index]
            if env.default_dof_pos.ndim == 2
            else env.default_dof_pos
        )
    kinematic_joint_ref = getattr(env, "kinematic_joint_ref", None)
    if kinematic_joint_ref is None:
        kinematic_joint_ref = env.default_dof_pos
    obs, obs_history, commands, _ = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = args.load_run
    train_cfg.runner.checkpoint = args.checkpoint
    # train_cfg.runner.checkpoint = -1

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder = ppo_runner.get_inference_encoder(device=env.device)

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        print(
            "num_actor_obs =",
            ppo_runner.alg.actor_critic.num_actor_obs
        )
        path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            args.task,
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print("Exported policy as jit script to: ", path)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor,
            path,
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            path,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )

    logger = Logger(env.dt)
    joint_index = 1  # which joint is used for logging
    stop_state_log = -1  # disable state plotting/logging during video recording
    stop_rew_log = (
        env.max_episode_length + 1
    )  # number of steps before print average episode rewards
    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1.0, 1.0, 0.0])
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    est = None
    frames_dir = None
    if RECORD_FRAMES:
        if env.viewer is None:
            raise RuntimeError(
                "Frame recording requires a viewer. Remove --headless/--no_viewer, "
                "or run with --no_record_frames for a headless check."
            )
        frames_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            args.task,
            train_cfg.runner.experiment_name,
            args.load_run if args.load_run is not None else "play_record",
            "exported",
            "frames",
        )
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Frame recording enabled. Saving frames to: {frames_dir}")
    else:
        if args.headless:
            print("Frame recording disabled. Running headless; no PNG files will be written.")
        else:
            print("Frame recording disabled. Viewer only; no PNG files will be written.")

    for i in range(int(env.max_episode_length)):
        # Write the fixed demo command before inference. Otherwise a scheduled
        # environment resample can make the Actor see a different command for
        # one policy step even though the simulator is overwritten afterward.
        env.commands[:, :] = commands_val.view(1, num_cmd).expand_as(env.commands)
        commands = env.commands[:, :3] * env.commands_scale
        est = encoder(obs_history)
        actions = policy(torch.cat((est, obs, commands), dim=-1).detach())

        num_cmd = env.commands.shape[1]
        if commands_val.shape[0] != num_cmd:
            commands_val = build_command(robot_type, num_cmd, env.device)
        obs, rews, dones, infos, obs_history, commands, _ = env.step(
            actions.detach()
        )
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(frames_dir, f"{img_idx:06d}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_offset = np.array(env_cfg.viewer.pos)
            target_position = np.array(
                env.base_position[robot_index, :].to(device="cpu")
            )
            target_position[2] = 0
            camera_position = target_position + camera_offset
            env.set_camera(camera_position, target_position)

        if i < stop_state_log:
            logger.log_states(
                {
                    "dof_pos_target": (
                        kinematic_joint_ref[robot_index, joint_index].item()
                        + actions[robot_index, joint_index].item() * action_scale
                        - raw_default_dof_pos[joint_index].item()
                    ) if robot_type == "WF_TRON1A" else actions[robot_index, joint_index].item() * action_scale,
                    "dof_pos": (
                        env.dof_pos[robot_index, joint_index]
                        - raw_default_dof_pos[joint_index]
                    ).item(),
                    "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                    "dof_torque": env.torques[robot_index, joint_index].item(),
                    "command_x": env.commands[robot_index, 0].item(),
                    "command_y": env.commands[robot_index, 1].item(),
                    "command_yaw": env.commands[robot_index, 2].item(),
                    "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                    "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                    "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                    "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                    "power": torch.sum(env.power[robot_index, :]).item(),
                    "contact_forces_z": env.contact_forces[
                        robot_index, env.feet_indices, 2
                    ]
                    .cpu()
                    .numpy(),
                }
            )
            # print(torch.sum(env.power[robot_index, :]).item())
            if est != None:
                logger.log_states(
                    {
                        "est_lin_vel_x": est[robot_index, 0].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_y": est[robot_index, 1].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                        "est_lin_vel_z": est[robot_index, 2].item()
                        / env.cfg.normalization.obs_scales.lin_vel,
                    }
                )
        elif i == stop_state_log:
            logger.plot_states()

        if 0 < i < stop_rew_log:
            if infos.get("episode"):
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = parse_record_frames_flag()
    VIEWER_OVERRIDE = parse_viewer_flag()
    SMALL_TERRAIN = parse_small_terrain_flag()
    MOVE_CAMERA = True
    args = get_args()
    if VIEWER_OVERRIDE is not None:
        args.headless = not VIEWER_OVERRIDE
    elif not RECORD_FRAMES:
        args.headless = False
        print("No frame recording requested; forcing --headless to avoid viewer/OpenGL crashes.")
    play(args)
