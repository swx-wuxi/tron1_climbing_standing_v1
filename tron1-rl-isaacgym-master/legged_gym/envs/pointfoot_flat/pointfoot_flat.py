import torch
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from .pointfoot_flat_config import BipedCfgPF

class BipedPF(BaseTask):
    
    def __init__(
        self, cfg: BipedCfgPF, sim_params, physics_engine, sim_device, headless
    ):
        """Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2

        self.group_idx = torch.arange(0, self.cfg.env.num_envs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def _init_buffers(self):
        super()._init_buffers()
        self.reference_yaw = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.reference_position_xy = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        self.command_segment_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.random_external_forces = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)

        Returns:
            obs (torch.Tensor): Tensor of shape (num_envs, num_observations_per_env)
            rewards (torch.Tensor): Tensor of shape (num_envs)
            dones (torch.Tensor): Tensor of shape (num_envs)
        """
        self._action_clip(actions)
        # step physics and render each frame
        self.render()
        self.pre_physics_step()
        for _ in range(self.cfg.control.decimation):
            self.action_fifo = torch.cat(
                (self.actions.unsqueeze(1), self.action_fifo[:, :-1, :]), dim=1
            )
            self.envs_steps_buf += 1
            self.torques = self._compute_torques(
                self.action_fifo[torch.arange(self.num_envs), self.action_delay_idx, :]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.rand_force:
                self._apply_random_force()
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf # make sure critic_obs update in every for loop
        )

    def _resample_commands(self, env_ids):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 1] = (
            self.command_ranges["lin_vel_y"][env_ids, 1]
            - self.command_ranges["lin_vel_y"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_y"
        ][
            env_ids, 0
        ]
        self.commands[env_ids, 2] = (
            self.command_ranges["ang_vel_yaw"][env_ids, 1]
            - self.command_ranges["ang_vel_yaw"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "ang_vel_yaw"
        ][
            env_ids, 0
        ]
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        smooth_mask = rough_mask = stair_mask = discrete_mask = None
        if hasattr(self, "terrain_types"):
            terrain_choice = (
                self.terrain_types[env_ids].float() / self.cfg.terrain.num_cols
                + 0.001
            )
            terrain_props = self.cfg.terrain.terrain_proportions
            smooth_end = terrain_props[0]
            rough_end = smooth_end + terrain_props[1]
            stair_end = rough_end + terrain_props[2] + terrain_props[3]
            smooth_mask = terrain_choice < smooth_end
            rough_mask = (terrain_choice >= smooth_end) & (terrain_choice < rough_end)
            stair_mask = (terrain_choice >= rough_end) & (terrain_choice < stair_end)
            discrete_mask = terrain_choice >= stair_end

            def resample_command_range(mask, lin_x, lin_y, yaw):
                masked_env_ids = env_ids[mask]
                if len(masked_env_ids) == 0:
                    return
                self.commands[masked_env_ids, 0] = torch_rand_float(
                    lin_x[0], lin_x[1], (len(masked_env_ids), 1), device=self.device
                ).squeeze(1)
                self.commands[masked_env_ids, 1] = torch_rand_float(
                    lin_y[0], lin_y[1], (len(masked_env_ids), 1), device=self.device
                ).squeeze(1)
                self.commands[masked_env_ids, 2] = torch_rand_float(
                    yaw[0], yaw[1], (len(masked_env_ids), 1), device=self.device
                ).squeeze(1)

            resample_command_range(
                rough_mask,
                self.cfg.commands.rough_lin_vel_x,
                self.cfg.commands.rough_lin_vel_y,
                self.cfg.commands.rough_ang_vel_yaw,
            )
            resample_command_range(
                discrete_mask,
                self.cfg.commands.discrete_lin_vel_x,
                self.cfg.commands.discrete_lin_vel_y,
                self.cfg.commands.discrete_ang_vel_yaw,
            )

        # set small commands to zero
        # self.commands[env_ids, :2] *= (
        #     torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_norm
        # ).unsqueeze(1)
        command_mode_rand = torch_rand_float(
            0, 1, (len(env_ids), 1), device=self.device
        ).squeeze(1)
        static_prob = self.cfg.commands.static_command_prob
        yaw_only_prob = self.cfg.commands.yaw_only_command_prob
        straight_prob = self.cfg.commands.straight_command_prob
        zero_command_idx = (
            (command_mode_rand < static_prob).nonzero(as_tuple=False).flatten()
        )
        yaw_only_idx = (
            (
                (command_mode_rand >= static_prob)
                & (command_mode_rand < static_prob + yaw_only_prob)
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        straight_idx = (
            (
                (command_mode_rand >= static_prob + yaw_only_prob)
                & (command_mode_rand < static_prob + yaw_only_prob + straight_prob)
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        zero_command_env_ids = env_ids[zero_command_idx]
        yaw_only_env_ids = env_ids[yaw_only_idx]
        straight_env_ids = env_ids[straight_idx]
        self.commands[zero_command_env_ids, :3] = 0
        self.commands[yaw_only_env_ids, :2] = 0
        self.commands[straight_env_ids, 1:3] = 0
        if len(yaw_only_env_ids) > 0:
            yaw_cmd = self.commands[yaw_only_env_ids, 2]
            min_yaw = self.cfg.rewards.cmd_min_yaw
            yaw_sign = torch.where(yaw_cmd >= 0, 1.0, -1.0)
            self.commands[yaw_only_env_ids, 2] = torch.where(
                torch.abs(yaw_cmd) < min_yaw,
                yaw_sign * min_yaw,
                yaw_cmd,
            )
        if getattr(self.cfg.commands, "stair_only_commands", False):
            stair_env_ids = env_ids
            stair_normal_idx = (
                (command_mode_rand >= static_prob + yaw_only_prob + straight_prob)
                .nonzero(as_tuple=False)
                .flatten()
            )
            stair_normal_env_ids = env_ids[stair_normal_idx]
            stair_straight_idx = straight_idx
            stair_straight_env_ids = straight_env_ids
        elif stair_mask is not None:
            stair_env_ids = env_ids[stair_mask]
            stair_normal_mask = stair_mask & (
                command_mode_rand >= static_prob + yaw_only_prob + straight_prob
            )
            stair_normal_env_ids = env_ids[stair_normal_mask]
            stair_straight_mask = stair_mask & (
                (command_mode_rand >= static_prob + yaw_only_prob)
                & (command_mode_rand < static_prob + yaw_only_prob + straight_prob)
            )
            stair_straight_env_ids = env_ids[stair_straight_mask]
        else:
            stair_env_ids = torch.empty(0, dtype=env_ids.dtype, device=self.device)
            stair_normal_env_ids = stair_env_ids
            stair_straight_env_ids = stair_env_ids

        if len(stair_env_ids) > 0:
            self.commands[stair_env_ids, 1] = 0.0
            self.commands[stair_env_ids, 2] = torch.clamp(
                self.commands[stair_env_ids, 2],
                self.cfg.commands.stair_ang_vel_yaw[0],
                self.cfg.commands.stair_ang_vel_yaw[1],
            )
        if len(stair_normal_env_ids) > 0:
            self.commands[stair_normal_env_ids, 0] = torch_rand_float(
                self.cfg.commands.stair_lin_vel_x[0],
                self.cfg.commands.stair_lin_vel_x[1],
                (len(stair_normal_env_ids), 1),
                device=self.device,
            ).squeeze(1)
        if len(stair_straight_env_ids) > 0:
            self.commands[stair_straight_env_ids, 0] = torch_rand_float(
                self.cfg.commands.stair_lin_vel_x[0],
                self.cfg.commands.stair_lin_vel_x[1],
                (len(stair_straight_env_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.commands[stair_straight_env_ids, 1:3] = 0.0
        if self.cfg.commands.heading_command:
            forward = quat_apply(
                self.base_quat[zero_command_env_ids],
                self.forward_vec[zero_command_env_ids],
            )
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[zero_command_env_ids, 3] = heading
        if hasattr(self, "reference_yaw"):
            self._reset_command_reference_pose(env_ids)

    def post_physics_step(self):
        """Base post-physics flow plus command-reference trajectory update."""
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_position = self.root_states[:, :3]
        self.base_lin_vel = (self.base_position - self.last_base_position) / self.dt
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.base_lin_vel)
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        self.dof_acc = (self.last_dof_vel - self.dof_vel) / self.dt
        self.dof_pos_int += (self.dof_pos - self.raw_default_dof_pos) * self.dt
        self.power = torch.abs(self.torques * self.dof_vel)

        self.compute_foot_state()
        self._post_physics_step_callback()
        self._update_command_reference_pose()

        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_actions[:, :, 1] = self.last_actions[:, :, 0]
        self.last_actions[:, :, 0] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_base_position[:] = self.base_position[:]
        self.last_foot_positions[:] = self.foot_positions[:]

    def _compute_torques(self, actions):
        """Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale

        control_type = self.cfg.control.control_type
        if control_type == "P":
            torques = (
                self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos)
                - self.d_gains * self.dof_vel
            )
        elif control_type == "V":
            torques = (
                self.p_gains * (actions_scaled - self.dof_vel)
                - self.d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(
            torques * self.torques_scale, -self.torque_limits, self.torque_limits
        )

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        # 0:3 base angular velocity
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )

        # 3:6 projected gravity
        noise_vec[3:6] = noise_scales.gravity * noise_level

        # 6:12 joint position
        noise_vec[6:12] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )

        # 12:18 joint velocity
        noise_vec[12:18] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )

        # 18:24 last actions, 24:26 clock sin/cos, 26:30 gait params.
        # These are internally generated quantities, so keep them noise-free.
        noise_vec[18:30] = 0.0
        return noise_vec
    
    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum:
            time_out_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            self.update_command_curriculum(time_out_env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._resample_gaits(env_ids)
        self._reset_command_reference_pose(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_pos[env_ids] = self.dof_pos[env_ids]
        self.last_base_position[env_ids] = self.base_position[env_ids]
        self.last_foot_positions[env_ids] = self.foot_positions[env_ids]
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.envs_steps_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0
        if getattr(self.cfg.gait, "randomize_phase_on_reset", False):
            phase_range = getattr(self.cfg.gait, "reset_phase_range", [0.0, 1.0])
            self.gait_indices[env_ids] = torch_rand_float(
                phase_range[0],
                phase_range[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.gait_indices[env_ids] = 0
        self.clock_inputs_sin[env_ids] = torch.sin(
            2 * np.pi * self.gait_indices[env_ids]
        )
        self.clock_inputs_cos[env_ids] = torch.cos(
            2 * np.pi * self.gait_indices[env_ids]
        )
        obs_buf, _ = self.compute_group_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(1, self.obs_history_length)
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        if hasattr(self, "last_init_xy_offset"):
            self.extras["episode"]["init_xy_offset"] = torch.mean(
                torch.norm(self.last_init_xy_offset[env_ids], dim=1)
            )
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["group_terrain_level"] = torch.mean(
                self.terrain_levels[self.group_idx].float()
            )
            self.extras["episode"]["group_terrain_level_stair_up"] = torch.mean(
                self.terrain_levels[self.stair_up_idx].float()
            )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def compute_group_observations(self):
        # note that observation noise need to modified accordingly !!!
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                self.clock_inputs_sin.view(self.num_envs, 1),
                self.clock_inputs_cos.view(self.num_envs, 1),
                self.gaits,
            ),
            dim=-1,
        )
        measured_heights = self.measured_heights
        if (
            not torch.is_tensor(measured_heights)
            or measured_heights.ndim != 2
            or measured_heights.shape[1] != self.cfg.env.num_height_samples
        ):
            measured_heights = torch.zeros(
                self.num_envs,
                self.cfg.env.num_height_samples,
                device=self.device,
                dtype=obs_buf.dtype,
            )
        height_scan = torch.clip(
            self.root_states[:, 2].unsqueeze(1)
            - self.cfg.rewards.base_height_target
            - measured_heights,
            -1.0,
            1.0,
        ) * self.obs_scales.height_measurements
        foot_contact_forces = (
            self.contact_forces[:, self.feet_indices, :]
            .reshape(self.num_envs, -1)
            * self.obs_scales.contact_forces
        )
        critic_obs_buf = torch.cat(
            (
                obs_buf,
                self.base_lin_vel * self.obs_scales.lin_vel,
                foot_contact_forces,
                height_scan,
            ),
            dim=-1,
        )
        assert obs_buf.shape[1] == self.cfg.env.num_observations
        assert critic_obs_buf.shape[1] == self.cfg.env.num_critic_observations
        return obs_buf, critic_obs_buf

    def _get_path_errors(self):
        """Return signed errors relative to the +x stair centerline."""
        lateral_error = self.base_position[:, 1] - self.env_origins[:, 1]
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading_error = wrap_to_pi(torch.atan2(forward[:, 1], forward[:, 0]))
        return lateral_error, heading_error

    def _reset_command_reference_pose(self, env_ids):
        forward = quat_apply(
            self.root_states[env_ids, 3:7],
            self.forward_vec[env_ids],
        )
        self.reference_yaw[env_ids] = torch.atan2(forward[:, 1], forward[:, 0])
        self.reference_position_xy[env_ids] = self.root_states[env_ids, :2]
        self.command_segment_time[env_ids] = 0.0

    def _update_command_reference_pose(self):
        self.command_segment_time += self.dt
        self.reference_yaw = wrap_to_pi(
            self.reference_yaw + self.commands[:, 2] * self.dt
        )
        ref_forward = torch.stack(
            (torch.cos(self.reference_yaw), torch.sin(self.reference_yaw)),
            dim=1,
        )
        ref_left = torch.stack(
            (-torch.sin(self.reference_yaw), torch.cos(self.reference_yaw)),
            dim=1,
        )
        ref_vel_world = (
            ref_forward * self.commands[:, 0:1]
            + ref_left * self.commands[:, 1:2]
        )
        self.reference_position_xy += ref_vel_world * self.dt

    def _get_feet_positions_body(self):
        """Express both feet in the robot body frame.

        This makes front/back and left/right posture rewards robust to yaw.
        The actor still does not receive foot positions; these are reward-only
        terms computed from simulator state.
        """
        relative_feet = self.foot_positions - self.base_position.unsqueeze(1)
        left_foot_body = quat_rotate_inverse(self.base_quat, relative_feet[:, 0, :])
        right_foot_body = quat_rotate_inverse(self.base_quat, relative_feet[:, 1, :])
        return left_foot_body, right_foot_body

    def _apply_random_force(self):
        """Apply a persistent random force on the base, resampled periodically."""
        force_interval = max(
            1,
            int(self.cfg.domain_rand.force_resampling_time_s / self.sim_params.dt),
        )
        resample_env_ids = (
            ((self.envs_steps_buf <= 1) | (self.envs_steps_buf % force_interval == 0))
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(resample_env_ids) > 0:
            self.random_external_forces[resample_env_ids] = torch_rand_float(
                -self.cfg.domain_rand.max_force,
                self.cfg.domain_rand.max_force,
                (len(resample_env_ids), 3),
                device=self.device,
            )
            self.random_external_forces[resample_env_ids, 2] *= 0.5

        self.rigid_body_external_forces[:] = 0
        self.rigid_body_external_forces[:, 0, 0:3] = quat_rotate(
            self.base_quat, self.random_external_forces
        )
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques),
            gymapi.ENV_SPACE,
        )
    
    # --------------------------- reward functions---------------------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        reward = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        return reward

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1)

    def _reward_action_smooth(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(
                self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.ang_tracking_sigma)

    def _reward_heading_alignment(self):
        """Penalize heading error from the command-segment heading while walking straight."""
        forward = quat_apply(self.base_quat, self.forward_vec)
        base_yaw = torch.atan2(forward[:, 1], forward[:, 0])
        heading_error = wrap_to_pi(self.reference_yaw - base_yaw)

        straight_mask = (
            (torch.abs(self.commands[:, 1]) < self.cfg.rewards.cmd_min_speed)
            & (torch.abs(self.commands[:, 2]) < self.cfg.rewards.cmd_min_yaw)
            & (torch.abs(self.commands[:, 0]) > self.cfg.rewards.cmd_min_speed)
        )

        return straight_mask.float() * torch.square(heading_error)

    def _get_command_world_frame(self):
        """Build command-relative direction in world frame."""
        cmd_xy = self.commands[:, :2]
        cmd_speed = torch.norm(cmd_xy, dim=1)
        valid = (cmd_speed > self.cfg.rewards.cmd_min_speed).float()

        forward_world = quat_apply_yaw(self.base_quat, self.forward_vec)[:, :2]
        forward_world = forward_world / torch.norm(
            forward_world, dim=1, keepdim=True
        ).clamp(min=1e-6)
        left_world = torch.stack((-forward_world[:, 1], forward_world[:, 0]), dim=1)

        desired_vel_world = (
            forward_world * self.commands[:, 0:1]
            + left_world * self.commands[:, 1:2]
        )
        cmd_dir_world = desired_vel_world / cmd_speed.unsqueeze(1).clamp(min=1e-6)
        cmd_lat_world = torch.stack((-cmd_dir_world[:, 1], cmd_dir_world[:, 0]), dim=1)
        return cmd_dir_world, cmd_lat_world, cmd_speed, valid

    def _reward_command_progress(self):
        """Reward progress along the commanded world-frame direction."""
        cmd_dir_world, _, cmd_speed, valid = self._get_command_world_frame()
        planar_vel_world = (
            self.base_position[:, :2] - self.last_base_position[:, :2]
        ) / self.dt
        progress_speed = torch.sum(planar_vel_world * cmd_dir_world, dim=1)
        progress_reward = torch.minimum(
            torch.clamp(progress_speed, min=0.0),
            cmd_speed,
        )
        return valid * progress_reward

    def _reward_command_lateral_drift(self):
        """Penalize velocity perpendicular to the commanded direction."""
        _, cmd_lat_world, _, valid = self._get_command_world_frame()
        planar_vel_world = (
            self.base_position[:, :2] - self.last_base_position[:, :2]
        ) / self.dt
        lateral_speed = torch.sum(planar_vel_world * cmd_lat_world, dim=1)
        return valid * torch.square(lateral_speed)

    def _reward_no_yaw_drift(self):
        """When there is no yaw command, discourage spontaneous yaw motion."""
        no_yaw_cmd = (
            torch.abs(self.commands[:, 2]) < self.cfg.rewards.cmd_min_yaw
        ).float()
        return no_yaw_cmd * torch.square(self.base_ang_vel[:, 2])

    def _reward_command_pose_error(self):
        """Penalize planar position error between reference and actual base pose."""
        pos_error_world = self.base_position[:, :2] - self.reference_position_xy
        pos_error = torch.norm(pos_error_world, dim=1)

        time_scale = torch.clamp(
            self.command_segment_time / self.cfg.rewards.command_pose_grace_time,
            min=1.0,
        )
        time_scale = torch.pow(
            time_scale,
            self.cfg.rewards.command_pose_time_decay_power,
        )
        pos_error = pos_error / time_scale
        pos_error = torch.clamp(
            pos_error,
            max=self.cfg.rewards.command_pose_max_error,
        )
        return torch.square(
            pos_error / self.cfg.rewards.command_pose_position_sigma
        )

    def _reward_expected_position_error(self):
        """Single-step command consistency penalty for all command modes."""
        actual_delta = self.base_position[:, :2] - self.last_base_position[:, :2]
        cmd_dir_world, cmd_lat_world, _, valid = self._get_command_world_frame()
        moving_cmd = valid.float()
        non_moving_cmd = 1.0 - moving_cmd

        lateral_step_err = torch.sum(actual_delta * cmd_lat_world, dim=1)
        lateral_step_err = torch.clamp(
            torch.abs(lateral_step_err)
            - self.cfg.rewards.expected_lateral_deadband,
            min=0.0,
        )
        lateral_penalty = torch.square(
            lateral_step_err / self.cfg.rewards.expected_lateral_sigma
        )

        actual_step_norm = torch.norm(actual_delta, dim=1)
        moving_enough = actual_step_norm > self.cfg.rewards.expected_angle_min_step
        actual_dir = actual_delta / actual_step_norm.unsqueeze(1).clamp(min=1e-6)
        cos_angle = torch.sum(actual_dir * cmd_dir_world, dim=1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle_err = torch.acos(cos_angle)
        angle_err = torch.clamp(
            angle_err - self.cfg.rewards.expected_angle_deadband,
            min=0.0,
        )
        angle_penalty = torch.square(
            angle_err / self.cfg.rewards.expected_angle_sigma
        )
        angle_penalty = angle_penalty * moving_enough.float() * moving_cmd

        static_step_err = torch.clamp(
            actual_step_norm - self.cfg.rewards.expected_lateral_deadband,
            min=0.0,
        )
        static_position_penalty = torch.square(
            static_step_err / self.cfg.rewards.expected_lateral_sigma
        )

        actual_yaw_step = self.base_ang_vel[:, 2] * self.dt
        expected_yaw_step = self.commands[:, 2] * self.dt
        yaw_step_err = actual_yaw_step - expected_yaw_step
        yaw_step_err = torch.clamp(
            torch.abs(yaw_step_err) - self.cfg.rewards.expected_yaw_deadband,
            min=0.0,
        )
        yaw_penalty = torch.square(
            yaw_step_err / self.cfg.rewards.expected_yaw_sigma
        )

        reward = (
            moving_cmd * lateral_penalty
            + non_moving_cmd * static_position_penalty
            + self.cfg.rewards.expected_angle_weight * angle_penalty
            + self.cfg.rewards.expected_yaw_weight * yaw_penalty
        )
        return torch.clamp(reward, max=25.0)

    def _reward_feet_x_separation(self):
        """Penalize one foot being far forward while the other stays far behind."""
        left_foot_body, right_foot_body = self._get_feet_positions_body()
        feet_dx = torch.abs(left_foot_body[:, 0] - right_foot_body[:, 0])

        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        contacts = foot_forces > 1.0
        both_contact = contacts[:, 0] & contacts[:, 1]
        stance_weight = torch.where(
            both_contact,
            torch.ones_like(feet_dx),
            torch.ones_like(feet_dx) * self.cfg.rewards.swing_feet_x_separation_weight,
        )

        excess_dx = torch.clamp(
            feet_dx - self.cfg.rewards.max_feet_x_separation,
            min=0.0,
        )
        return stance_weight * torch.square(
            excess_dx / self.cfg.rewards.max_feet_x_separation
        )

    def _reward_feet_x_body_position(self):
        """Keep each foot from becoming an extreme front probe or rear prop."""
        left_foot_body, right_foot_body = self._get_feet_positions_body()
        foot_x = torch.stack((left_foot_body[:, 0], right_foot_body[:, 0]), dim=1)

        front_excess = torch.clamp(
            foot_x - self.cfg.rewards.max_foot_x_forward,
            min=0.0,
        )
        rear_excess = torch.clamp(
            -foot_x - self.cfg.rewards.max_foot_x_backward,
            min=0.0,
        )
        front_penalty = torch.square(front_excess / self.cfg.rewards.max_foot_x_forward)
        rear_penalty = torch.square(rear_excess / self.cfg.rewards.max_foot_x_backward)
        return torch.sum(front_penalty + rear_penalty, dim=1)

    def _reward_feet_y_distance(self):
        """Prevent the two feet from becoming too narrow or crossing laterally."""
        left_foot_body, right_foot_body = self._get_feet_positions_body()
        feet_dy = torch.abs(left_foot_body[:, 1] - right_foot_body[:, 1])
        too_narrow = torch.clamp(
            self.cfg.rewards.min_feet_y_distance - feet_dy,
            min=0.0,
        )
        return torch.square(too_narrow / self.cfg.rewards.min_feet_y_distance)

    def _reward_feet_y_symmetry(self):
        """Encourage left/right feet to stay mirrored about the body centerline."""
        left_foot_body, right_foot_body = self._get_feet_positions_body()
        center_y = 0.5 * (left_foot_body[:, 1] + right_foot_body[:, 1])
        return torch.square(center_y / self.cfg.rewards.feet_y_symmetry_sigma)

    def _reward_human_gait_mirror(self):
        """Encourage small alternating sagittal steps without long probing.

        This term is gated mainly by sagittal commands, so it does not fight
        pure lateral motion.  It complements the full-command progress reward
        rather than replacing it.
        """
        left_foot_body, right_foot_body = self._get_feet_positions_body()

        left_x = left_foot_body[:, 0]
        right_x = right_foot_body[:, 0]
        dx = left_x - right_x
        abs_dx = torch.abs(dx)

        sagittal_cmd = self.commands[:, 0]
        command_gate = (torch.abs(sagittal_cmd) > 0.05).float()

        nominal_freq = 0.5 * (
            self.cfg.gait.ranges.frequencies[0]
            + self.cfg.gait.ranges.frequencies[1]
        )
        target_step_length = torch.clamp(
            torch.abs(sagittal_cmd) / (2.0 * nominal_freq),
            min=self.cfg.rewards.human_step_length_min,
            max=self.cfg.rewards.human_step_length_max,
        )

        x_center = 0.5 * (left_x + right_x)
        center_penalty = torch.square(
            x_center / self.cfg.rewards.human_gait_center_sigma
        )

        step_length_penalty = torch.square(
            (abs_dx - target_step_length) / self.cfg.rewards.human_step_length_sigma
        )

        swing_weight = torch.clamp(1.0 - self.desired_contact_states, 0.0, 1.0)
        foot_z_vel = self.foot_velocities[:, :, 2]
        late_swing = (
            (foot_z_vel < -0.02)
            | (
                (self.foot_heights < self.cfg.rewards.human_gait_late_swing_height)
                & (foot_z_vel < 0.02)
            )
        ).float()

        left_swing_more = swing_weight[:, 0] > swing_weight[:, 1]
        forward_sign = torch.where(
            sagittal_cmd >= 0.0,
            torch.ones_like(dx),
            -torch.ones_like(dx),
        )
        phase_sign = torch.where(
            left_swing_more,
            torch.ones_like(dx),
            -torch.ones_like(dx),
        ) * forward_sign

        swing_asymmetry = torch.abs(swing_weight[:, 0] - swing_weight[:, 1])
        selected_late_swing = torch.where(
            left_swing_more,
            late_swing[:, 0],
            late_swing[:, 1],
        )

        signed_target_dx = phase_sign * target_step_length
        phase_penalty = torch.square(
            (dx - signed_target_dx) / self.cfg.rewards.human_gait_phase_sigma
        )

        phase_gate = swing_asymmetry * selected_late_swing

        penalty = (
            self.cfg.rewards.human_gait_center_weight * center_penalty
            + self.cfg.rewards.human_gait_step_length_weight * step_length_penalty
            + self.cfg.rewards.human_gait_phase_weight * phase_gate * phase_penalty
        )

        return command_gate * penalty

    def _reward_swing_foot_clearance(self):
        swing_progress, in_swing = self._get_swing_progress()
        swing_mask = (1.0 - self.desired_contact_states) * in_swing.float()
        peak_height = self._get_adaptive_clearance_peaks()
        phase_profile = torch.sin(torch.pi * swing_progress).clamp(min=0.0)
        phase_profile = torch.pow(
            phase_profile, self.cfg.rewards.feet_height_phase_power
        )
        target_height = peak_height * phase_profile
        clearance_low_error = torch.clamp(
            target_height - self.foot_heights,
            min=0.0,
        )
        clearance_upper = peak_height + self.cfg.rewards.feet_height_upper_margin
        clearance_high_error = torch.clamp(
            self.foot_heights - clearance_upper,
            min=0.0,
        )
        clearance = torch.exp(
            -torch.square(clearance_low_error)
            / self.cfg.rewards.feet_clearance_sigma
        ) * torch.exp(
            -torch.square(clearance_high_error)
            / self.cfg.rewards.feet_clearance_high_sigma
        )
        return torch.sum(swing_mask * clearance, dim=1) / (
            torch.sum(swing_mask, dim=1) + 1e-6
        )

    def _get_swing_progress(self):
        """Return normalized [0, 1] swing progress and a per-foot swing mask."""
        offsets = self.gaits[:, 1]
        foot_phase = torch.remainder(
            torch.stack(
                (self.gait_indices, self.gait_indices + offsets), dim=1
            ),
            1.0,
        )
        stance_duration = self.gaits[:, 2].unsqueeze(1).expand_as(foot_phase)
        in_swing = foot_phase >= stance_duration
        swing_progress = torch.clamp(
            (foot_phase - stance_duration) / (1.0 - stance_duration + 1e-6),
            0.0,
            1.0,
        )
        return swing_progress, in_swing

    def _sample_terrain_heights_xy(self, points_xy):
        """Sample conservative terrain heights at world-frame XY points."""
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(points_xy.shape[:-1], device=self.device)
        if self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = points_xy + self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[..., 0], 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[..., 1], 0, self.height_samples.shape[1] - 2)

        # Taking the maximum makes a stair edge visible before the foot reaches
        # the riser instead of averaging it away.
        heights = torch.maximum(
            self.height_samples[px, py], self.height_samples[px + 1, py]
        )
        heights = torch.maximum(heights, self.height_samples[px, py + 1])
        return heights * self.terrain.cfg.vertical_scale

    def _get_adaptive_clearance_peaks(self):
        """Compute each foot's peak clearance from command-direction terrain samples."""
        current_ground = self._sample_terrain_heights_xy(
            self.foot_positions[:, :, :2]
        )

        cmd_dir_world, _, _, valid = self._get_command_world_frame()
        forward = quat_apply_yaw(self.base_quat, self.forward_vec)[:, :2]
        forward = forward / torch.norm(forward, dim=1, keepdim=True).clamp(min=1e-6)
        sample_dir = torch.where(
            valid.unsqueeze(1) > 0.5,
            cmd_dir_world,
            forward,
        )
        distances = torch.as_tensor(
            self.cfg.rewards.feet_height_lookahead,
            device=self.device,
            dtype=self.foot_positions.dtype,
        )
        sample_points = (
            self.foot_positions[:, :, None, :2]
            + sample_dir[:, None, None, :] * distances[None, None, :, None]
        )
        upcoming_ground = self._sample_terrain_heights_xy(sample_points)
        obstacle_height = torch.clamp(
            torch.max(upcoming_ground, dim=-1).values - current_ground,
            min=0.0,
        )
        adaptive_peak = torch.maximum(
            torch.full_like(obstacle_height, self.cfg.rewards.feet_height_target_flat),
            obstacle_height + self.cfg.rewards.feet_height_safety_margin,
        )
        return torch.clamp(
            adaptive_peak, max=self.cfg.rewards.feet_height_target
        )

    def _reward_swing_foot_forward(self):
        swing_progress, in_swing = self._get_swing_progress()
        swing_mask = (1.0 - self.desired_contact_states) * in_swing.float()
        peak_height = self._get_adaptive_clearance_peaks()
        phase_profile = torch.sin(torch.pi * swing_progress).clamp(min=0.0)
        phase_profile = torch.pow(
            phase_profile, self.cfg.rewards.feet_height_phase_power
        )
        target_height = peak_height * phase_profile
        # Swing motion is rewarded along the commanded world-frame direction.
        # This avoids biasing the policy toward body +x when reverse/lateral
        # commands are sampled.
        clearance_low_error = torch.clamp(
            target_height - self.foot_heights,
            min=0.0,
        )
        clearance_upper = peak_height + self.cfg.rewards.feet_height_upper_margin
        clearance_high_error = torch.clamp(
            self.foot_heights - clearance_upper,
            min=0.0,
        )
        height_gate = torch.exp(
            -torch.square(clearance_low_error)
            / self.cfg.rewards.feet_clearance_sigma
        ) * torch.exp(
            -torch.square(clearance_high_error)
            / self.cfg.rewards.feet_clearance_high_sigma
        )
        cmd_dir_world, _, _, valid = self._get_command_world_frame()
        forward_world = quat_apply_yaw(self.base_quat, self.forward_vec)[:, :2]
        forward_world = forward_world / torch.norm(
            forward_world, dim=1, keepdim=True
        ).clamp(min=1e-6)
        swing_dir = torch.where(valid.unsqueeze(1) > 0.5, cmd_dir_world, forward_world)
        base_planar_vel_world = (
            self.base_position[:, :2] - self.last_base_position[:, :2]
        ) / self.dt
        foot_relative_vel_world = (
            self.foot_velocities[:, :, :2] - base_planar_vel_world.unsqueeze(1)
        )
        swing_vel = torch.sum(
            foot_relative_vel_world * swing_dir.unsqueeze(1),
            dim=2,
        )
        swing_vel = torch.clip(
            swing_vel,
            0.0,
            self.cfg.rewards.swing_forward_vel_target,
        )
        return torch.sum(swing_mask * height_gate * swing_vel, dim=1) / (
            torch.sum(swing_mask, dim=1) + 1e-6
        )

    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        if self.reward_scales["tracking_contacts_shaped_force"] > 0:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * torch.exp(
                    -foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma)
        else:
            for i in range(len(self.feet_indices)):
                reward += (1 - desired_contact[:, i]) * (
                    1 - torch.exp(-foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))

        return reward / len(self.feet_indices)

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocities, dim=-1)
        desired_contact = self.desired_contact_states
        reward = 0
        if self.reward_scales["tracking_contacts_shaped_vel"] > 0:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * torch.exp(
                    -foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma
                )
        else:
            for i in range(len(self.feet_indices)):
                reward += desired_contact[:, i] * (
                    1 - torch.exp(-foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma))
        return reward / len(self.feet_indices)

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1)
        return reward

    def _reward_feet_regulation(self):
        feet_height = self.cfg.rewards.base_height_target * 0.001
        reward = torch.sum(
            torch.exp(-self.foot_heights / feet_height)
            * torch.square(torch.norm(self.foot_velocities[:, :, :2], dim=-1)), dim=1)
        return reward

    def _reward_collision(self):
        return torch.sum(
            torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _reward_foot_landing_vel(self):
        z_vels = self.foot_velocities[:, :, 2]
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        about_to_land = (self.foot_heights < self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward
