from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from .wheelfoot_flat_config import BipedCfgWF
from legged_gym.utils.helpers import class_to_dict

class BipedWF(BaseTask):
    def __init__(
        self, cfg: BipedCfgWF, sim_params, physics_engine, sim_device, headless
    ):
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
        # Start training from the same fully initialized state used after every
        # normal reset. Without this, actors begin at raw terrain origins and
        # Scene 6 has no valid spawn reference during the first rollout.
        self.reset()
        # The runner intentionally randomizes startup episode ages. Keep that
        # first partial episode out of curriculum decisions until its first
        # genuine reset, while retaining the desynchronization benefit.
        self.curriculum_episode_valid[:] = False

    def _get_env_origins(self):
        """Classify the six WF terrain categories, including Scene 6."""
        super()._get_env_origins()
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.smooth_slope_idx = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self.rough_slope_idx = empty
            self.stair_up_idx = empty
            self.stair_down_idx = empty
            self.discrete_idx = empty
            self.stair_course_idx = empty
            self.none_smooth_idx = empty
            return

        proportions = torch.tensor(
            np.cumsum(self.cfg.terrain.terrain_proportions),
            dtype=torch.float,
            device=self.device,
        )
        if len(proportions) != 6:
            raise ValueError(
                "WF terrain requires six proportions: flat, rough, stairs "
                "up, stairs down, discrete, and Scene 6"
            )
        terrain_choice = (
            self.terrain_types.float() / self.cfg.terrain.num_cols + 0.001
        )

        def category_ids(lower, upper):
            return (
                ((terrain_choice >= lower) & (terrain_choice < upper))
                .nonzero(as_tuple=False)
                .flatten()
            )

        zero = torch.zeros((), device=self.device)
        self.smooth_slope_idx = category_ids(zero, proportions[0])
        self.rough_slope_idx = category_ids(proportions[0], proportions[1])
        self.stair_up_idx = category_ids(proportions[1], proportions[2])
        self.stair_down_idx = category_ids(proportions[2], proportions[3])
        self.discrete_idx = category_ids(proportions[3], proportions[4])
        self.stair_course_idx = category_ids(
            proportions[4], proportions[5] + 1.0e-6
        )
        self.none_smooth_idx = torch.cat(
            (
                self.rough_slope_idx,
                self.stair_up_idx,
                self.stair_down_idx,
                self.discrete_idx,
                self.stair_course_idx,
            )
        )
    def _reset_root_states(self, env_ids):
        """Generate all terrain-specific spawn states before one simulator write."""
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]

        if self.custom_origins:
            self.root_states[env_ids, :2] += torch_rand_float(
                -0.7, 0.7, (len(env_ids), 2), device=self.device
            )

            if hasattr(self, "stair_up_straight_env_mask"):
                straight_ids = env_ids[
                    self.stair_up_straight_env_mask[env_ids]
                ]
                if len(straight_ids) > 0:
                    self.root_states[straight_ids, 0] = (
                        self.base_init_state[0]
                        + self.env_origins[straight_ids, 0]
                        + torch_rand_float(
                            -0.2,
                            0.2,
                            (len(straight_ids), 1),
                            device=self.device,
                        ).squeeze(1)
                    )
                    self.root_states[straight_ids, 1] = (
                        self.base_init_state[1]
                        + self.env_origins[straight_ids, 1]
                        + torch_rand_float(
                            -self.cfg.terrain.stair_straight_spawn_y_jitter,
                            self.cfg.terrain.stair_straight_spawn_y_jitter,
                            (len(straight_ids), 1),
                            device=self.device,
                        ).squeeze(1)
                    )

            if hasattr(self, "stair_course_env_mask"):
                scene6_ids = env_ids[self.stair_course_env_mask[env_ids]]
                if len(scene6_ids) > 0:
                    jitter_x, jitter_y = (
                        self.cfg.terrain.scene6_spawn_xy_jitter
                    )
                    self.root_states[scene6_ids, :2] = (
                        self.base_init_state[:2]
                        + self.env_origins[scene6_ids, :2]
                    )
                    self.root_states[scene6_ids, 0] += torch_rand_float(
                        -jitter_x,
                        jitter_x,
                        (len(scene6_ids), 1),
                        device=self.device,
                    ).squeeze(1)
                    self.root_states[scene6_ids, 1] += torch_rand_float(
                        -jitter_y,
                        jitter_y,
                        (len(scene6_ids), 1),
                        device=self.device,
                    ).squeeze(1)

        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.1, 0.1, (len(env_ids), 6), device=self.device
        )
        # Start every episode facing its configured heading without an
        # artificial yaw-rate impulse. Normal commanded turning remains active.
        self.root_states[env_ids, 12] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _get_straight_stair_path_progress(self, env_ids):
        """Convert net +X progress to distance along the nominal stair slope."""
        terrain_difficulty = self.terrain_levels[env_ids].float() / max(
            self.cfg.terrain.num_rows - 1, 1
        )
        terrain_difficulty = torch.clamp(
            terrain_difficulty, min=0.0, max=1.0
        )
        max_stair_depth, min_stair_depth = (
            self.cfg.terrain.stair_width_range
        )
        min_stair_height, max_stair_height = (
            self.cfg.terrain.stair_height_range
        )
        stair_depth = (
            max_stair_depth
            + terrain_difficulty * (min_stair_depth - max_stair_depth)
        )
        stair_height = (
            min_stair_height
            + terrain_difficulty * (max_stair_height - min_stair_height)
        )
        slope_distance_scale = torch.sqrt(
            torch.square(stair_depth) + torch.square(stair_height)
        ) / torch.clamp(stair_depth, min=1.0e-6)
        net_forward_x = (
            self.root_states[env_ids, 0]
            - self.straight_stair_spawn_x[env_ids]
        )
        first_riser_distance = torch.clamp(
            self.env_origins[env_ids, 0]
            + 0.5 * self.cfg.terrain.stair_platform_size
            - self.straight_stair_spawn_x[env_ids],
            min=0.0,
        )
        positive_forward_x = torch.clamp(net_forward_x, min=0.0)
        flat_approach_progress = torch.minimum(
            positive_forward_x, first_riser_distance
        )
        stair_horizontal_progress = torch.clamp(
            positive_forward_x - first_riser_distance, min=0.0
        )
        forward_path_progress = (
            flat_approach_progress
            + stair_horizontal_progress * slope_distance_scale
        )
        # Preserve signed backward progress without applying an uphill scale.
        return torch.where(
            net_forward_x >= 0.0,
            forward_path_progress,
            net_forward_x,
        )

    def _get_straight_stair_climbed_steps(self, env_ids):
        """Infer reached stair index after excluding the flat approach."""
        terrain_difficulty = self.terrain_levels[env_ids].float() / max(
            self.cfg.terrain.num_rows - 1, 1
        )
        terrain_difficulty = torch.clamp(
            terrain_difficulty, min=0.0, max=1.0
        )
        max_stair_depth, min_stair_depth = (
            self.cfg.terrain.stair_width_range
        )
        min_stair_height, max_stair_height = (
            self.cfg.terrain.stair_height_range
        )
        stair_depth = (
            max_stair_depth
            + terrain_difficulty * (min_stair_depth - max_stair_depth)
        )
        stair_height = (
            min_stair_height
            + terrain_difficulty * (max_stair_height - min_stair_height)
        )
        stair_path_step = torch.sqrt(
            torch.square(stair_depth) + torch.square(stair_height)
        )
        first_riser_distance = torch.clamp(
            self.env_origins[env_ids, 0]
            + 0.5 * self.cfg.terrain.stair_platform_size
            - self.straight_stair_spawn_x[env_ids],
            min=0.0,
        )
        stair_only_progress = torch.clamp(
            self._get_straight_stair_path_progress(env_ids)
            - first_riser_distance,
            min=0.0,
        )
        climbed_steps = torch.floor(
            stair_only_progress / torch.clamp(stair_path_step, min=1.0e-6)
        )
        climbed_steps += (stair_only_progress > 1.0e-6).to(
            climbed_steps.dtype
        )

        forward_stair_run = max(
            0.5
            * (
                self.cfg.terrain.terrain_length
                - self.cfg.terrain.stair_platform_size
            ),
            0.0,
        )
        total_steps = torch.ceil(
            torch.full_like(stair_depth, forward_stair_run)
            / torch.clamp(stair_depth, min=1.0e-6)
        )
        return torch.minimum(climbed_steps, total_steps), total_steps

    def _update_terrain_curriculum(self, env_ids):
        """Promote only sustained forward success; demote explicit failures."""
        if not self.init_done or len(env_ids) == 0:
            return

        episode_steps = self.episode_length_buf[env_ids]
        forward_progress = (
            self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
        )
        physical_failure = self.fail_buf[env_ids] > 0
        scene6_failure = self.scene6_failure_buf[env_ids]
        valid_curriculum_episode = self.curriculum_episode_valid[env_ids]
        stair_full_command_mask = (
            self.stair_up_full_command_env_mask[env_ids]
        )

        tracking_failure = torch.zeros_like(physical_failure)
        full_command_tracking_success = torch.zeros_like(physical_failure)
        tracking_name = None
        if "tracking_lin_vel" in self.episode_sums:
            tracking_name = "tracking_lin_vel"
        elif "tracking_lin_vel_x" in self.episode_sums:
            tracking_name = "tracking_lin_vel_x"
        if tracking_name is not None:
            theoretical_tracking = (
                episode_steps.to(self.episode_sums[tracking_name].dtype)
                * self.reward_scales[tracking_name]
            )
            tracking_failure = (
                self.episode_sums[tracking_name][env_ids]
                < 0.5 * theoretical_tracking
            )
            full_command_tracking_success = (
                self.episode_sums[tracking_name][env_ids]
                >= (
                    self.cfg.terrain.stair_full_command_upgrade_tracking_ratio
                    * theoretical_tracking
                )
            )

        angular_tracking_failure = torch.zeros_like(physical_failure)
        if "tracking_ang_vel" in self.episode_sums:
            theoretical_angular_tracking = (
                episode_steps.to(
                    self.episode_sums["tracking_ang_vel"].dtype
                )
                * self.reward_scales["tracking_ang_vel"]
            )
            angular_tracking_failure = (
                self.episode_sums["tracking_ang_vel"][env_ids]
                < (
                    self.cfg.terrain.stair_full_command_downgrade_tracking_ratio
                    * theoretical_angular_tracking
                )
            )
            full_command_tracking_success &= (
                self.episode_sums["tracking_ang_vel"][env_ids]
                >= (
                    self.cfg.terrain.stair_full_command_upgrade_tracking_ratio
                    * theoretical_angular_tracking
                )
            )
        else:
            full_command_tracking_success[:] = False

        move_down = (
            physical_failure | scene6_failure | tracking_failure
        ) & valid_curriculum_episode
        move_down |= (
            stair_full_command_mask
            & angular_tracking_failure
            & valid_curriculum_episode
        )
        survived_long_enough = (
            episode_steps
            >= int(self.cfg.terrain.terrain_upgrade_min_steps)
        ) & valid_curriculum_episode
        common_move_up = (
            survived_long_enough
            & (
                forward_progress
                >= self.cfg.terrain.terrain_upgrade_forward_distance
            )
            & (~move_down)
        )
        straight_stair_mask = self.stair_up_straight_env_mask[env_ids]
        straight_climbed_steps, straight_total_steps = (
            self._get_straight_stair_climbed_steps(env_ids)
        )
        straight_required_steps = torch.minimum(
            torch.full_like(
                straight_total_steps,
                self.cfg.terrain.stair_straight_upgrade_max_steps,
            ),
            (
                self.cfg.terrain.stair_straight_upgrade_total_step_ratio
                * straight_total_steps
            ),
        )
        straight_move_up = (
            survived_long_enough
            & (straight_climbed_steps >= straight_required_steps)
            & (~move_down)
        )
        # Apply the stair-capability target only to ordinary straight stairs.
        # The flat approach is excluded, and commanded speed/distance no longer
        # changes the promotion threshold.
        common_move_up = torch.where(
            straight_stair_mask,
            straight_move_up,
            common_move_up,
        )
        # Scene 6 retains the 920-step + 4 m rule and additionally promotes
        # after a separately confirmed completion of the full descent.
        scene6_mask = self.stair_course_env_mask[env_ids]
        scene6_descent_success = (
            scene6_mask
            & self.scene6_descent_complete_latched[env_ids]
        )
        # A physically stable full-course completion is definitive success.
        # Do not let the aggregate tracking threshold simultaneously demote it.
        move_down &= ~scene6_descent_success
        scene6_move_up = (
            (
                survived_long_enough
                & self.scene6_upgrade_latched[env_ids]
            )
            | scene6_descent_success
        ) & (~move_down)
        move_up = torch.where(
            scene6_mask,
            scene6_move_up,
            common_move_up,
        )
        full_command_move_up = (
            survived_long_enough
            & full_command_tracking_success
            & (~move_down)
        )
        move_up = torch.where(
            stair_full_command_mask,
            full_command_move_up,
            move_up,
        )

        self.success_ids = env_ids[move_up]
        self.fail_ids = env_ids[move_down]
        self.terrain_levels[env_ids] += (
            move_up.to(self.terrain_levels.dtype)
            - move_down.to(self.terrain_levels.dtype)
        )
        self.terrain_levels[env_ids] = torch.clamp(
            self.terrain_levels[env_ids],
            min=0,
            max=self.max_terrain_level - 1,
        )
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def reset_idx(self, env_ids):
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
        self.scene6_spawn_x[env_ids] = self.root_states[env_ids, 0]
        self.straight_stair_spawn_x[env_ids] = self.root_states[env_ids, 0]
        self.straight_stair_spawn_y[env_ids] = self.root_states[env_ids, 1]
        # The runner randomizes the initial episode age only once at startup.
        # After this first real reset, future curriculum decisions use complete
        # episodes whose age genuinely starts from zero.
        self.curriculum_episode_valid[env_ids] = True
        self._resample_commands(env_ids, episode_reset=True)

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
        self.fail_buf[env_ids] = 0
        self.action_fifo[env_ids] = 0
        self.dof_pos_int[env_ids] = 0
        self.wheel_contact[env_ids] = False
        self.reflection_active[env_ids] = False
        self.reflection_reward_active[env_ids] = False
        self.reflection_swing_leg[env_ids] = -1
        self.reflection_block_counter[env_ids] = 0
        self.reflection_timer[env_ids] = 0
        self.reflection_cooldown[env_ids] = 0
        self.reflection_airborne_reached[env_ids] = False
        self.reflection_air_time[env_ids] = 0.0
        self.reflection_landing_counter[env_ids] = 0
        self.reflection_landing_event[env_ids] = False
        self.reflection_landing_forward_progress[env_ids] = 0.0
        self.reflection_nominal_foot_scale[env_ids] = 1.0
        self.reflection_start_position[env_ids] = 0.0
        self.reflection_forward_xy[env_ids] = 0.0
        self.last_wheel_surface_speed[env_ids] = 0.0
        self.wheel_axis_reversal_error[env_ids] = 0.0
        self.stair_progress_deficit[env_ids] = 0.0
        self.stair_impact_rollback_counter[env_ids] = 0
        self.stair_rollback_distance[env_ids] = 0.0
        self.stair_rollback_latched[env_ids] = False
        self.scene6_entered_stairs[env_ids] = False
        self.scene6_failure_buf[env_ids] = False
        self.scene6_success_buf[env_ids] = False
        self.scene6_upgrade_latched[env_ids] = False
        self.scene6_descent_complete_counter[env_ids] = 0
        self.scene6_descent_complete_latched[env_ids] = False
        self.scene6_centerline_bias[env_ids] = 0.0
        self.scene6_heading_bias[env_ids] = 0.0
        self.obs_history[env_ids] = 0
        obs_buf, _ = self.compute_group_observations()
        self.obs_history[env_ids] = obs_buf[env_ids].repeat(
            1, self.obs_history_length
        )
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids])
                / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["group_terrain_level"] = torch.mean(
                self.terrain_levels[self.group_idx].float()
            )
            if len(self.stair_up_idx) > 0:
                self.extras["episode"]["group_terrain_level_stair_up"] = (
                    torch.mean(self.terrain_levels[self.stair_up_idx].float())
                )
                straight_stair_levels = self.terrain_levels[
                    self.stair_up_straight_env_mask
                ]
                full_command_stair_levels = self.terrain_levels[
                    self.stair_up_full_command_env_mask
                ]
                if len(straight_stair_levels) > 0:
                    self.extras["episode"][
                        "group_terrain_level_stair_up_straight"
                    ] = torch.mean(straight_stair_levels.float())
                if len(full_command_stair_levels) > 0:
                    self.extras["episode"][
                        "group_terrain_level_stair_up_full_command"
                    ] = torch.mean(full_command_stair_levels.float())
            if len(self.stair_course_idx) > 0:
                self.extras["episode"]["scene6_current_level"] = torch.mean(
                    self.terrain_levels[self.stair_course_idx].float()
                )
        if self.cfg.terrain.curriculum and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.mean(
                self.command_ranges["lin_vel_x"][self.smooth_slope_idx, 1].float()
            )
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf | self.edge_reset_buf

    def step(self, actions):
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
                self.action_fifo[
                    torch.arange(self.num_envs, device=self.device),
                    self.action_delay_idx,
                    :,
                ]
            ).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            if self.cfg.domain_rand.push_robots:
                self._push_robots()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.compute_dof_vel()
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.nan_to_num(
            self.obs_buf, nan=0.0, posinf=clip_obs, neginf=-clip_obs
        )
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        self.critic_obs_buf = torch.nan_to_num(
            self.critic_obs_buf,
            nan=0.0,
            posinf=clip_obs,
            neginf=-clip_obs,
        )
        self.critic_obs_buf = torch.clip(
            self.critic_obs_buf, -clip_obs, clip_obs
        )
        self.critic_obs_buf[:, 3 : 3 + self.num_obs] = self.obs_buf
        self.obs_history = torch.nan_to_num(
            self.obs_history,
            nan=0.0,
            posinf=clip_obs,
            neginf=-clip_obs,
        )
        self.obs_history = torch.clip(
            self.obs_history, -clip_obs, clip_obs
        )
        self.obs_history[:, -self.num_obs :] = self.obs_buf
        return (
            self.obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,
            self.commands[:, :3] * self.commands_scale,
            self.critic_obs_buf # make sure critic_obs update in every for loop
        )

    def pre_physics_step(self):
        """Store unrelaxed tracking potential for PB reward differencing."""
        self.full_command_backward_blocked_prev = (
            self._get_full_command_backward_stair_blocked()
        )
        self.rwd_linVelTrackPrev = self._tracking_lin_vel_potential()
        self.rwd_angVelTrackPrev = self._reward_tracking_ang_vel()
        
    def _action_clip(self, actions):
        self.actions = actions
        
    def _compute_torques(self, actions):
        pos_action = (
            torch.cat(
                (
                    actions[:, 0:3], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                    actions[:, 4:7], torch.zeros_like(actions[:, 0]).view(self.num_envs, 1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_pos
        )
        vel_action = (
            torch.cat(
                (
                    torch.zeros_like(actions[:, 0:3]), actions[:, 3].view(self.num_envs, 1),
                    torch.zeros_like(actions[:, 0:3]), actions[:, 7].view(self.num_envs, 1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_vel
        )
        # pd controller
        torques = self.p_gains * (pos_action + self.default_dof_pos - self.dof_pos) + self.d_gains * (vel_action - self.dof_vel)
        torques = torch.clip(torques, -self.torque_limits, self.torque_limits )  # torque limit is lower than the torque-requiring lower bound
        return torques * self.torques_scale #notice that even send torque at torque limit , real motor may generate bigger torque that limit!!!!!!!!!!

    def post_physics_step(self):
        super().post_physics_step()
        self.wheel_lin_vel = self.foot_velocities[:, 0, :] + self.foot_velocities[:, 1, :]

    def _push_robots(self):
        """Apply random pushes to all enabled training environments."""
        env_ids = (
            (
                self.envs_steps_buf
                % int(
                    self.cfg.domain_rand.push_interval_s
                    / self.sim_params.dt
                )
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        if len(env_ids) == 0:
            return

        max_push_force = (
            self.base_mass.mean().item()
            * self.cfg.domain_rand.max_push_vel_xy
            / self.sim_params.dt
        )
        self.rigid_body_external_forces[:] = 0
        rigid_body_external_forces = torch_rand_float(
            -max_push_force,
            max_push_force,
            (self.num_envs, 3),
            device=self.device,
        )
        self.rigid_body_external_forces[env_ids, 0, 0:3] = quat_rotate(
            self.base_quat[env_ids],
            rigid_body_external_forces[env_ids],
        )
        self.rigid_body_external_forces[env_ids, 0, 2] *= 0.5

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_external_forces),
            gymtorch.unwrap_tensor(self.rigid_body_external_torques),
            gymapi.ENV_SPACE,
        )

    def check_termination(self):
        super().check_termination()

        straight_stair_centerline_failure = (
            self.stair_up_straight_env_mask
            & (
                torch.abs(
                    self.base_position[:, 1] - self.env_origins[:, 1]
                )
                > self.cfg.terrain.stair_straight_centerline_reset_distance
            )
        )
        # This is an immediate physical failure, not a timeout. It therefore
        # enters the existing straight-stair curriculum demotion path.
        self.time_out_buf[straight_stair_centerline_failure] = False
        self.edge_reset_buf[straight_stair_centerline_failure] = False
        self.reset_buf |= straight_stair_centerline_failure
        self.fail_buf[straight_stair_centerline_failure] = int(
            self.cfg.env.fail_to_terminal_time_s / self.dt
        ) + 1

        if (
            self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]
            or not getattr(self.cfg.terrain, "scene6_enabled", False)
        ):
            return

        scene6_mask = self.stair_course_env_mask
        scene6_forward_progress = self.base_position[:, 0] - self.scene6_spawn_x
        scene6_forward_from_origin = (
            self.base_position[:, 0] - self.env_origins[:, 0]
        )
        scene6_timeout_steps = int(
            round(self.cfg.terrain.scene6_episode_length_s / self.dt)
        )
        scene6_time_out = scene6_mask & (
            self.episode_length_buf > scene6_timeout_steps
        )

        self.scene6_entered_stairs |= (
            scene6_mask
            & (
                scene6_forward_from_origin
                >= self.cfg.terrain.scene6_spawn_length
            )
        )
        base_outside_stair_width = (
            torch.abs(self.base_position[:, 1] - self.env_origins[:, 1])
            > 0.5 * self.cfg.terrain.scene6_stair_width
        )
        wheel_safe_half_width = max(
            0.5 * self.cfg.terrain.scene6_stair_width
            - self.cfg.terrain.scene6_wheel_edge_margin,
            0.0,
        )
        wheel_lateral_offset = torch.abs(
            self.foot_positions[:, :, 1]
            - self.env_origins[:, 1].unsqueeze(1)
        )
        any_wheel_at_stair_edge = torch.any(
            wheel_lateral_offset >= wheel_safe_half_width,
            dim=1,
        )
        scene6_side_fall = (
            scene6_mask
            & self.scene6_entered_stairs
            & (self.terrain_levels > 0)
            & (any_wheel_at_stair_edge | base_outside_stair_width)
        )

        # Use exactly the same accumulated contact/tilt failure as ordinary
        # up stairs. Leaving the narrow stair after entry is the sole Scene-6
        # specific failure and terminates immediately.
        parent_physical_failure = (
            scene6_mask
            & (
                self.fail_buf
                > self.cfg.env.fail_to_terminal_time_s / self.dt
            )
        )
        scene6_failure = scene6_side_fall | parent_physical_failure
        scene6_success = (
            scene6_mask
            & (
                scene6_forward_progress
                >= self.cfg.terrain.scene6_success_distance
            )
            & (~scene6_failure)
        )

        # Completing the full descent is an additional Scene-6 curriculum
        # success. Require both wheels to reach the trailing ground and remain
        # loaded with low vertical body speed for a short consecutive window.
        descent_finish_distance = (
            self.cfg.terrain.terrain_length
            - self.cfg.terrain.scene6_spawn_edge_margin
            - self.cfg.terrain.scene6_trailing_ground_length
            + self.cfg.terrain.scene6_descent_finish_margin
        )
        wheel_forward_from_origin = (
            self.foot_positions[:, :, 0]
            - self.env_origins[:, 0].unsqueeze(1)
        )
        both_wheels_finished_descent = torch.all(
            wheel_forward_from_origin >= descent_finish_distance,
            dim=1,
        )
        descent_stable_raw = (
            scene6_mask
            & (self.terrain_levels > 0)
            & self.scene6_entered_stairs
            & both_wheels_finished_descent
            & torch.all(self.wheel_contact, dim=1)
            & (self.fail_buf == 0)
            & (
                torch.abs(self.base_lin_vel[:, 2])
                <= self.cfg.terrain.scene6_descent_max_vertical_speed
            )
            & (~scene6_failure)
        )
        self.scene6_descent_complete_counter[:] = torch.where(
            descent_stable_raw,
            self.scene6_descent_complete_counter + 1,
            torch.zeros_like(self.scene6_descent_complete_counter),
        )
        descent_confirm_steps = max(
            1,
            int(round(
                self.cfg.terrain.scene6_descent_confirm_time / self.dt
            )),
        )
        scene6_descent_completed = (
            descent_stable_raw
            & (
                self.scene6_descent_complete_counter
                >= descent_confirm_steps
            )
        )
        self.scene6_descent_complete_latched |= scene6_descent_completed

        self.scene6_failure_buf[:] = scene6_failure
        self.scene6_success_buf[:] = (
            scene6_success | scene6_descent_completed
        )
        self.scene6_upgrade_latched |= scene6_success
        # Preserve the parent's ordinary collision/tilt/edge/time-out reset
        # result instead of replacing it with Scene-6-specific rules.
        self.time_out_buf |= scene6_time_out
        self.reset_buf |= scene6_time_out
        # Treat course completion as a successful timeout so PPO bootstraps the
        # terminal value instead of learning it as a fall/collision penalty.
        self.time_out_buf |= scene6_descent_completed
        self.reset_buf |= scene6_descent_completed
        self.time_out_buf[scene6_side_fall] = False
        self.reset_buf |= scene6_side_fall
        self.fail_buf[scene6_side_fall] = int(
            self.cfg.env.fail_to_terminal_time_s / self.dt
        ) + 1

    def compute_reward(self):
        """Use the common reward formulas, then bound Scene-6 return spikes."""
        super().compute_reward()
        if not getattr(self.cfg.terrain, "scene6_enabled", False):
            return
        reward_limit = float(self.cfg.rewards.scene6_total_reward_clip)
        if reward_limit <= 0.0:
            return
        scene6_mask = self.stair_course_env_mask
        # torch.clamp does not repair NaN values. Keep a single malformed
        # Scene-6 reward from contaminating GAE targets and the critic update.
        scene6_reward = torch.nan_to_num(
            self.rew_buf[scene6_mask],
            nan=-reward_limit,
            posinf=reward_limit,
            neginf=-reward_limit,
        )
        self.rew_buf[scene6_mask] = torch.clamp(
            scene6_reward,
            min=-reward_limit,
            max=reward_limit,
        )
        # A Scene-6 physical/edge failure must be the worst outcome inside the
        # same bounded reward scale, never an attractive early exit.
        scene6_failure = scene6_mask & self.scene6_failure_buf
        self.rew_buf[scene6_failure] = -reward_limit

    def compute_group_observations(self):
        # note that observation noise need to modified accordingly !!!
        dof_list = [0,1,2,4,5,6]
        dof_pos = (self.dof_pos - self.default_dof_pos)[:,dof_list]
        # dof_pos = torch.remainder(dof_pos + self.pi, 2 * self.pi) - self.pi

        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                dof_pos * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        # Critic-only privileged vision: local terrain heights relative to the
        # nominal base height. Actor observations above remain unchanged.
        critic_height_image = torch.clip(
            self.root_states[:, 2].unsqueeze(1)
            - self.measured_heights
            - self.cfg.rewards.base_height_target,
            -1.0,
            1.0,
        ) * self.obs_scales.height_measurements
        critic_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                obs_buf,
                critic_height_image,
            ),
            dim=-1,
        )
        return obs_buf, critic_obs_buf
    
    def _post_physics_step_callback(self):
        """Callback called before computing terminations, rewards, and observations
        Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        scheduled_env_ids = (
            (
                self.episode_length_buf
                % int(self.cfg.commands.resampling_time / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .flatten()
        )
        env_ids = self._resample_commands(
            scheduled_env_ids, episode_reset=False
        )
        # A progress deficit belongs to the command under which it was
        # accumulated. Do not carry it into a newly sampled command.
        self.stair_progress_deficit[env_ids] = 0.0
        self.stair_impact_rollback_counter[env_ids] = 0
        self.stair_rollback_distance[env_ids] = 0.0
        self.stair_rollback_latched[env_ids] = False

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = 0.1 * wrap_to_pi(self.commands[:, 3] - heading)
        self.commands[self.stair_course_env_mask, 1:3] = 0.0
        self._update_scene6_directional_bias()

        if self.cfg.terrain.measure_heights or self.cfg.terrain.critic_measure_heights:
            self.measured_heights = self._get_heights()

        self.base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        self._update_wheel_contact_and_stair_impact_state()
        self._update_wheel_axis_reversal_state()
        self._update_stair_rollback_state()
        self._update_stair_stall_state()

    def _get_horizontal_contact_point_height(self, horizontal_force):
        """Estimate horizontal-force application height above the wheel bottom.

        Isaac Gym's GPU contact tensor exposes the net force on a rigid body,
        but not individual contact-point positions. Preserve fully vectorized
        training by probing the terrain immediately beyond the wheel rim on
        the side opposite the net force (the side touching an obstacle). For a
        flat-tread friction force this surface remains at wheel-bottom height;
        for a stair-riser force it lies on the raised step. Clipping the probed
        surface to the wheel center approximates the point on a vertical face.
        """
        if (
            self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]
            or self.height_samples is None
        ):
            return torch.zeros_like(horizontal_force[:, :, 0])

        force_magnitude = torch.norm(horizontal_force, dim=-1, keepdim=True)
        force_direction = horizontal_force / torch.clamp(
            force_magnitude, min=1.0e-6
        )
        probe_distance = (
            self.cfg.asset.foot_radius
            + self.cfg.rewards.stair_contact_probe_margin
        )
        # Contact force points away from the obstacle, so the contact side is
        # opposite the force direction.
        probe_xy = (
            self.foot_positions[:, :, :2]
            - force_direction * probe_distance
            + self.terrain.cfg.border_size
        )
        probe_indices = (
            probe_xy / self.terrain.cfg.horizontal_scale
        ).long()
        px = torch.clip(
            probe_indices[:, :, 0], 0, self.height_samples.shape[0] - 2
        )
        py = torch.clip(
            probe_indices[:, :, 1], 0, self.height_samples.shape[1] - 2
        )
        # Use the highest adjacent cell at a discontinuity so a probe on the
        # vertical boundary observes the raised tread rather than the floor.
        terrain_height = torch.maximum(
            torch.maximum(
                self.height_samples[px, py],
                self.height_samples[px + 1, py],
            ),
            torch.maximum(
                self.height_samples[px, py + 1],
                self.height_samples[px + 1, py + 1],
            ),
        ) * self.terrain.cfg.vertical_scale
        wheel_center_height = self.foot_positions[:, :, 2]
        wheel_bottom_height = (
            wheel_center_height
            - self.cfg.asset.foot_radius
            - self.foot_heights
        )
        estimated_contact_height = torch.minimum(
            terrain_height, wheel_center_height
        )
        height_above_bottom = torch.clamp(
            estimated_contact_height - wheel_bottom_height,
            min=0.0,
            max=2.0 * self.cfg.asset.foot_radius,
        )
        return torch.where(
            force_magnitude.squeeze(-1) > 1.0e-6,
            height_above_bottom,
            torch.zeros_like(height_above_bottom),
        )

    def _update_wheel_contact_and_stair_impact_state(self):
        """Update generic wheel contact and elevated-impact rollback hold."""
        wheel_contact_force = torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=-1
        )
        self.wheel_contact[:] = torch.where(
            self.wheel_contact,
            wheel_contact_force > self.cfg.rewards.wheel_contact_off_force,
            wheel_contact_force > self.cfg.rewards.wheel_contact_on_force,
        )
        self.stair_impact_rollback_counter[:] = torch.clamp(
            self.stair_impact_rollback_counter - 1, min=0
        )

        horizontal_force = self.contact_forces[:, self.feet_indices, :2]
        horizontal_contact_force = torch.norm(horizontal_force, dim=-1)
        contact_point_height = self._get_horizontal_contact_point_height(
            horizontal_force
        )
        elevated_impact_per_wheel = (
            (
                horizontal_contact_force
                > self.cfg.rewards.stair_horizontal_force_threshold
            )
            & (
                contact_point_height
                > self.cfg.rewards.stair_contact_height_filter
            )
        )
        impact_rollback_steps = max(
            1,
            int(round(
                self.cfg.rewards.stair_impact_rollback_hold_time / self.dt
            )),
        )
        self.stair_impact_rollback_counter[
            elevated_impact_per_wheel
        ] = impact_rollback_steps
        self._update_reflection_state(
            horizontal_force,
            contact_point_height,
        )

    def _update_reflection_state(
        self, horizontal_force, contact_point_height
    ):
        """Latch a reward-only lift reflex when a wheel hits an obstacle ahead."""
        self.reflection_reward_active[:] = False
        self.reflection_landing_event[:] = False
        self.reflection_landing_forward_progress[:] = 0.0
        self.reflection_cooldown[:] = torch.clamp(
            self.reflection_cooldown - 1, min=0
        )

        forward_world = quat_apply(self.base_quat, self.forward_vec)[:, :2]
        forward_world = forward_world / torch.clamp(
            torch.norm(forward_world, dim=1, keepdim=True), min=1.0e-6
        )
        opposing_force = -torch.sum(
            horizontal_force * forward_world.unsqueeze(1), dim=-1
        )
        forward_command = (
            self.commands[:, 0]
            > self.cfg.rewards.reflection_min_forward_command
        )
        blocked_raw = (
            (
                opposing_force
                > self.cfg.rewards.reflection_block_force_threshold
            )
            & (
                contact_point_height
                > self.cfg.rewards.stair_contact_height_filter
            )
            & forward_command.unsqueeze(1)
            & torch.all(self.wheel_contact, dim=1).unsqueeze(1)
            & (self.fail_buf == 0).unsqueeze(1)
        )

        inactive = (
            (~self.reflection_active)
            & (self.reflection_cooldown == 0)
        )
        self.reflection_block_counter[:] = torch.where(
            blocked_raw & inactive.unsqueeze(1),
            self.reflection_block_counter + 1,
            torch.zeros_like(self.reflection_block_counter),
        )
        confirmed = (
            self.reflection_block_counter
            >= self.cfg.rewards.reflection_confirm_frames
        )
        start = inactive & torch.any(confirmed, dim=1)
        selected_score = torch.where(
            confirmed,
            opposing_force,
            torch.full_like(opposing_force, -1.0e6),
        )
        selected_leg = torch.argmax(selected_score, dim=1)
        self.reflection_active[start] = True
        self.reflection_swing_leg[start] = selected_leg[start]
        self.reflection_timer[start] = 0
        self.reflection_airborne_reached[start] = False
        self.reflection_air_time[start] = 0.0
        self.reflection_landing_counter[start] = 0
        self.reflection_block_counter[start] = 0
        selected_index = selected_leg.unsqueeze(1).unsqueeze(2)
        selected_position = torch.gather(
            self.foot_positions,
            1,
            selected_index.expand(-1, 1, 3),
        ).squeeze(1)
        self.reflection_start_position[start] = selected_position[start]
        self.reflection_forward_xy[start] = forward_world[start]

        active = self.reflection_active
        self.reflection_timer[active] += 1
        swing_index = torch.clamp(
            self.reflection_swing_leg, min=0
        ).unsqueeze(1)
        swing_contact = torch.gather(
            self.wheel_contact, 1, swing_index
        ).squeeze(1)
        swing_height = torch.gather(
            self.foot_heights, 1, swing_index
        ).squeeze(1)
        swing_position = torch.gather(
            self.foot_positions,
            1,
            swing_index.unsqueeze(2).expand(-1, 1, 3),
        ).squeeze(1)
        swing_vertical_force = torch.gather(
            self.contact_forces[:, self.feet_indices, 2],
            1,
            swing_index,
        ).squeeze(1)
        swing_vertical_speed = torch.abs(torch.gather(
            self.foot_velocities[:, :, 2],
            1,
            swing_index,
        ).squeeze(1))
        airborne_now = (
            active
            & (~swing_contact)
            & (
                swing_height
                >= self.cfg.rewards.reflection_airborne_height
            )
        )
        self.reflection_airborne_reached[:] |= airborne_now
        self.reflection_air_time[:] += (
            active & (~swing_contact)
        ).float() * self.dt

        forward_progress = torch.sum(
            (
                swing_position[:, :2]
                - self.reflection_start_position[:, :2]
            )
            * self.reflection_forward_xy,
            dim=1,
        )
        height_gain = (
            swing_position[:, 2]
            - self.reflection_start_position[:, 2]
        )
        landing_candidate = (
            active
            & self.reflection_airborne_reached
            & swing_contact
            & (
                forward_progress
                >= self.cfg.rewards.reflection_landing_min_forward_progress
            )
            & (
                height_gain
                >= self.cfg.rewards.reflection_landing_min_height_gain
            )
            & (
                swing_vertical_force
                >= self.cfg.rewards.reflection_landing_min_vertical_force
            )
            & (
                swing_vertical_speed
                <= self.cfg.rewards.reflection_landing_max_vertical_speed
            )
        )
        self.reflection_landing_counter[:] = torch.where(
            landing_candidate,
            self.reflection_landing_counter + 1,
            torch.zeros_like(self.reflection_landing_counter),
        )
        landing_confirmed = (
            self.reflection_landing_counter
            >= self.cfg.rewards.reflection_landing_confirm_frames
        )
        timeout_steps = max(
            1,
            int(round(
                self.cfg.rewards.reflection_timeout / self.dt
            )),
        )
        timed_out = active & (self.reflection_timer >= timeout_steps)
        command_stopped = active & (~forward_command)
        finish = landing_confirmed | timed_out | command_stopped

        self.reflection_landing_event[:] = landing_confirmed
        self.reflection_landing_forward_progress[landing_confirmed] = (
            forward_progress[landing_confirmed]
        )
        self.reflection_reward_active[:] = (
            active & (~finish) & (~landing_candidate)
        )
        self.reflection_active[finish] = False
        self.reflection_swing_leg[finish] = -1
        self.reflection_timer[finish] = 0
        self.reflection_airborne_reached[finish] = False
        self.reflection_landing_counter[finish] = 0
        cooldown_steps = max(
            1,
            int(round(
                self.cfg.rewards.reflection_cooldown / self.dt
            )),
        )
        self.reflection_cooldown[finish] = cooldown_steps

        # Smoothly relax only the selected swing leg's nominal-foot reward.
        # A per-leg state also gives a smooth recovery after swing_leg resets.
        nominal_scale_target = torch.ones_like(
            self.reflection_nominal_foot_scale
        )
        relax_mask = self.reflection_reward_active
        relax_env_ids = relax_mask.nonzero(as_tuple=False).flatten()
        if len(relax_env_ids) > 0:
            relax_leg_ids = self.reflection_swing_leg[relax_env_ids]
            nominal_scale_target[relax_env_ids, relax_leg_ids] = (
                self.cfg.rewards.reflection_nominal_foot_min_scale
            )
        relax_time = max(
            float(self.cfg.rewards.reflection_nominal_foot_relax_time),
            self.dt,
        )
        relax_alpha = min(self.dt / relax_time, 1.0)
        self.reflection_nominal_foot_scale += relax_alpha * (
            nominal_scale_target - self.reflection_nominal_foot_scale
        )

    def _update_stair_stall_state(self):
        """Accumulate a net-progress deficit under a nonzero stair command.

        Unlike a consecutive low-speed counter, this state is not cleared by a
        brief forward burst. Forward travel repays the deficit in proportion to
        its distance, while stopping and reverse travel add to it. Therefore a
        forward/backward oscillation with little net displacement still reaches
        the stall penalty.
        """
        command_direction = torch.sign(self.commands[:, 0])
        command_active = torch.abs(self.commands[:, 0]) > 0.1
        directed_forward_speed = command_direction * self.base_lin_vel[:, 0]
        active = (
            self.stair_env_mask
            & command_active
            & (~self._get_full_command_backward_stair_blocked())
        )
        speed_threshold = max(
            self.cfg.rewards.stair_stall_speed_threshold, 1.0e-6
        )
        deficit_delta = (
            1.0 - directed_forward_speed / speed_threshold
        ) * self.dt
        max_deficit = (
            self.cfg.rewards.stair_stall_grace_time
            + self.cfg.rewards.stair_stall_ramp_time
        )
        updated_deficit = torch.clamp(
            self.stair_progress_deficit + deficit_delta,
            min=0.0,
            max=max_deficit,
        )
        self.stair_progress_deficit[:] = torch.where(
            active,
            updated_deficit,
            torch.zeros_like(self.stair_progress_deficit),
        )

    def _get_full_command_backward_stair_blocked(self):
        """Allow the full-command stair group to hold after a rearward block."""
        backward_command = self.commands[:, 0] < -0.1
        elevated_stair_contact = torch.any(
            self.stair_impact_rollback_counter > 0, dim=1
        )
        return (
            self.stair_up_full_command_env_mask
            & backward_command
            & elevated_stair_contact
        )

    def _get_stair_rollback_window(self):
        """Return environments inside the generic elevated-impact hold."""
        impact_hold_active = torch.any(
            self.stair_impact_rollback_counter > 0, dim=1
        )
        command_active = torch.abs(self.commands[:, 0]) > 0.1
        return impact_hold_active & command_active

    def _update_stair_rollback_state(self):
        """Latch strong punishment after cumulative reverse travel is excessive."""
        monitoring = self._get_stair_rollback_window()
        command_direction = torch.sign(self.commands[:, 0])
        directed_forward_speed = command_direction * self.base_lin_vel[:, 0]
        backward_distance_step = torch.clip(
            -directed_forward_speed, min=0.0
        ) * self.dt
        accumulated_distance = (
            self.stair_rollback_distance + backward_distance_step
        )
        self.stair_rollback_distance[:] = torch.where(
            monitoring,
            accumulated_distance,
            torch.zeros_like(self.stair_rollback_distance),
        )
        exceeded = self.stair_rollback_distance >= (
            self.cfg.rewards.stair_rollback_distance_threshold
        )
        self.stair_rollback_latched[:] = torch.where(
            monitoring,
            self.stair_rollback_latched | exceeded,
            torch.zeros_like(self.stair_rollback_latched),
        )

    def _update_wheel_axis_reversal_state(self):
        """Latch bounded rolling-direction reversals for the current reward."""
        _, wheel_surface_speed, _ = self._get_wheel_rolling_speeds()
        tolerance = self.cfg.rewards.wheel_axis_reversal_speed_tolerance
        genuine_reversal = (
            (wheel_surface_speed * self.last_wheel_surface_speed < 0.0)
            & (torch.abs(wheel_surface_speed) > tolerance)
            & (torch.abs(self.last_wheel_surface_speed) > tolerance)
            & self.wheel_contact
        )
        speed_change = torch.clamp(
            torch.abs(
                wheel_surface_speed - self.last_wheel_surface_speed
            ),
            max=self.cfg.rewards.wheel_axis_reversal_max_speed_change,
        )
        self.wheel_axis_reversal_error[:] = (
            speed_change * genuine_reversal
        )
        self.last_wheel_surface_speed[:] = wheel_surface_speed

    def _resample_commands(self, env_ids, episode_reset=False):
        """Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return env_ids

        self.commands[env_ids, 0] = (
            self.command_ranges["lin_vel_x"][env_ids, 1]
            - self.command_ranges["lin_vel_x"][env_ids, 0]
        ) * torch.rand(len(env_ids), device=self.device) + self.command_ranges[
            "lin_vel_x"
        ][
            env_ids, 0
        ]
        # Ordinary up stairs and Scene 6 use the same forward-only command
        # distribution. All other terrain types retain the full distribution.
        straight_stair_mask = (
            self.stair_up_straight_env_mask
            | self.stair_course_env_mask
        )
        stair_up_ids = env_ids[straight_stair_mask[env_ids]]
        if len(stair_up_ids) > 0:
            stair_min, stair_max = self.cfg.commands.stair_lin_vel_x
            self.commands[stair_up_ids, 0] = (
                (stair_max - stair_min)
                * torch.rand(len(stair_up_ids), device=self.device)
                + stair_min
            )
        stair_full_command_ids = env_ids[
            self.stair_up_full_command_env_mask[env_ids]
        ]
        if len(stair_full_command_ids) > 0:
            stair_full_min, stair_full_max = (
                self.cfg.commands.stair_full_command_lin_vel_x
            )
            self.commands[stair_full_command_ids, 0] = (
                (stair_full_max - stair_full_min)
                * torch.rand(
                    len(stair_full_command_ids), device=self.device
                )
                + stair_full_min
            )
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
        if len(stair_full_command_ids) > 0:
            stair_full_yaw_min, stair_full_yaw_max = (
                self.cfg.commands.stair_full_command_ang_vel_yaw
            )
            self.commands[stair_full_command_ids, 2] = (
                (stair_full_yaw_max - stair_full_yaw_min)
                * torch.rand(
                    len(stair_full_command_ids), device=self.device
                )
                + stair_full_yaw_min
            )
            # Keep an explicit pure-standstill share in the forward-plus-turn
            # stair group. These environments are excluded from the generic
            # command-mode override below, so handle the zero command here.
            stair_full_standstill_probability = (
                self.cfg.commands.stair_full_command_standstill_probability
            )
            stair_full_standstill_ids = stair_full_command_ids[
                torch.rand(
                    len(stair_full_command_ids), device=self.device
                ) < stair_full_standstill_probability
            ]
            self.commands[stair_full_standstill_ids, 0:3] = 0.0
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["heading"][0],
                self.command_ranges["heading"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        if len(stair_up_ids) > 0:
            stair_yaw_min, stair_yaw_max = self.cfg.commands.stair_ang_vel_yaw
            self.commands[stair_up_ids, 1] = 0.0
            self.commands[stair_up_ids, 2] = (
                (stair_yaw_max - stair_yaw_min)
                * torch.rand(len(stair_up_ids), device=self.device)
                + stair_yaw_min
            )

            # Straight stair training keeps a configurable standstill share;
            # yaw-only samples remain disabled in the current configuration.
            stair_mode = torch.rand(len(stair_up_ids), device=self.device)
            stair_standstill_probability = (
                self.cfg.commands.stair_command_standstill_probability
            )
            stair_turn_probability = (
                self.cfg.commands.stair_command_in_place_turn_probability
            )
            stair_standstill_ids = stair_up_ids[
                stair_mode < stair_standstill_probability
            ]
            stair_in_place_turn_ids = stair_up_ids[
                (stair_mode >= stair_standstill_probability)
                & (
                    stair_mode
                    < stair_standstill_probability + stair_turn_probability
                )
            ]
            self.commands[stair_standstill_ids, 0:3] = 0.0
            if len(stair_in_place_turn_ids) > 0:
                stair_min_abs_yaw = (
                    self.cfg.commands.stair_command_in_place_turn_min_abs_yaw
                )
                stair_max_abs_yaw = max(
                    abs(stair_yaw_min), abs(stair_yaw_max)
                )
                stair_yaw_magnitude = stair_min_abs_yaw + (
                    stair_max_abs_yaw - stair_min_abs_yaw
                ) * torch.rand(
                    len(stair_in_place_turn_ids), device=self.device
                )
                stair_yaw_sign = torch.where(
                    torch.rand(
                        len(stair_in_place_turn_ids), device=self.device
                    )
                    < 0.5,
                    -torch.ones(
                        len(stair_in_place_turn_ids), device=self.device
                    ),
                    torch.ones(
                        len(stair_in_place_turn_ids), device=self.device
                    ),
                )
                self.commands[stair_in_place_turn_ids, 0:2] = 0.0
                self.commands[stair_in_place_turn_ids, 2] = (
                    stair_yaw_sign * stair_yaw_magnitude
                )

        # Smooth/rough slopes, down stairs and discrete obstacles use all
        # commands, including explicit standstill and in-place turning samples.
        # Forward-plus-turn stairs keep nonzero forward motion, so exclude them
        # from these generic mode overrides.
        special_mask = (
            self.stair_up_straight_env_mask
            | self.stair_up_full_command_env_mask
            | self.stair_course_env_mask
        )
        full_command_ids = env_ids[~special_mask[env_ids]]
        if len(full_command_ids) > 0:
            mode = torch.rand(len(full_command_ids), device=self.device)
            standstill_probability = (
                self.cfg.commands.full_command_standstill_probability
            )
            turn_probability = (
                self.cfg.commands.full_command_in_place_turn_probability
            )
            standstill_ids = full_command_ids[mode < standstill_probability]
            in_place_turn_ids = full_command_ids[
                (mode >= standstill_probability)
                & (mode < standstill_probability + turn_probability)
            ]
            self.commands[standstill_ids, 0:3] = 0.0
            if len(in_place_turn_ids) > 0:
                yaw_min, yaw_max = self.cfg.commands.ranges.ang_vel_yaw
                min_abs_yaw = (
                    self.cfg.commands.full_command_in_place_turn_min_abs_yaw
                )
                max_abs_yaw = max(abs(yaw_min), abs(yaw_max))
                yaw_magnitude = min_abs_yaw + (
                    max_abs_yaw - min_abs_yaw
                ) * torch.rand(len(in_place_turn_ids), device=self.device)
                yaw_sign = torch.where(
                    torch.rand(len(in_place_turn_ids), device=self.device) < 0.5,
                    -torch.ones(len(in_place_turn_ids), device=self.device),
                    torch.ones(len(in_place_turn_ids), device=self.device),
                )
                self.commands[in_place_turn_ids, 0:2] = 0.0
                self.commands[in_place_turn_ids, 2] = yaw_sign * yaw_magnitude

        return env_ids
            
    def _get_noise_scale_vec(self, cfg):
        """Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = (
            noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        )
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:12] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[12:20] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[20:28] = 0.0  # previous actions
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        self.measured_heights = torch.zeros(
            self.num_envs,
            self.cfg.env.num_height_samples,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.stair_env_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.stair_up_env_mask = torch.zeros_like(self.stair_env_mask)
        if hasattr(self, "stair_up_idx"):
            self.stair_up_env_mask[self.stair_up_idx] = True
            self.stair_env_mask[self.stair_up_idx] = True
            self.stair_env_mask[self.stair_down_idx] = True
        # Split ordinary up stairs by fixed terrain columns so the configured
        # full-command share is exact and stable across resets.
        terrain_proportions = np.cumsum(
            self.cfg.terrain.terrain_proportions
        )
        stair_share = terrain_proportions[2] - terrain_proportions[1]
        full_command_share = float(
            getattr(
                self.cfg.terrain,
                "stair_full_command_probability",
                0.0,
            )
        )
        if full_command_share < 0.0 or full_command_share > stair_share:
            raise ValueError(
                "stair_full_command_probability must be within the ordinary "
                f"up-stair share [0, {stair_share}], got {full_command_share}"
            )
        if hasattr(self, "terrain_types"):
            terrain_choice = (
                self.terrain_types.float()
                / self.cfg.terrain.num_cols
                + 0.001
            )
        else:
            # Plane environments have no terrain columns. Keep all terrain
            # category masks empty while preserving the common command group.
            terrain_choice = torch.zeros(
                self.num_envs, dtype=torch.float, device=self.device
            )
        full_command_lower = terrain_proportions[2] - full_command_share
        self.stair_up_full_command_env_mask = (
            self.stair_up_env_mask
            & (terrain_choice >= full_command_lower)
        )
        self.stair_up_straight_env_mask = (
            self.stair_up_env_mask
            & ~self.stair_up_full_command_env_mask
        )
        self.stair_course_env_mask = torch.zeros_like(self.stair_env_mask)
        if getattr(self.cfg.terrain, "scene6_enabled", False):
            self.stair_course_env_mask[self.stair_course_idx] = True
            self.stair_env_mask[self.stair_course_idx] = True

        self.wheel_axis_local = torch.zeros_like(self.foot_positions)
        self.wheel_axis_local[:, :, 1] = 1.0
        self.world_up = torch.zeros_like(self.foot_positions)
        self.world_up[:, :, 2] = 1.0
        wheel_buffer_shape = (self.num_envs, len(self.feet_indices))
        self.wheel_contact = torch.zeros(
            wheel_buffer_shape, dtype=torch.bool, device=self.device
        )
        self.reflection_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.reflection_reward_active = torch.zeros_like(
            self.reflection_active
        )
        self.reflection_swing_leg = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.reflection_block_counter = torch.zeros(
            wheel_buffer_shape, dtype=torch.long, device=self.device
        )
        self.reflection_timer = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reflection_cooldown = torch.zeros_like(
            self.reflection_timer
        )
        self.reflection_airborne_reached = torch.zeros_like(
            self.reflection_active
        )
        self.reflection_air_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.reflection_landing_counter = torch.zeros_like(
            self.reflection_timer
        )
        self.reflection_landing_event = torch.zeros_like(
            self.reflection_active
        )
        self.reflection_landing_forward_progress = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.reflection_nominal_foot_scale = torch.ones(
            wheel_buffer_shape, dtype=torch.float, device=self.device
        )
        self.reflection_start_position = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self.reflection_forward_xy = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        self.last_wheel_surface_speed = torch.zeros(
            wheel_buffer_shape, dtype=torch.float, device=self.device
        )
        self.wheel_axis_reversal_error = torch.zeros_like(
            self.last_wheel_surface_speed
        )
        self.stair_progress_deficit = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.stair_impact_rollback_counter = torch.zeros(
            wheel_buffer_shape, dtype=torch.long, device=self.device
        )
        self.stair_rollback_distance = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.stair_rollback_latched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.scene6_entered_stairs = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.scene6_failure_buf = torch.zeros_like(
            self.scene6_entered_stairs
        )
        self.scene6_success_buf = torch.zeros_like(
            self.scene6_entered_stairs
        )
        self.scene6_upgrade_latched = torch.zeros_like(
            self.scene6_entered_stairs
        )
        self.scene6_descent_complete_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.scene6_descent_complete_latched = torch.zeros_like(
            self.scene6_entered_stairs
        )
        # The startup episode may receive a randomized pre-existing age from
        # the runner, so it must not affect terrain promotion or demotion.
        self.curriculum_episode_valid = torch.zeros_like(
            self.scene6_entered_stairs
        )
        # Use the current root X as a safe fallback even before the explicit
        # startup reset; every later reset overwrites it with the actual spawn.
        self.scene6_spawn_x = self.root_states[:, 0].clone()
        self.straight_stair_spawn_x = self.root_states[:, 0].clone()
        # Per-episode reference lane for the ordinary straight-stair reward.
        # Each real reset overwrites it with the actual randomized spawn Y.
        self.straight_stair_spawn_y = self.root_states[:, 1].clone()
        self.scene6_centerline_bias = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )
        self.scene6_heading_bias = torch.zeros_like(
            self.scene6_centerline_bias
        )
        self.wheel_lin_vel = torch.zeros_like(self.foot_velocities)
        self.wheel_ang_vel = torch.zeros_like(self.base_ang_vel)

    # ------------ reward functions----------------

    def _get_straight_stair_signed_centerline_error(self):
        """Normalize lateral drift from the episode's starting lane."""
        terrain_center_offset = (
            self.base_position[:, 1] - self.env_origins[:, 1]
        )
        spawn_lane_offset = (
            self.base_position[:, 1] - self.straight_stair_spawn_y
        )
        signed_offset = torch.where(
            self.stair_up_straight_env_mask,
            spawn_lane_offset,
            terrain_center_offset,
        )
        deadband = torch.where(
            self.stair_up_straight_env_mask,
            torch.full_like(
                signed_offset,
                self.cfg.rewards.stair_straight_centerline_deadband,
            ),
            torch.full_like(
                signed_offset,
                self.cfg.rewards.scene6_centerline_deadband,
            ),
        )
        straight_symmetric_distance = (
            self.cfg.terrain.stair_straight_centerline_reset_distance
            - self.cfg.terrain.stair_straight_spawn_y_jitter
        )
        # Use one fixed scale on both sides of the spawn lane. The scale is the
        # minimum reset clearance over the configured spawn range, so equal
        # left/right drift always receives exactly the same penalty.
        straight_available_distance = torch.full_like(
            signed_offset,
            straight_symmetric_distance,
        )
        scene6_half_width = 0.5 * self.cfg.terrain.scene6_stair_width
        available_distance = torch.where(
            self.stair_up_straight_env_mask,
            straight_available_distance,
            torch.full_like(signed_offset, scene6_half_width),
        )
        usable_distance = torch.clamp(
            available_distance - deadband,
            min=1.0e-6,
        )
        return (
            torch.sign(signed_offset)
            * torch.clip(torch.abs(signed_offset) - deadband, min=0.0)
            / usable_distance
        )

    def _get_scene6_signed_heading_error(self):
        body_forward = quat_apply(self.base_quat, self.forward_vec)
        yaw_error = torch.atan2(body_forward[:, 1], body_forward[:, 0])
        deadband = self.cfg.rewards.scene6_heading_deadband
        return (
            torch.sign(yaw_error)
            * torch.clip(torch.abs(yaw_error) - deadband, min=0.0)
            / max(self.cfg.rewards.scene6_heading_normalization, 1.0e-6)
        )

    def _update_scene6_directional_bias(self):
        """Track persistent signed lateral/yaw errors independently per env."""
        time_constant = max(
            self.cfg.rewards.scene6_direction_bias_time_constant,
            self.dt,
        )
        alpha = min(self.dt / time_constant, 1.0)
        straight_stair_mask = (
            self.stair_up_straight_env_mask
            | self.stair_course_env_mask
        )
        centerline_error = torch.clip(
            self._get_straight_stair_signed_centerline_error(),
            min=-1.0,
            max=1.0,
        )
        heading_error = torch.clip(
            self._get_scene6_signed_heading_error(), min=-1.0, max=1.0
        )
        self.scene6_centerline_bias[:] = torch.where(
            straight_stair_mask,
            (1.0 - alpha) * self.scene6_centerline_bias
            + alpha * centerline_error,
            torch.zeros_like(self.scene6_centerline_bias),
        )
        self.scene6_heading_bias[:] = torch.where(
            straight_stair_mask,
            (1.0 - alpha) * self.scene6_heading_bias + alpha * heading_error,
            torch.zeros_like(self.scene6_heading_bias),
        )

    def _get_scene6_directional_multiplier(self, signed_error, signed_bias):
        threshold = min(
            max(self.cfg.rewards.scene6_direction_bias_threshold, 0.0),
            0.999,
        )
        persistent_strength = torch.clip(
            (torch.abs(signed_bias) - threshold) / (1.0 - threshold),
            min=0.0,
            max=1.0,
        )
        same_direction = (signed_error * signed_bias > 0.0).float()
        max_multiplier = max(
            self.cfg.rewards.scene6_direction_bias_max_multiplier, 1.0
        )
        return 1.0 + (
            max_multiplier - 1.0
        ) * persistent_strength * same_direction

    def _reward_feet_distance(self):
        # Penalize base height away from target
        feet_distance = torch.norm(
            self.foot_positions[:, 0, :2] - self.foot_positions[:, 1, :2], dim=-1
        )
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        return reward

    def _reward_scene6_centerline(self):
        """Penalize lateral drift on straight-climbing stair terrains."""
        signed_error = self._get_straight_stair_signed_centerline_error()
        # The fixed normalization above makes mirrored left/right trajectories
        # identical. Retain signed persistence only to amplify a sustained
        # one-sided drift; mirroring both error and bias leaves this unchanged.
        directional_multiplier = self._get_scene6_directional_multiplier(
            signed_error, self.scene6_centerline_bias
        )
        centerline_penalty = torch.clip(
            torch.square(signed_error) * directional_multiplier,
            min=0.0,
            max=2.0,
        )
        straight_stair_mask = (
            self.stair_up_straight_env_mask
            | self.stair_course_env_mask
        )
        straight_group_multiplier = torch.where(
            self.stair_up_straight_env_mask,
            torch.full_like(
                centerline_penalty,
                self.cfg.rewards.stair_straight_centerline_penalty_multiplier,
            ),
            torch.ones_like(centerline_penalty),
        )
        return (
            centerline_penalty
            * straight_stair_mask
            * straight_group_multiplier
        )

    def _reward_scene6_heading(self):
        """Penalize yaw from world +X on straight-climbing stair terrains."""
        signed_error = self._get_scene6_signed_heading_error()
        directional_multiplier = self._get_scene6_directional_multiplier(
            signed_error, self.scene6_heading_bias
        )
        heading_penalty = torch.clip(
            torch.square(signed_error) * directional_multiplier,
            min=0.0,
            max=2.0,
        )
        straight_stair_mask = (
            self.stair_up_straight_env_mask
            | self.stair_course_env_mask
        )
        straight_group_multiplier = torch.where(
            self.stair_up_straight_env_mask,
            torch.full_like(
                heading_penalty,
                self.cfg.rewards.stair_straight_heading_penalty_multiplier,
            ),
            torch.ones_like(heading_penalty),
        )
        return (
            heading_penalty
            * straight_stair_mask
            * straight_group_multiplier
        )

    def _reward_collision(self):
        return torch.sum(
            torch.norm(
                self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 1.0, dim=1)

    def _reward_feet_contact_forces(self):
        """Penalize only excessive wheel impacts, not normal support forces."""
        contact_force = torch.norm(
            self.contact_forces[:, self.feet_indices, :], dim=-1
        )
        excess_force = torch.clip(
            contact_force - self.cfg.rewards.max_contact_force, min=0.0
        )
        return torch.sum(excess_force, dim=1)

    def _get_wheel_rolling_speeds(self):
        """Return longitudinal center/surface speeds and lateral speed."""
        wheel_axis_world = quat_apply(
            self.foot_quat.reshape(-1, 4),
            self.wheel_axis_local.reshape(-1, 3),
        ).view_as(self.foot_positions)
        rolling_direction = torch.cross(
            wheel_axis_world, self.world_up, dim=-1
        )
        rolling_direction = rolling_direction / torch.clamp(
            torch.norm(rolling_direction, dim=-1, keepdim=True), min=1.0e-6
        )
        wheel_ang_vel_world = self.rigid_body_state[
            :, self.feet_indices, 10:13
        ]
        wheel_angular_speed = torch.sum(
            wheel_ang_vel_world * wheel_axis_world, dim=-1
        )
        wheel_center_speed = torch.sum(
            self.foot_velocities * rolling_direction, dim=-1
        )
        wheel_surface_speed = (
            wheel_angular_speed * self.cfg.asset.foot_radius
        )
        lateral_speed = torch.sum(
            self.foot_velocities * wheel_axis_world, dim=-1
        )
        return wheel_center_speed, wheel_surface_speed, lateral_speed

    def _reward_wheel_slip(self):
        """Penalize slip using the local contact surface on every contact."""
        wheel_center_speed, wheel_surface_speed, lateral_speed = (
            self._get_wheel_rolling_speeds()
        )

        tolerance = self.cfg.rewards.wheel_slip_tolerance
        longitudinal_slip = torch.clip(
            torch.abs(wheel_center_speed - wheel_surface_speed) - tolerance,
            min=0.0,
        )
        lateral_slip = torch.clip(
            torch.abs(lateral_speed) - tolerance, min=0.0
        )
        ground_slip_error = (
            torch.square(longitudinal_slip) + torch.square(lateral_slip)
        )

        # The horizontal-ground relation v_center ~= omega*r is invalid on a
        # stair riser: valid rolling there is tangent to the vertical face.
        # Estimate that contact point from the same terrain probe used by the
        # elevated-impact detector, then penalize velocity tangent to the riser.
        horizontal_force = self.contact_forces[
            :, self.feet_indices, :2
        ]
        horizontal_force_magnitude = torch.norm(
            horizontal_force, dim=-1
        )
        contact_height = self._get_horizontal_contact_point_height(
            horizontal_force
        )
        riser_contact = (
            (
                horizontal_force_magnitude
                >= self.cfg.rewards.stair_horizontal_force_threshold
            )
            & (
                contact_height
                > self.cfg.rewards.stair_contact_height_filter
            )
        )
        force_direction = horizontal_force / torch.clamp(
            horizontal_force_magnitude.unsqueeze(-1), min=1.0e-6
        )
        wheel_radius = self.cfg.asset.foot_radius
        contact_z = torch.clamp(
            contact_height - wheel_radius,
            min=-wheel_radius,
            max=wheel_radius,
        )
        contact_horizontal_radius = torch.sqrt(
            torch.clamp(
                wheel_radius ** 2 - torch.square(contact_z), min=0.0
            )
        )
        contact_offset = torch.cat(
            (
                -force_direction
                * contact_horizontal_radius.unsqueeze(-1),
                contact_z.unsqueeze(-1),
            ),
            dim=-1,
        )
        wheel_angular_velocity = self.rigid_body_state[
            :, self.feet_indices, 10:13
        ]
        contact_velocity = self.foot_velocities + torch.cross(
            wheel_angular_velocity, contact_offset, dim=-1
        )
        riser_normal = torch.cat(
            (
                force_direction,
                torch.zeros_like(contact_z).unsqueeze(-1),
            ),
            dim=-1,
        )
        normal_velocity = torch.sum(
            contact_velocity * riser_normal, dim=-1, keepdim=True
        )
        riser_tangent_speed = torch.norm(
            contact_velocity - normal_velocity * riser_normal, dim=-1
        )
        riser_slip = torch.clip(
            riser_tangent_speed - tolerance, min=0.0
        )
        slip_error = torch.where(
            riser_contact,
            torch.square(riser_slip),
            ground_slip_error,
        )

        # Keep the existing Schmitt-filtered 5 N-on / 1 N-off contact gate.
        return torch.sum(slip_error * self.wheel_contact, dim=1)

    def _reward_wheel_axis_reversal(self):
        """Penalize rapid rolling-direction flips while a wheel is in contact."""
        return torch.sum(self.wheel_axis_reversal_error, dim=1)

    def _reward_stair_rollback(self):
        """Prevent recoil briefly after an elevated stair-riser impact."""
        wheel_center_speed, wheel_surface_speed, _ = (
            self._get_wheel_rolling_speeds()
        )
        command_direction = torch.sign(self.commands[:, 0]).unsqueeze(1)
        command_active = torch.abs(self.commands[:, 0]).unsqueeze(1) > 0.1
        reverse_tolerance = self.cfg.rewards.command_reverse_speed_tolerance
        backward_center_speed = torch.clip(
            -command_direction * wheel_center_speed - reverse_tolerance,
            min=0.0,
        )
        backward_surface_speed = torch.clip(
            -command_direction * wheel_surface_speed - reverse_tolerance,
            min=0.0,
        )
        holding = (
            (self.stair_impact_rollback_counter > 0)
            & command_active
        )
        rollback_error = (
            backward_center_speed
            + torch.square(backward_center_speed)
            + 0.25 * backward_surface_speed
            + 0.25 * torch.square(backward_surface_speed)
        )
        velocity_penalty = torch.sum(rollback_error * holding, dim=1)
        persistent_penalty = (
            self.cfg.rewards.stair_rollback_latched_penalty
        )
        return (
            velocity_penalty + persistent_penalty
        ) * self.stair_rollback_latched

    def _reward_stair_stall(self):
        """Ramp a penalty after the accumulated net-progress grace is spent."""
        ramp_time = max(self.cfg.rewards.stair_stall_ramp_time, self.dt)
        return torch.clip(
            (self.stair_progress_deficit - self.cfg.rewards.stair_stall_grace_time)
            / ramp_time,
            min=0.0,
            max=1.0,
        )

    def _reward_command_reverse_motion(self):
        """Penalize motion opposite to the commanded longitudinal direction."""
        command_direction = torch.sign(self.commands[:, 0])
        command_active = torch.abs(self.commands[:, 0]) > 0.1
        directed_forward_speed = command_direction * self.base_lin_vel[:, 0]
        reverse_speed = torch.clip(
            -directed_forward_speed
            - self.cfg.rewards.command_reverse_speed_tolerance,
            min=0.0,
        )
        return (reverse_speed + torch.square(reverse_speed)) * command_active

    def _reward_reflection_feet_air_time(self):
        """Reward controlled swing duration at the first reflected-leg contact."""
        return (
            torch.clamp(
                self.reflection_air_time,
                min=0.0,
                max=self.cfg.rewards.reflection_air_time_cap,
            )
            * self.reflection_landing_event
        )

    def _reward_reflection_landing_forward_progress(self):
        """Reward 8--15 cm net swing progress only after valid landing."""
        min_progress = float(
            self.cfg.rewards.reflection_landing_min_forward_progress
        )
        full_progress = max(
            float(
                self.cfg.rewards.reflection_landing_full_forward_progress
            ),
            min_progress + 1.0e-6,
        )
        progress_score = torch.clamp(
            (
                self.reflection_landing_forward_progress - min_progress
            )
            / (full_progress - min_progress),
            min=0.0,
            max=1.0,
        )
        return progress_score * self.reflection_landing_event.float()

    def _reward_reflection_contact_number(self):
        """Match one support contact and one airborne swing during reflection."""
        active = self.reflection_reward_active
        swing_index = torch.clamp(
            self.reflection_swing_leg, min=0
        ).unsqueeze(1)
        support_index = 1 - swing_index
        swing_contact = torch.gather(
            self.wheel_contact, 1, swing_index
        ).squeeze(1)
        support_contact = torch.gather(
            self.wheel_contact, 1, support_index
        ).squeeze(1)
        support_term = torch.where(
            support_contact,
            torch.ones_like(self.reflection_air_time),
            torch.full_like(
                self.reflection_air_time,
                -self.cfg.rewards.reflection_contact_mismatch_penalty,
            ),
        )
        swing_term = torch.where(
            ~swing_contact,
            torch.ones_like(self.reflection_air_time),
            torch.full_like(
                self.reflection_air_time,
                -self.cfg.rewards.reflection_contact_mismatch_penalty,
            ),
        )
        return (support_term + swing_term) * active

    def _reward_reflection_feet_clearance(self):
        """Reward terrain-adaptive clearance during reflected swing."""
        swing_index = torch.clamp(
            self.reflection_swing_leg, min=0
        ).unsqueeze(1)
        swing_height = torch.gather(
            self.foot_heights, 1, swing_index
        ).squeeze(1)
        terrain_difficulty = self.terrain_levels.float() / max(
            self.cfg.terrain.num_rows - 1, 1
        )
        terrain_difficulty = torch.clamp(
            terrain_difficulty, min=0.0, max=1.0
        )
        min_stair_height, max_stair_height = (
            self.cfg.terrain.stair_height_range
        )
        stair_height = min_stair_height + terrain_difficulty * (
            max_stair_height - min_stair_height
        )
        clearance_resolution = self.cfg.rewards.reflection_clearance_resolution
        clearance_target = (
            torch.floor(
                torch.clamp(stair_height, min=0.0)
                / max(clearance_resolution, 1.0e-6)
                + 1.0e-6
            )
            * clearance_resolution
            + self.cfg.rewards.reflection_clearance_margin
        )
        clearance_target = torch.clamp(
            clearance_target,
            min=self.cfg.rewards.reflection_clearance_min,
            max=self.cfg.rewards.reflection_clearance_max,
        )
        clearance_error = swing_height - clearance_target
        sigma_low = max(
            float(self.cfg.rewards.reflection_clearance_sigma_low),
            1.0e-6,
        )
        sigma_high = max(
            float(self.cfg.rewards.reflection_clearance_sigma_high),
            1.0e-6,
        )
        clearance_sigma = torch.where(
            clearance_error < 0.0,
            torch.full_like(clearance_error, sigma_low),
            torch.full_like(clearance_error, sigma_high),
        )
        clearance_reward = torch.exp(
            -0.5 * torch.square(clearance_error / clearance_sigma)
        )
        return (
            clearance_reward * self.reflection_reward_active.float()
        )

    def _reward_reflection_joint_ratio(self):
        """Shape the reflected leg toward a 1:2 hip-to-knee action ratio."""
        swing_leg = torch.clamp(self.reflection_swing_leg, min=0)
        env_indices = torch.arange(self.num_envs, device=self.device)
        hip_indices = 1 + 4 * swing_leg
        knee_indices = 2 + 4 * swing_leg
        hip_action = torch.abs(
            self.actions[env_indices, hip_indices]
        )
        knee_action = torch.abs(
            self.actions[env_indices, knee_indices]
        )
        ratio_error = torch.square(knee_action - 2.0 * hip_action)
        return ratio_error * self.reflection_reward_active

    def _reward_reflection_load_transfer(self):
        """Reward stable support-leg loading before the reflected leg lifts."""
        swing_index = torch.clamp(
            self.reflection_swing_leg, min=0
        ).unsqueeze(1)
        support_index = 1 - swing_index

        vertical_load = torch.clamp(
            self.contact_forces[:, self.feet_indices, 2], min=0.0
        )
        support_load = torch.gather(
            vertical_load, 1, support_index
        ).squeeze(1)
        total_load = torch.clamp(
            torch.sum(vertical_load, dim=1), min=1.0e-6
        )
        support_ratio = support_load / total_load

        if hasattr(self, "terrain_levels"):
            terrain_difficulty = torch.clamp(
                self.terrain_levels.float()
                / max(self.cfg.terrain.num_rows - 1, 1),
                min=0.0,
                max=1.0,
            )
        else:
            terrain_difficulty = torch.zeros_like(support_ratio)
        min_target_ratio = max(
            float(self.cfg.rewards.reflection_support_load_ratio_min),
            0.5001,
        )
        max_target_ratio = max(
            float(self.cfg.rewards.reflection_support_load_ratio_max),
            min_target_ratio,
        )
        stair_target_ratio = min_target_ratio + terrain_difficulty * (
            max_target_ratio - min_target_ratio
        )
        target_ratio = torch.where(
            self.stair_env_mask,
            stair_target_ratio,
            torch.full_like(
                stair_target_ratio,
                float(
                    self.cfg.rewards.
                    reflection_support_load_ratio_fallback
                ),
            ),
        )
        transfer_score = torch.clamp(
            (support_ratio - 0.5) / (target_ratio - 0.5),
            min=0.0,
            max=1.0,
        )
        orientation_error = torch.sum(
            torch.square(self.projected_gravity[:, :2]), dim=1
        )
        orientation_score = torch.exp(
            -orientation_error
            / max(
                float(
                    self.cfg.rewards.
                    reflection_load_transfer_orientation_sigma
                ),
                1.0e-6,
            )
        )
        transfer_phase = (
            self.reflection_reward_active
            & (~self.reflection_airborne_reached)
            & torch.all(self.wheel_contact, dim=1)
        )
        return transfer_score * orientation_score * transfer_phase.float()

    def _reward_nominal_foot_position(self):
        #1. calculate foot postion wrt base in base frame  
        nominal_base_height = -(self.cfg.rewards.base_height_target- self.cfg.asset.foot_radius)
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        per_leg_reward = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=self.foot_positions.dtype,
            device=self.device,
        )
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
            height_error = nominal_base_height - foot_positions_base[:, i, 2]
            per_leg_reward[:, i] = torch.exp(
                -(height_error ** 2)
                / self.cfg.rewards.nominal_foot_position_tracking_sigma
            )
        per_leg_reward *= self.reflection_nominal_foot_scale
        reward = torch.mean(per_leg_reward, dim=1)
        vel_cmd_norm = torch.norm(self.commands[:, :3], dim=1)
        return reward * torch.exp(-(vel_cmd_norm ** 2)/self.cfg.rewards.nominal_foot_position_tracking_sigma_wrt_v)

    def _reward_nominal_joint_posture(self):
        """Track the symmetric crouched posture without changing observations."""
        leg_indices = [0, 1, 2, 4, 5, 6]
        target = self.dof_pos.new_tensor(
            self.cfg.rewards.nominal_joint_posture_target
        )
        sigma = max(
            float(self.cfg.rewards.nominal_joint_posture_sigma), 1.0e-6
        )
        posture_error = self.dof_pos[:, leg_indices] - target
        return torch.exp(
            -torch.mean(torch.square(posture_error), dim=1)
            / (sigma * sigma)
        )
    
    def _reward_same_foot_z_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_z_position_err = foot_positions_base[:,0,2] - foot_positions_base[:,1,2]
        return foot_z_position_err ** 2

    def _reward_leg_symmetry(self):
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        leg_symmetry_err = (abs(foot_positions_base[:,0,1])-abs(foot_positions_base[:,1,1]))
        return torch.exp(-(leg_symmetry_err ** 2)/ self.cfg.rewards.leg_symmetry_tracking_sigma)

    def _reward_same_foot_x_position(self):
        reward = 0
        foot_positions_base = self.foot_positions - \
                            (self.base_position).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        foot_x_position_err = foot_positions_base[:,0,0] - foot_positions_base[:,1,0]
        # reward = torch.exp(-(foot_x_position_err ** 2)/ self.cfg.rewards.foot_x_position_sigma)
        reward = torch.abs(foot_x_position_err)
        return reward

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

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square(self.dof_acc), dim=1)

    def _reward_action_rate(self):
        """Penalize first-order action changes."""
        action_delta = self.actions - self.last_actions[:, :, 0]
        action_rate = torch.square(action_delta)
        action_rate[:, [3, 7]] *= (
            self.cfg.rewards.wheel_action_rate_multiplier
        )
        return torch.sum(action_rate, dim=1)

    def _reward_action_smooth(self):
        """Penalize second-order action changes."""
        action_acceleration = (
            self.actions
            - 2 * self.last_actions[:, :, 0]
            + self.last_actions[:, :, 1]
        )
        return torch.sum(torch.square(action_acceleration), dim=1)

    def _reward_keep_balance(self):
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _tracking_lin_vel_potential(self):
        """Return the unrelaxed linear-velocity tracking potential."""
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
            dim=1,
        )
        tracking_potential = torch.exp(
            -lin_vel_error / self.cfg.rewards.tracking_sigma
        )

        # Once a backward-command robot in the full-command stair group is
        # blocked by a riser, holding zero planar velocity is also successful.
        # Preserve the original target as an equally valid option so a robot
        # that can safely continue backward is not discouraged.
        hold_error = torch.sum(
            torch.square(self.base_lin_vel[:, :2]), dim=1
        )
        hold_potential = torch.exp(
            -hold_error / self.cfg.rewards.tracking_sigma
        )
        backward_blocked = (
            self._get_full_command_backward_stair_blocked()
        )
        return torch.where(
            backward_blocked,
            torch.maximum(tracking_potential, hold_potential),
            tracking_potential,
        )

    def _reward_tracking_lin_vel(self):
        return self._tracking_lin_vel_potential()

    def _reward_tracking_lin_vel_pb(self):
        # Difference the unrelaxed potential first, then scale it. This avoids
        # paying an artificial PB reward merely because relaxation changed.
        nonterminal_mask = (self.reset_buf == 0).to(
            self.rwd_linVelTrackPrev.dtype
        )
        backward_blocked = (
            self._get_full_command_backward_stair_blocked()
        )
        block_related_step = (
            backward_blocked | self.full_command_backward_blocked_prev
        )
        nonterminal_mask *= (~block_related_step).to(
            nonterminal_mask.dtype
        )
        delta_phi = nonterminal_mask * (
            self._tracking_lin_vel_potential() - self.rwd_linVelTrackPrev
        )
        return delta_phi / self.dt

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        tracking_potential = torch.exp(
            -ang_vel_error / self.cfg.rewards.ang_tracking_sigma
        )
        hold_potential = torch.exp(
            -torch.square(self.base_ang_vel[:, 2])
            / self.cfg.rewards.ang_tracking_sigma
        )
        backward_blocked = (
            self._get_full_command_backward_stair_blocked()
        )
        return torch.where(
            backward_blocked,
            torch.maximum(tracking_potential, hold_potential),
            tracking_potential,
        )

    def _reward_tracking_ang_vel_pb(self):
        nonterminal_mask = (self.reset_buf == 0).to(
            self.rwd_angVelTrackPrev.dtype
        )
        backward_blocked = (
            self._get_full_command_backward_stair_blocked()
        )
        nonterminal_mask *= (
            ~(backward_blocked | self.full_command_backward_blocked_prev)
        ).to(nonterminal_mask.dtype)
        delta_phi = nonterminal_mask * (
            self._reward_tracking_ang_vel() - self.rwd_angVelTrackPrev
        )
        # return ang_vel_error
        return delta_phi / self.dt
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.abs(base_height - self.cfg.rewards.base_height_target)