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

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils

if not hasattr(np, "float323232"):
    np.float323232 = np.float32

class Terrain:
    def __init__(self, cfg, num_robots) -> None:
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [
            np.sum(cfg.terrain_proportions[: i + 1])
            for i in range(len(cfg.terrain_proportions))
        ]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size / self.cfg.horizontal_scale)
        if self.type == "trimesh":
            self.tot_cols = (
                int(cfg.num_cols * self.width_per_env_pixels) + 4 * self.border
            )
        else:
            self.tot_cols = (
                int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
            )
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.terrain_num = np.zeros(7, dtype=np.int16)
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:
            self.randomized_terrain()

        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            (
                self.vertices,
                self.triangles,
            ) = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.cfg.horizontal_scale,
                self.cfg.vertical_scale,
                self.cfg.slope_treshold,
            )

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / max(self.cfg.num_rows - 1, 1)
                choice = j / self.cfg.num_cols + 0.001
                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop("type")
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain(
                "terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.vertical_scale,
                horizontal_scale=self.horizontal_scale,
            )

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)

    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(
            "terrain",
            width=self.width_per_env_pixels,
            length=self.width_per_env_pixels,
            vertical_scale=self.cfg.vertical_scale,
            horizontal_scale=self.cfg.horizontal_scale,
        )
        slope = difficulty * 0.5
        random_height = 0.05 + difficulty * 0.15
        configured_step_width_range = getattr(
            self.cfg, "stair_width_range", None
        )
        if configured_step_width_range is None:
            default_step_width = getattr(
                self.cfg, "stair_step_width", 0.37
            )
        else:
            max_step_width, min_step_width = configured_step_width_range
            default_step_width = max_step_width + np.clip(
                difficulty, 0.0, 1.0
            ) * (min_step_width - max_step_width)
        max_step_height = getattr(self.cfg, "max_stair_step_height", 0.3)
        configured_step_height_range = getattr(
            self.cfg, "stair_height_range", None
        )
        target_step_height = getattr(self.cfg, "stair_step_height", None)
        if configured_step_height_range is not None:
            min_step_height, max_configured_step_height = (
                configured_step_height_range
            )
            step_height = min_step_height + np.clip(
                difficulty, 0.0, 1.0
            ) * (max_configured_step_height - min_step_height)
        else:
            step_height = (
                target_step_height
                if target_step_height is not None
                else 0.05 + difficulty * 0.23
            )
        step_slope = step_height / default_step_width
        discrete_obstacles_height = 0.05 + difficulty * 0.25
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty == 0 else 0.1
        gap_size = 1.0 * difficulty
        pit_depth = 1.0 * difficulty
        if choice < self.proportions[0]:
            self.terrain_num[0] += 1
            if getattr(self.cfg, "smooth_slope_as_flat", False):
                slope = 0.0
            if choice < self.proportions[0] / 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
        elif choice < self.proportions[1]:
            self.terrain_num[1] += 1
            if (
                choice
                < self.proportions[0] + (self.proportions[1] - self.proportions[0]) / 2
            ):
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(
                terrain, slope=slope, platform_size=3.0
            )
            terrain_utils.random_uniform_terrain(
                terrain,
                min_height=-random_height,
                max_height=random_height,
                step=0.005,
                downsampled_scale=0.2,
            )
        elif choice < self.proportions[3]:
            step_scale = np.array(
                getattr(
                    self.cfg,
                    "stair_step_width_scales",
                    [1, 1.05, 0.95, 1.1, 0.9, 1.2, 0.8],
                )
            )
            if choice < self.proportions[2]:
                self.terrain_num[2] += 1
                step_height *= -1
                step_slope *= -1
                step_width = (
                    default_step_width
                    * step_scale[
                        int((self.terrain_num[2] - 1) / self.cfg.num_rows)
                        % len(step_scale)
                    ]
                )
            else:
                self.terrain_num[3] += 1
                step_width = (
                    default_step_width
                    * step_scale[
                        int((self.terrain_num[3] - 1) / self.cfg.num_rows)
                        % len(step_scale)
                    ]
                )
            terrain_utils.pyramid_stairs_terrain(
                terrain,
                step_width=step_width,
                step_height=np.clip(
                    step_slope * step_width, -max_step_height, max_step_height
                ),
                platform_size=3.0,
            )
        elif choice < self.proportions[4]:
            self.terrain_num[4] += 1
            num_rectangles = 20
            rectangle_min_size = 1.0
            rectangle_max_size = 2.0
            terrain_utils.discrete_obstacles_terrain(
                terrain,
                discrete_obstacles_height,
                rectangle_min_size,
                rectangle_max_size,
                num_rectangles,
                platform_size=3.0,
            )
        elif choice < self.proportions[5]:
            self.terrain_num[5] += 1
            if getattr(self.cfg, "scene6_enabled", False):
                straight_stair_course_terrain(
                    terrain,
                    step_depth=default_step_width,
                    step_height=abs(
                        np.clip(step_height, -max_step_height, max_step_height)
                    ),
                    stair_width=self.cfg.scene6_stair_width,
                    spawn_length=self.cfg.scene6_spawn_length,
                    spawn_edge_margin=self.cfg.scene6_spawn_edge_margin,
                    trailing_ground_length=(
                        self.cfg.scene6_trailing_ground_length
                    ),
                )
            else:
                terrain_utils.stepping_stones_terrain(
                    terrain,
                    stone_size=stepping_stones_size,
                    stone_distance=stone_distance,
                    max_height=0.0,
                    platform_size=4.0,
                )
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.0)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        scene6_spawn_origin_x = getattr(
            terrain, "scene6_spawn_origin_x", None
        )
        env_origin_x = (
            i * self.env_length + scene6_spawn_origin_x
            if scene6_spawn_origin_x is not None
            else (i + 0.5) * self.env_length
        )
        env_origin_y = (j + 0.5) * self.env_width
        origin_local_x = (
            scene6_spawn_origin_x
            if scene6_spawn_origin_x is not None
            else self.env_length / 2.0
        )
        x1 = max(0, int((origin_local_x - 1) / terrain.horizontal_scale))
        x2 = min(
            terrain.length,
            int((origin_local_x + 1) / terrain.horizontal_scale),
        )
        y1 = int((self.env_width / 2.0 - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width / 2.0 + 1) / terrain.horizontal_scale)
        if scene6_spawn_origin_x is not None:
            origin_x_px = int(
                np.clip(
                    round(origin_local_x / terrain.horizontal_scale),
                    0,
                    terrain.length - 1,
                )
            )
            origin_y_px = int(
                np.clip(
                    round((self.env_width / 2.0) / terrain.horizontal_scale),
                    0,
                    terrain.width - 1,
                )
            )
            env_origin_z = (
                terrain.height_field_raw[origin_x_px, origin_y_px]
                * terrain.vertical_scale
            )
        else:
            env_origin_z = (
                np.max(terrain.height_field_raw[x1:x2, y1:y2])
                * terrain.vertical_scale
            )
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def straight_stair_course_terrain(
    terrain,
    step_depth,
    step_height,
    stair_width=1.5,
    spawn_length=1.5,
    spawn_edge_margin=1.0,
    trailing_ground_length=0.4,
):
    """Build a straight course with approach, ascent, top, and descent."""
    stair_width_px = max(1, int(round(stair_width / terrain.horizontal_scale)))
    center_y = terrain.width // 2
    y_start = max(0, center_y - stair_width_px // 2)
    y_end = min(terrain.width, y_start + stair_width_px)

    step_depth_px = max(1, int(step_depth / terrain.horizontal_scale))
    step_height_raw = int(
        round(abs(step_height) / terrain.vertical_scale)
    )
    # Leave enough level ground around the spawn for the full robot footprint.
    spawn_origin_px = max(
        1, int(round(spawn_edge_margin / terrain.horizontal_scale))
    )
    first_riser_px = min(
        terrain.length - 1,
        spawn_origin_px
        + max(1, int(round(spawn_length / terrain.horizontal_scale))),
    )
    trailing_ground_px = max(
        1,
        int(round(trailing_ground_length / terrain.horizontal_scale)),
    )
    top_platform_px = max(
        step_depth_px, int(round(0.5 / terrain.horizontal_scale))
    )
    available_run_px = max(
        0,
        terrain.length
        - first_riser_px
        - trailing_ground_px
        - top_platform_px,
    )
    num_steps = available_run_px // (2 * step_depth_px)
    ascent_end_px = first_riser_px + num_steps * step_depth_px
    descent_start_px = terrain.length - (
        trailing_ground_px + num_steps * step_depth_px
    )

    centerline_profile = np.zeros(
        terrain.length, dtype=terrain.height_field_raw.dtype
    )
    for step_idx in range(num_steps):
        x_start = first_riser_px + step_idx * step_depth_px
        x_end = x_start + step_depth_px
        centerline_profile[x_start:x_end] = (
            step_idx + 1
        ) * step_height_raw

    top_height_raw = num_steps * step_height_raw
    centerline_profile[ascent_end_px:descent_start_px] = top_height_raw

    for step_idx in range(num_steps):
        x_start = descent_start_px + step_idx * step_depth_px
        x_end = x_start + step_depth_px
        centerline_profile[x_start:x_end] = (
            num_steps - step_idx - 1
        ) * step_height_raw

    terrain.height_field_raw.fill(0)
    terrain.height_field_raw[:, y_start:y_end] = centerline_profile[:, None]
    terrain.scene6_spawn_origin_x = (
        spawn_origin_px * terrain.horizontal_scale
    )


def gap_terrain(terrain, gap_size, platform_size=1.0):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    terrain.height_field_raw[
        center_x - x2 : center_x + x2, center_y - y2 : center_y + y2
    ] = -1000
    terrain.height_field_raw[
        center_x - x1 : center_x + x1, center_y - y1 : center_y + y1
    ] = 0


def pit_terrain(terrain, depth, platform_size=1.0):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
