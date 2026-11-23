"""Microduck velocity walking on HARDER rough terrain, with a difficulty curriculum.

Added 2026-11-07 (Atlantis). Motivation: terrain sweep of the flat-trained walker
(atlantis_10k model_9999) and of Pollen's rough-trained policies showed all of them
trip on a 2 cm step up and stall on 15 deg slopes, because MICRODUCK_ROUGH_TERRAINS_CFG
tops out at 1.5 cm stairs / 1 cm grid / ~5.7 deg slopes and has no difficulty curriculum.

Everything (rewards, obs, DR, commands) is the stock Mjlab-Velocity-Rough-MicroDuck
config; only the terrain generator changes:
  - curriculum=True: one column per terrain type, 10 difficulty rows; mjlab's
    terrain_levels_vel curriculum (kept in rough mode) promotes robots that walk
    far enough and demotes ones that do not.
  - ranges at max difficulty: stairs 4 cm risers, random grid 3 cm, slopes ~17 deg.
Intended to be warm-started from the flat walker (MICRODUCK_WARM_START seed): the
inherited step-based velocity curricula are collapsed to their final stage (as in
velstand) because the loaded policy was trained under those final conditions.
"""

from __future__ import annotations

import copy
import dataclasses

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from .microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg
from .microduck_velstand_env_cfg import _collapse_curricula_to_final

MICRODUCK_HARD_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=4,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.10),
        "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
            proportion=0.30,
            step_height_range=(0.005, 0.04),  # 0.5 cm -> 4 cm risers
            step_width=0.15,
            platform_width=2.0,
            border_width=1.0,
        ),
        # Same grid_width as the stock rough cfg (0.45 m; memory-safe, must not
        # divide 8 m evenly). Height range raised 1 cm -> 3 cm.
        "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.0, 0.03),
            platform_width=1.5,
        ),
        # rise/run 0.03 -> 0.30 (~1.7 deg -> ~16.7 deg); 1 mm vertical quantization.
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.30,
            slope_range=(0.03, 0.30),
            platform_width=2.0,
            vertical_scale=0.001,
        ),
    },
    add_lights=False,
)


def make_microduck_velocity_hard_env_cfg(play: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=True)
    gen = copy.deepcopy(MICRODUCK_HARD_TERRAINS_CFG)
    if play:
        gen.curriculum = False
        gen.num_rows = 5
        gen.num_cols = 5
    cfg.scene.terrain.terrain_generator = gen
    _collapse_curricula_to_final(cfg)  # terrain_levels has no *_stages -> untouched
    return cfg


MicroduckVelocityHardRlCfg = dataclasses.replace(
    MicroduckRlCfg, experiment_name="velocity_hard", run_name="velocity_hard"
)
