# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
# flake8: noqa: E303, E111
import argparse
import os

import numpy as np
import torch
import warp as wp
import yaml

from graspgen_utils import add_arg_to_group, add_isaac_lab_args_if_needed, print_blue
from object import ObjectConfig
from warp_kernels import get_joint_pos_kernel, transform_kernel


default_grasp_file = os.path.join(
    os.environ.get("GRASP_DATASET_DIR", ""),
    "grasp_guess_data/robotiq_2f_85/Sphere.yaml",
)
default_max_num_grasps = 0
default_grasp_indices = []

default_start_with_pregrasp_cspace_position = False
default_env_spacing = 1.0

def collect_grasp_display_args(input_dict):
    desired_keys = [
        "default_grasp_file",
        "default_max_num_grasps",
        "default_grasp_indices",
        "default_start_with_pregrasp_cspace_position",
        "default_env_spacing",
    ]
    kwargs = {}
    for key in desired_keys:
        if key in input_dict:
            kwargs[key] = input_dict[key]
        else:
            # Use local default if not provided in input_dict
            kwargs[key] = globals()[key]
    return kwargs


def add_grasp_display_args(
    parser,
    _,
    default_grasp_file=default_grasp_file,
    default_max_num_grasps=default_max_num_grasps,
    default_grasp_indices=default_grasp_indices,
    default_start_with_pregrasp_cspace_position=default_start_with_pregrasp_cspace_position,
    default_env_spacing=default_env_spacing,
):
    from graspgen_utils import register_argument_group

    register_argument_group(parser, "grasp_display", "grasp_display", "Grasp display options")

    add_arg_to_group(
        "grasp_display",
        parser,
        "--grasp_file",
        type=str,
        default=default_grasp_file,
        help="Path to grasp file.",
    )
    add_arg_to_group(
        "grasp_display",
        parser,
        "--max_num_grasps",
        type=int,
        default=default_max_num_grasps,
        help="Maximum number of grasps to process (0 means use all grasps).",
    )
    add_arg_to_group(
        "grasp_display",
        parser,
        "--grasp_indices",
        type=list,
        default=default_grasp_indices,
        help="List of grasp indices to display.",
    )
    add_arg_to_group(
        "grasp_display",
        parser,
        "--start_with_pregrasp_cspace_position",
        action="store_true",
        default=default_start_with_pregrasp_cspace_position,
        help="Start with the pregrasp cspace position instead of the cspace position from the grasp input.",
    )
    add_arg_to_group(
        "grasp_display",
        parser,
        "--env_spacing",
        type=float,
        default=default_env_spacing,
        help="Spacing between environments so the camera can move between them.",
    )

    add_isaac_lab_args_if_needed(parser)


class GraspingDisplayConfig:
    def __init__(
        self,
        grasp_file,
        max_num_grasps,
        grasp_indices,
        start_with_pregrasp_cspace_position,
        device,
        env_spacing,
    ):
        self.grasp_file = grasp_file
        self.max_num_grasps = max_num_grasps
        self.grasp_indices = grasp_indices
        self.start_with_pregrasp_cspace_position = start_with_pregrasp_cspace_position
        self.device = device
        self.env_spacing = env_spacing


class GraspingDisplay:
    def __init__(self, config, wait_for_debugger_attach=False):
        self.config = config
        self.force_headed = True
        self.wait_for_debugger_attach = wait_for_debugger_attach
        self._simulation_app = None

        self.active_env_id = 0
        self.camera_eye_offset = torch.tensor(
            [0.09488407096425105, 0.4248694091778259, 0.4521177672484193],
            dtype=torch.float32,
        )
        self.camera_target_offset = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        self._keyboard = None
        self._keyboard_sub = None
        self._input = None
        self._scene = None
        self._sim = None

        self.cspace_joint_indices = None

        self.load_grasp_file()

    def load_grasp_file(self):
        with open(self.config.grasp_file, "r") as f:
            self.original_grasp_yaml_data = yaml.safe_load(f)

        if not self.original_grasp_yaml_data or "grasps" not in self.original_grasp_yaml_data:
            raise ValueError("Invalid grasp file format")

        if len(self.config.grasp_indices) > 0:
            valid_grasp_indices = [
                [i, g]
                for i, g in enumerate(self.original_grasp_yaml_data["grasps"].values())
                if i in self.config.grasp_indices
            ]
        else:
            valid_grasp_indices = [
                [i, g] for i, g in enumerate(self.original_grasp_yaml_data["grasps"].values())
            ]

        # Apply max_num_grasps limit if specified (0 means use all grasps)
        if self.config.max_num_grasps > 0:
            valid_grasp_indices = valid_grasp_indices[: self.config.max_num_grasps]

        # Pre-allocate numpy array for grasps
        num_grasps = len(valid_grasp_indices)
        grasps = np.zeros((num_grasps, 7), dtype=np.float32)
        bite_points = np.zeros((num_grasps, 3), dtype=np.float32)
        grasp_idx_map = np.zeros(num_grasps, dtype=int)

        cspace_key = (
            "pregrasp_cspace_position"
            if self.config.start_with_pregrasp_cspace_position
            else "cspace_position"
        )
        bite_key = (
            "pregrasp_bite_point"
            if self.config.start_with_pregrasp_cspace_position
            else "bite_point"
        )

        grasp_names = list(self.original_grasp_yaml_data["grasps"].keys())
        cspace_joint_names = list(
            self.original_grasp_yaml_data["grasps"][grasp_names[0]][cspace_key].keys()
        )
        cspace_positions = [[0.0] * len(cspace_joint_names) for _ in range(num_grasps)]

        for i in range(num_grasps):
            grasp = valid_grasp_indices[i][1]
            grasp_idx_map[i] = valid_grasp_indices[i][0]

            position = np.array(grasp["position"], dtype=np.float32)
            orientation_xyz = np.array(grasp["orientation"]["xyz"], dtype=np.float32)
            orientation_w = np.float32(grasp["orientation"]["w"])

            grasps[i] = np.concatenate([position, orientation_xyz, [orientation_w]])
            bite_points[i] = np.array(grasp[bite_key], dtype=np.float32)

            for j, cspace_joint_name in enumerate(cspace_joint_names):
                cspace_positions[i][j] = grasp[cspace_key][cspace_joint_name]

        self.grasps = wp.array(grasps, dtype=wp.transform, device=self.config.device)
        self.bite_points = wp.array(bite_points, dtype=wp.vec3, device=self.config.device)
        self.grasp_idx_map = grasp_idx_map
        self.cspace_joint_names = cspace_joint_names
        self.cspace_positions = wp.array(
            cspace_positions, dtype=wp.float32, device=self.config.device
        )

        self.gripper_file = self.original_grasp_yaml_data.get("gripper_file")
        self.object_config = ObjectConfig.from_isaac_grasp_dict(self.original_grasp_yaml_data)

    def get_usd_path(self, file_path):
        # Expand user path (handle ~ in file paths)
        file_path = os.path.expanduser(file_path)

        if file_path.lower().endswith((".usd", ".usda", ".usdz")):
            return file_path

        usd_file = os.path.splitext(file_path)[0] + ".usd"
        if self.object_config.obj2usd_use_existing_usd and os.path.exists(usd_file):
            print_blue(f"Using existing USD file: {usd_file}")
            return usd_file

        print_blue(f"Creating USD file: {usd_file}")
        usd_file = self.create_usd(usd_file, file_path)
        return usd_file

    def create_usd(self, usd_file, file_path):
        from graspgen_utils import get_simulation_app

        _ = get_simulation_app(
            __file__,
            force_headed=self.force_headed,
            wait_for_debugger_attach=self.wait_for_debugger_attach,
        )

        from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
        from mesh_utils import convert_mesh_to_usd

        physics_material = RigidBodyMaterialCfg(
            static_friction=self.object_config.obj2usd_friction,
            dynamic_friction=self.object_config.obj2usd_friction,
        )

        usd_file = convert_mesh_to_usd(
            usd_file,
            file_path,
            overwrite=True,
            #vertex_scale=self.object_config.object_scale,
            mass=1.0,
            collision_approximation=self.object_config.obj2usd_collision_approximation,
            physics_material=physics_material,
        )
        return usd_file

    def build_grasp_display_scene_cfg(self, num_envs):
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass

        @configclass
        class GraspingSceneCfg(InteractiveSceneCfg):
            dome_light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
            )

            gripper = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                init_state=ArticulationCfg.InitialStateCfg(),
                actuators={
                    "gripper": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=None,
                        damping=None,
                    ),
                },
            )

            object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                    lin_vel=(0.0, 0.0, 0.0),
                    ang_vel=(0.0, 0.0, 0.0),
                ),
            )

        scene_cfg = GraspingSceneCfg(
            num_envs=num_envs,
            env_spacing=self.config.env_spacing,
            filter_collisions=True,
            replicate_physics=True,
        )

        scene_cfg.gripper.spawn = sim_utils.UsdFileCfg(
            usd_path=self.gripper_file,
            activate_contact_sensors=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )
        scene_cfg.object.spawn = sim_utils.UsdFileCfg(
            usd_path=self.usd_path,
            scale=(
                self.object_config.object_scale,
                self.object_config.object_scale,
                self.object_config.object_scale,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )

        return scene_cfg

    def get_initial_joint_pos(self, scene, joint_pos, num_envs, start_idx):
        if self.cspace_joint_indices is None:
            self.cspace_joint_indices = [
                scene["gripper"].data.joint_names.index(name) for name in self.cspace_joint_names
            ]
            self.cspace_joint_indices = wp.array(
                self.cspace_joint_indices, dtype=wp.int32, device=self.config.device
            )

        wp.launch(
            kernel=get_joint_pos_kernel,
            dim=(num_envs, len(self.cspace_joint_names)),
            inputs=[start_idx, 0, self.cspace_positions, self.cspace_joint_indices],
            outputs=[joint_pos],
            device=self.config.device,
        )

    def _set_camera_to_env(self, env_id: int):
        env_id = int(max(0, min(env_id, self._scene.num_envs - 1)))
        self.active_env_id = env_id

        origin = self._scene.env_origins[env_id].detach().cpu()
        eye = tuple((origin + self.camera_eye_offset).tolist())
        target = tuple((origin + self.camera_target_offset).tolist())

        self._sim.set_camera_view(eye=eye, target=target)
        print_blue(
            f"Viewing env {env_id + 1}/{self._scene.num_envs} "
            f"(grasp index {self.grasp_idx_map[env_id]})"
        )

    def _move_camera_by(self, delta: int):
        if self._scene is None:
            return
        new_env = (self.active_env_id + delta) % self._scene.num_envs
        self._set_camera_to_env(new_env)

    def _on_keyboard_event(self, event, *args, **kwargs):
        import carb.input
        from carb.input import KeyboardEventType

        if event.type not in (KeyboardEventType.KEY_PRESS, KeyboardEventType.KEY_REPEAT):
            return True

        if event.input in (carb.input.KeyboardInput.RIGHT, carb.input.KeyboardInput.D):
            self._move_camera_by(+1)
            return True

        if event.input in (carb.input.KeyboardInput.LEFT, carb.input.KeyboardInput.A):
            self._move_camera_by(-1)
            return True

        if event.input == carb.input.KeyboardInput.HOME:
            self._set_camera_to_env(0)
            return True

        return True

    def _subscribe_keyboard(self):
        import carb.input
        import omni.appwindow

        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    def _unsubscribe_keyboard(self):
        self._keyboard_sub = None
        self._keyboard = None
        self._input = None

    def display_grasps(self):
        self.usd_path = self.get_usd_path(self.object_config.object_file)
        num_envs = len(self.grasps)

        from graspgen_utils import get_simulation_app

        simulation_app = get_simulation_app(
            __file__,
            force_headed=self.force_headed,
            wait_for_debugger_attach=self.wait_for_debugger_attach,
        )

        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import build_simulation_context

        sim_cfg = sim_utils.SimulationCfg(device=self.config.device)

        with build_simulation_context(
            device=self.config.device,
            gravity_enabled=False,
            auto_add_lighting=True,
            sim_cfg=sim_cfg,
        ) as sim:
            print("\033[0m", end="")
            sim._app_control_on_stop_handle = None

            scene_cfg = self.build_grasp_display_scene_cfg(num_envs)
            scene = InteractiveScene(scene_cfg)

            self._scene = scene
            self._sim = sim

            sim.reset()

            self._subscribe_keyboard()
            self._set_camera_to_env(0)

            print_blue("Camera controls: Right/D = next env, Left/A = previous env, Home = env 0")

            do_render = True
            gripper = scene["gripper"]
            sim_dt = sim.get_physics_dt()

            count = 0
            try:
                while simulation_app.is_running():
                    if count == 0:
                        gripper_state = gripper.data.default_root_state.clone()

                        wp.launch(
                            kernel=transform_kernel,
                            dim=num_envs,
                            inputs=[0, 0, self.grasps, gripper_state[:, :7], True],
                            device=self.config.device,
                        )

                        # Shift each gripper root pose into its environment's world-space origin.
                        gripper_state[:, :3] += scene.env_origins

                        gripper.write_root_pose_to_sim(gripper_state[:, :7])
                        gripper.write_root_velocity_to_sim(gripper_state[:, 7:])

                        joint_pos = gripper.data.default_joint_pos.clone()
                        joint_vel = gripper.data.default_joint_vel.clone()
                        self.get_initial_joint_pos(scene, joint_pos, num_envs, 0)

                        gripper.set_joint_position_target(joint_pos)
                        gripper.write_joint_state_to_sim(joint_pos, joint_vel)

                        if hasattr(gripper.data, "joint_vel_limits"):
                            temp_vel_limits = gripper.data.joint_vel_limits.clone()
                            vel_limits = wp.array(
                                gripper.data.joint_vel_limits.clone(),
                                dtype=wp.float32,
                                device=self.config.device,
                            )
                        else:
                            temp_vel_limits = gripper.data.joint_velocity_limits.clone()
                            vel_limits = wp.array(
                                gripper.data.joint_velocity_limits.clone(),
                                dtype=wp.float32,
                                device=self.config.device,
                            )

                        vel_limits.fill_(1000000000.0)
                        gripper.write_joint_velocity_limit_to_sim(
                            wp.to_torch(vel_limits, requires_grad=False)
                        )
                        scene.reset()

                        for _ in range(2):
                            scene.write_data_to_sim()
                            sim.step(render=False)
                            scene.update(sim_dt)

                        gripper.write_joint_velocity_limit_to_sim(temp_vel_limits)
                        gripper.write_joint_velocity_to_sim(joint_vel)

                        scene.reset()

                    scene.write_data_to_sim()
                    sim.step(render=do_render)
                    count += 1
                    scene.update(sim_dt)

            finally:
                self._unsubscribe_keyboard()


def main(args):
    grasp_display_cfg = GraspingDisplayConfig(
        grasp_file=args.grasp_file,
        max_num_grasps=args.max_num_grasps,
        grasp_indices=args.grasp_indices,
        start_with_pregrasp_cspace_position=args.start_with_pregrasp_cspace_position,
        device=args.device,
        env_spacing=args.env_spacing,
    )
    grasp_display = GraspingDisplay(
        grasp_display_cfg, wait_for_debugger_attach=args.wait_for_debugger_attach
    )
    grasp_display.display_grasps()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grasp validation through simulation part of Grasp Gen Data Generation."
    )
    add_grasp_display_args(parser, globals(), **collect_grasp_display_args(globals()))
    args_cli = parser.parse_args()
    
    main(args_cli)

