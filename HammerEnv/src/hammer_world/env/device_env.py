# Copyright 2025 OPPO

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PIL import Image
from absl import logging
from android_env.env_interface import AndroidEnvInterface
from android_world.env.representation_utils import forest_to_ui_elements, xml_dump_to_ui_elements

import dm_env
import numpy as np
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import platform
import subprocess

class DeviceEnv(AndroidEnvInterface):
    def __init__(self, device_name: str, adb_path: str):
        self.device_name = device_name
        self.adb_path = adb_path

    def action_spec(self) -> dict[str, dm_env.specs.Array]:
        """Returns the action specification."""

    def observation_spec(self) -> dict[str, dm_env.specs.Array]:
        """Returns the observation specification."""

    def reset(self) -> dm_env.TimeStep:
        """Resets the current episode."""

    def step(self, *args, **kwargs) -> dm_env.TimeStep:
        """Executes `action` and returns a `TimeStep`."""
        adb_command = ["shell", "screencap -p /sdcard/screen.png"]
        # screenshot
        args = " ".join(adb_command)
        _ = self.execute_adb_call(args)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
            adb_command = ["pull", f"/sdcard/screen.png {tmp_file.name}"]
            args = " ".join(adb_command)
            _ = self.execute_adb_call(args)
            screenshot = Image.open(tmp_file.name)
        timestep = dm_env.TimeStep(
            step_type=None,
            reward=None,
            discount=None,
            observation={"pixels": np.array(screenshot)},
        )
        return timestep

    def close(self) -> None:
        """Frees up resources."""

    def execute_adb_call(self, args) -> subprocess.CompletedProcess:
        """Executes `call` and returns its response."""
        cmd = f"{self.adb_path} -s {self.device_name} {args}"
        result = subprocess.CompletedProcess(args=["/bin/bash", "-c", cmd], returncode=-1)
        try:
            #result = subprocess.run(
            #    ["/bin/bash", "-c", cmd],
            #    check=True,
            #    capture_output=True,
            #    text=True,
            #)
            if platform.system() == "Windows":
                # 直接在 Windows 上执行命令
                result = subprocess.run(
                    cmd,  # Windows 直接使用命令列表
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
            else:
                # Linux/macOS 系统通过 bash 执行命令
                result = subprocess.run(
                    ["/bin/bash", "-c", cmd],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Error: {e.stderr}")
        return result
