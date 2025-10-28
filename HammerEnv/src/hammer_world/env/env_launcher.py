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

from android_world.env import interface

from hammer_world.env.device_controller import get_controller
from hammer_world.env.interface import AsyncAndroidDeviceEnv


def _get_env(device_name: str, adb_path: str) -> interface.AsyncEnv:
    """Creates an AsyncEnv by connecting to an existing Android environment."""
    controller = get_controller(device_name=device_name, adb_path=adb_path)
    return AsyncAndroidDeviceEnv(controller=controller)


def load_and_setup_env(device_name: str, adb_path: str = None) -> interface.AsyncEnv:
    env = _get_env(device_name=device_name, adb_path=adb_path)
    return env
