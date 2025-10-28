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

import os
from absl import logging
from dotenv import load_dotenv
from pathlib import Path

from hammer_world.env.device_controller import get_controller
from hammer_world.env.interface import AsyncAndroidDeviceEnv

WORK_HOME = Path(__file__).parent.parent.parent.parent
print(WORK_HOME)
load_dotenv((WORK_HOME / ".env").as_posix())
logging.set_verbosity("info")


def test_async_android_device_env():
    device_controller = get_controller(
        device_name="6f24b6db", adb_path=os.environ.get("ADB_PATH") or "adb"
    )
    device_env = AsyncAndroidDeviceEnv(controller=device_controller)
    obs = device_env.get_state()
    logging.info(obs)


if __name__ == "__main__":
    test_async_android_device_env()
