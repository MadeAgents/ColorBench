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

from absl import logging

import json

from server.client import HammerEnvClient

logging.set_verbosity("info")


def test_hammer_env_client():
    client = HammerEnvClient("http://localhost:7860/")

    avaliable_devices = client.avaliable_devices
    logging.info(f"available devices: {avaliable_devices}")

    try:
        # request device
        device_info = client.request_device()
        logging.info(f"device info: {device_info}")
        # init task
        obs = client.init_task("ContactsAddContact")
        logging.info(f"screenshot: {obs[:100]}...")
        # action
        action = {"name": "scroll", "arguments": json.dumps({"direction": "up"})}
        obs = client.step(action=action)
        logging.info(f"screenshot: {obs[:100]}...")

    except Exception as e:
        logging.error(e)

    # release
    client.close()


if __name__ == "__main__":
    test_hammer_env_client()
