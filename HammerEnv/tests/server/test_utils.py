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
from dotenv import load_dotenv
from pathlib import Path

from server.utils import DeviceManager, get_devices


WORK_HOME = Path(__file__).parent.parent.parent
load_dotenv((WORK_HOME / ".env").as_posix())
logging.set_verbosity("info")


def test_get_devices():
    devices = get_devices()
    logging.info(devices)


def test_device_manager():
    device_manager = DeviceManager()
    logging.info(device_manager.get_available_devices())

    devices = []
    for _ in range(2):
        try:
            device = device_manager.request_device()
            devices.append(device)
            logging.info(device)
        except Exception as e:
            logging.info(e)
    logging.info(device_manager.get_available_devices())
    for device in devices:
        device_manager.release_device(device_name=device.device_name)
        logging.info(f"{device} has been released")
    logging.info(device_manager.get_available_devices())


if __name__ == "__main__":
    test_get_devices()
    test_device_manager()
