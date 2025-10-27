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
