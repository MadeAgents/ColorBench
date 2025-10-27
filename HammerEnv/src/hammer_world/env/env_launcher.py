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
