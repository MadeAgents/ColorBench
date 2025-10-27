from absl import logging
from android_env.env_interface import AndroidEnvInterface
from android_env.wrappers.base_wrapper import BaseWrapper
from android_world.env.representation_utils import UIElement, xml_dump_to_ui_elements
from android_world.env.android_world_controller import (
    OBSERVATION_KEY_FOREST,
    OBSERVATION_KEY_UI_ELEMENTS,
)
from subprocess import CompletedProcess
from typing import Any

import dm_env
import os

from hammer_world.env import adb_utils
from hammer_world.env.device_env import DeviceEnv


class DeviceController(BaseWrapper):
    def __init__(self, device_name, adb_path=None):
        self.adb_path = adb_path or os.environ.get("ADB_PATH", "adb")
        self.device_name = device_name
        self._env = DeviceEnv(device_name=self.device_name, adb_path=self.adb_path)

    @property
    def device_screen_size(self) -> tuple[int, int]:
        """Returns the screen size of the environment in pixels: (width, height)."""
        return adb_utils.get_screen_size(self._env)

    @property
    def env(self) -> AndroidEnvInterface:
        return self._env

    def execute_adb_call(self, args) -> CompletedProcess:
        return self._env.execute_adb_call(args=args)

    def get_ui_elements(self) -> list[UIElement]:
        """Returns the most recent UI elements from the device."""
        return xml_dump_to_ui_elements(adb_utils.uiautomator_dump(self._env))

    def step(self, action: Any, get_ui_elements: bool = False) -> dm_env.TimeStep:
        action = self._process_action(action)
        return self._process_timestep(self._env.step(action), get_ui_elements=get_ui_elements)

    def _process_timestep(
        self, timestep: dm_env.TimeStep, get_ui_elements: bool = False
    ) -> dm_env.TimeStep:
        """Adds a11y tree info to the observation."""
        if not get_ui_elements:
            timestep.observation[OBSERVATION_KEY_FOREST] = None
            timestep.observation[OBSERVATION_KEY_UI_ELEMENTS] = []
            return timestep
        forest = None
        ui_elements = []
        try:
            ui_elements = self.get_ui_elements()
        except Exception as e:
            logging.warning(f"failed to get UI elements, {e}")
            pass
        timestep.observation[OBSERVATION_KEY_FOREST] = forest
        timestep.observation[OBSERVATION_KEY_UI_ELEMENTS] = ui_elements
        return timestep


def get_controller(device_name, adb_path=None) -> DeviceController:
    return DeviceController(device_name=device_name, adb_path=adb_path)
