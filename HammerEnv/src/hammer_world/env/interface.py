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

import time

import dm_env
from android_world.env.interface import (
    AsyncEnv,
    State,
    _get_no_op_action,
)
from android_world.env.android_world_controller import (
    OBSERVATION_KEY_FOREST,
    OBSERVATION_KEY_UI_ELEMENTS,
)

from hammer_world.env import adb_utils
from hammer_world.env import json_action
from hammer_world.env.actuation import execute_adb_action
from hammer_world.env.device_controller import DeviceController


class AsyncAndroidDeviceEnv(AsyncEnv):
    interaction_cache = ""

    def __init__(self, controller: DeviceController):
        self._controller = controller

    @property
    def controller(self) -> DeviceController:
        return self._controller

    def reset(self) -> State:
        # go home
        action = json_action.JSONAction(action_type="navigate_home")
        execute_adb_action(
            action=action,
            screen_elements=[],
            screen_size=self.logical_screen_size,
            env=self.controller,
        )
        # close recents
        action = json_action.JSONAction(action_type="close_recents")
        execute_adb_action(
            action=action,
            screen_elements=[],
            screen_size=self.logical_screen_size,
            env=self.controller,
        )

    def _get_state(self, get_ui_elements: bool = False):
        return _process_timestep(
            self.controller.step(_get_no_op_action(), get_ui_elements=get_ui_elements)
        )

    def get_state(self, wait_to_stabilize: bool = False, get_ui_elements: bool = False) -> State:
        if wait_to_stabilize:
            time.sleep(1)
        return self._get_state(get_ui_elements=get_ui_elements)

    def ask_question(self, question: str, timeout_seconds: float = -1.0) -> str | None:
        pass

    def execute_action(self, action: json_action.JSONAction) -> None:
        """Executes action on the environment."""
        if action.action_type == json_action.ANSWER:
            self.interaction_cache = action.text
            if action.text:
                self.display_message(action.text, header="Agent answered:")
            return
        if action.action_type == json_action.STATUS:
            # Do nothing if it is a termination action.
            return
        state = self.get_state(wait_to_stabilize=False)
        execute_adb_action(
            action=action,
            screen_elements=state.ui_elements,
            screen_size=self.logical_screen_size,
            env=self.controller,
        )

    @property
    def foreground_activity_name(self) -> str:
        """Returns the activity name of the app currently opened in foreground."""
        pass

    @property
    def device_screen_size(self) -> tuple[int, int]:
        """Returns the screen size of the environment in pixels: (width, height)."""
        return self.controller.device_screen_size

    @property
    def logical_screen_size(self) -> tuple[int, int]:
        return adb_utils.get_logical_screen_size(self.controller)

    def close(self):
        pass

    def hide_automation_ui(self) -> None:
        """Hides any UI, such as screen coordinates,."""

    @property
    def orientation(self) -> int:
        """Returns the orientation of the environment.

        Returns: 0 for portrait, 1 for landscape, 2 for reverse portrait,
        3 for reverse landscape.
        """

    @property
    def physical_frame_boundary(self) -> tuple[int, int, int, int]:
        """Returns the physical frame boundary of the environment.

        Returns: First two integers are the coordinates for top left corner, last
        two are for lower right corner. All coordinates are given in portrait
        orientation.
        """


def _process_timestep(timestep: dm_env.TimeStep) -> State:
    """Parses timestep observation and returns State."""
    return State(
        pixels=timestep.observation["pixels"],
        forest=timestep.observation[OBSERVATION_KEY_FOREST],
        ui_elements=timestep.observation[OBSERVATION_KEY_UI_ELEMENTS],
        auxiliaries={},
    )
