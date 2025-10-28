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
from typing import Any

import copy
import time

from hammer_world.env import adb_utils
from hammer_world.env import json_action
from hammer_world.env.device_env import DeviceEnv


def execute_adb_action(
    action: json_action.JSONAction,
    screen_elements: list[Any],  # list[UIElement]
    screen_size: tuple[int, int],
    env: DeviceEnv,
) -> None:
    """Execute an action based on a JSONAction object."""
    if action.action_type in ["click", "double_tap", "long_press"]:
        idx = action.index
        x = action.x
        y = action.y
        if idx is not None:
            if idx < 0 or idx >= len(screen_elements):
                raise ValueError(
                    f"Invalid element index: {idx}, must be between 0 and"
                    f" {len(screen_elements)-1}."
                )
            element = screen_elements[idx]
            if element.bbox_pixels is None:
                raise ValueError("Bbox is not present on element.")
            x, y = element.bbox_pixels.center
            x, y = int(x), int(y)
            if action.action_type == "click":
                adb_utils.tap_screen(x, y, env)
            elif action.action_type == "double_tap":
                adb_utils.double_tap(x, y, env)
            else:
                adb_utils.long_press(x, y, env)
        elif x is not None and y is not None:
            x, y = int(x), int(y)
            if action.action_type == "click":
                adb_utils.tap_screen(x, y, env)
            elif action.action_type == "double_tap":
                adb_utils.double_tap(x, y, env)
            else:
                adb_utils.long_press(x, y, env)
        else:
            raise ValueError(f"Invalid click action: {action}")

    elif action.action_type == "input_text":
        text = action.text
        if text:
            if action.index is not None or (action.x is not None and action.y is not None):
                # First focus on enter text UI element.
                click_action = copy.deepcopy(action)
                click_action.action_type = "click"
                execute_adb_action(click_action, screen_elements, screen_size, env)
                time.sleep(1.0)
            adb_utils.type_text(text, env)
            adb_utils.press_enter_button(env)
        else:
            logging.warning(
                "Input_text action indicated, but no text provided. No " "action will be executed."
            )

    elif action.action_type == "keyboard_enter":
        adb_utils.press_enter_button(env)

    elif action.action_type == "navigate_home":
        adb_utils.press_home_button(env)

    elif action.action_type == "navigate_back":
        adb_utils.press_back_button(env)

    elif action.action_type == "press_keyboard":
        adb_utils.press_keyboard_generic(action.keycode, env)
    elif action.action_type == "drag_and_drop":
        if action.touch_xy is not None and action.lift_xy is not None:
            command = adb_utils.generate_drag_and_drop_command(
                action.touch_xy[0],
                action.touch_xy[1],
                action.lift_xy[0],
                action.lift_xy[1],
            )
            adb_utils.issue_generic_request(command, env)
        else:
            logging.warning(
                "Drag and drop action indicated, but no coordinates provided. No "
                "action will be executed."
            )
    elif action.action_type == "scroll":
        screen_width, screen_height = screen_size
        if action.index:
            x_min, y_min, x_max, y_max = (
                max(screen_elements[action.index].bbox_pixels.x_min, 0),
                max(screen_elements[action.index].bbox_pixels.y_min, 0),
                min(screen_elements[action.index].bbox_pixels.x_max, screen_width),
                min(screen_elements[action.index].bbox_pixels.y_max, screen_height),
            )
        else:
            x_min, y_min, x_max, y_max = (0, 0, screen_width, screen_height)

        start_x, start_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        direction = action.direction
        if direction == "down":
            end_x, end_y = (x_min + x_max) // 2, y_min
        elif direction == "up":
            end_x, end_y = (x_min + x_max) // 2, y_max
        elif direction == "right":
            end_x, end_y = x_min, (y_min + y_max) // 2
        elif direction == "left":
            end_x, end_y = x_max, (y_min + y_max) // 2
        else:
            logging.info("Invalid direction")
            return
        command = adb_utils.generate_swipe_command(
            int(start_x), int(start_y), int(end_x), int(end_y)
        )
        response = adb_utils.issue_generic_request(args=command, env=env)
        if response.returncode != 0:
            logging.error(response)

    elif action.action_type == "swipe":  # Inverse of scroll.
        screen_width, screen_height = screen_size
        if (action.touch_xy is not None) and (action.lift_xy is not None):
            start_x, start_y = action.touch_xy
            end_x, end_y = action.lift_xy
        elif (action.touch_xy is not None) and (action.direction is not None):
            start_x, start_y = action.touch_xy
            direction = action.direction
            if direction == "down":
                end_x, end_y = start_x, screen_height
            elif direction == "up":
                end_x, end_y = start_x, 0
            elif direction == "left":
                end_x, end_y = screen_width, start_y
            elif direction == "right":
                end_x, end_y = 0, start_y
            else:
                logging.info("Invalid direction")
                return
        elif action.index and (action.direction is not None):
            x_min = max(screen_elements[action.index].bbox_pixels.x_min, 0)
            y_min = max(screen_elements[action.index].bbox_pixels.y_min, 0)
            x_max = min(screen_elements[action.index].bbox_pixels.x_max, screen_width)
            y_max = (min(screen_elements[action.index].bbox_pixels.y_max, screen_height),)
            direction = action.direction
            if direction == "down":
                start_x, start_y = mid_x, 0
                end_x, end_y = mid_x, screen_height
            elif direction == "up":
                start_x, start_y = mid_x, screen_height
                end_x, end_y = mid_x, 0
            elif direction == "left":
                start_x, start_y = 0, mid_y
                end_x, end_y = screen_width, mid_y
            elif direction == "right":
                start_x, start_y = screen_width, mid_y
                end_x, end_y = 0, mid_y
            else:
                logging.info("Invalid direction")
                return
        else:
            mid_x, mid_y = 0.5 * screen_width, 0.5 * screen_height
            direction = action.direction
            if direction == "down":
                start_x, start_y = mid_x, 0
                end_x, end_y = mid_x, screen_height
            elif direction == "up":
                start_x, start_y = mid_x, screen_height
                end_x, end_y = mid_x, 0
            elif direction == "left":
                start_x, start_y = 0, mid_y
                end_x, end_y = screen_width, mid_y
            elif direction == "right":
                start_x, start_y = screen_width, mid_y
                end_x, end_y = 0, mid_y
            else:
                logging.info("Invalid direction")
                return
        command = adb_utils.generate_swipe_command(
            int(start_x), int(start_y), int(end_x), int(end_y)
        )
        response = adb_utils.issue_generic_request(args=command, env=env)
        if response.returncode != 0:
            logging.error(response)

    elif action.action_type == "open_app":
        app_name = action.app_name
        if app_name:
            adb_utils.launch_app(app_name, env)
        else:
            raise ValueError("No app name provided")

    elif action.action_type == "wait":
        time.sleep(1.0)

    elif action.action_type == "launch_adb_activity":
        if action.activity_nickname == "app_drawer":
            adb_utils.press_home_button(env)
            time.sleep(1.0)
            start_x, start_y = int(screen_size[0] / 2), int(screen_size[1] * 0.9)
            end_x = start_x
            end_y = int(0.3 * screen_size[1])
            request = adb_utils.generate_swipe_command(start_x, start_y, end_x, end_y)
            adb_utils.issue_generic_request(request, env)
        elif action.activity_nickname == "quick_settings":
            start_x, start_y = int(screen_size[0] / 2), 30
            end_x = start_x
            end_y = int(0.3 * screen_size[1])
            request = adb_utils.generate_swipe_command(
                start_x, start_y, end_x, end_y, duration_ms=10
            )
            adb_utils.issue_generic_request(request, env)
    elif action.action_type == "change_orientation":
        adb_utils.change_orientation(action.orientation, env)
    elif action.action_type == json_action.CLOSE_RECENTS:
        adb_utils.close_recents(env)
    elif action.action_type == json_action.UNKNOWN:
        logging.info("Unknown action type; no action will be executed. Try again...")

    else:
        logging.info("Invalid action type")
