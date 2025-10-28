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

"""Ref:
https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/mobile_agent.ipynb
https://github.com/QwenLM/Qwen2.5-VL/blob/main/cookbooks/utils/agent_function_call.py
"""

import math
import time
from absl import logging
from openai import OpenAI
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from qwen_agent.tools.base import BaseTool, register_tool
from typing import Tuple, Union

import json
import re
import random
from server.client import HammerEnvClient

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


@register_tool("mobile_use")
class MobileUse(BaseTool):
    @property
    def description(self):
        return f"""
Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {self.display_width_px}x{self.display_height_px}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.
""".strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "click",
                    "long_press",
                    "swipe",
                    "type",
                    "open",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "coordinate": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.",
                "type": "array",
            },
            "coordinate2": {
                "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.",
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=key`, `action=type`, and `action=open`.",
                "type": "string",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                "type": "number",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object",
    }

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params: Union[str, dict], **kwargs):
        params = self._verify_json_format_args(params)
        action = params["action"]
        if action == "key":
            return self._key(params["text"])
        elif action == "click":
            return self._click(coordinate=params["coordinate"])
        elif action == "long_press":
            return self._long_press(coordinate=params["coordinate"], time=params["time"])
        elif action == "swipe":
            return self._swipe(coordinate=params["coordinate"], coordinate2=params["coordinate2"])
        elif action == "type":
            return self._type(params["text"])
        elif action == "open":
            return self._open(params["text"])
        elif action == "wait":
            return self._wait(params["time"])
        elif action == "terminate":
            return self._terminate(params["status"])
        else:
            raise ValueError(f"Unknown action: {action}")

    def _key(self, text: str):
        raise NotImplementedError()

    def _click(self, coordinate: Tuple[int, int]):
        raise NotImplementedError()

    def _long_press(self, coordinate: Tuple[int, int], time: int):
        raise NotImplementedError()

    def _swipe(self, coordinate: Tuple[int, int], coordinate2: Tuple[int, int]):
        raise NotImplementedError()

    def _type(self, text: str):
        raise NotImplementedError()


    def _open(self, text: str):
        raise NotImplementedError()

    def _wait(self, time: int):
        raise NotImplementedError()

    def _terminate(self, status: str):
        raise NotImplementedError()


class Operator:
    def __init__(
        self, device_client: HammerEnvClient, model_name="Qwen2.5-VL-72B-Instruct", max_steps=20
    ):
        self.device_client = device_client
        self.max_steps = max_steps
        self.model_name = model_name

    def run(self, task):
        logging.info(f"task: {task}")
        tic = time.perf_counter()
        screenshot_base64 = self.device_client.init_task(task)
        screen_size = self.device_client.device_info["screen_size"]
        toc = time.perf_counter()
        logging.debug(f"client init_task execution time: {toc - tic:.2f}s")

        step = 0
        history = []
        historical_actions = []
        while step < self.max_steps:
            messages = _input_messages(
                task=task,
                screenshot_base64=screenshot_base64,
                history=historical_actions,
                screen_size=screen_size,
            )
            response = get_chat_completion(messages=messages, model_id=self.model_name)
            logging.info(f"response: {response}")

            history.append(
                {
                    "observation": screenshot_base64,
                    "response": response,
                    "action": None,
                }
            )
            try:
                action,thought = _extract_action(response)
                print("333333333333333",action,thought)
                history_combine = f"action:{action},thought:{thought}"
                historical_actions.append(json.dumps(history_combine, ensure_ascii=False))
                logging.debug(f"action: {action}")
                if action is not None:
                    action = _convert_action(action=action, screen_size=screen_size)
                history[-1]["action"] = json.dumps(action, ensure_ascii=False)
                
                # next step
                if action is None:
                    break
                if action["action_type"] == "status":
                    break
                action_type = action.pop("action_type")
                action_args = action
                action = {
                    "name": action_type,
                    "arguments": json.dumps(action_args, ensure_ascii=False),
                }
                screenshot_base64 = self.device_client.step(action=action)
            except Exception as e:
                logging.error(e)
                break
            step += 1

        screenshot_base64 = self.device_client.save_task()

        return {
            "task": task,
            "trajectory": history,
        }


def _input_messages(
    task,
    screenshot_base64,
    history,
    screen_size,
    patch_size=14,
    merge_size=2,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
):
    w, h = screen_size
    resized_height, resized_width = smart_resize(
        height=h,
        width=w,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
    )
    directions = ["左上方", "右上方", "左下方", "右下方"]
    selected = random.choice(directions)
    tmp_query = f"注意：只能探索如下区域内的内容：{selected}区域"
    system_message = NousFnCallPrompt().preprocess_fncall_messages(
        messages=[
            Message(
                role="system",
                content=[ContentItem(text=f"You are a helpful assistant. At each step, you MUST first output a thought (within 10 words) explaining why you take the action, wrapped in <thought></thought> tags. Then, output the function call. Note: 1. Do not repeatedly retry the same action with identical parameters (e.g., clicking the same coordinate repeatedly).")],
            ),
        ],
        functions=[mobile_use.function],
        lang=None,
    )

    history = [f"Step {idx}: {action}" for idx, action in enumerate(history, start=1)]
    history = "; ".join(history)

    messages = []
    system_message = system_message[0].model_dump()
    #logging.debug(f"system message: {system_message}\nhistorical actions: {history}")
    
    messages.append(
        {
            "role": "system",
            "content": [{"type": "text", "text": msg["text"]} for msg in system_message["content"]],
        }
    )
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": screenshot_base64},
                },
                {
                    "type": "text",
                    "text": f"The user query: {task}. (You have done the following operation on the current device): {history}",
                },
            ],
        }
    )
    return messages


def _extract_action(response):
    
    action = None
    thought = None
    try:
        response = response.replace("📐", "</tool_call>")
        response = response.replace("⚗", "</tool_call>")
        response = response.replace("⚗️", "</tool_call>")
        matched = re.search(r"<tool_call>\n(.*?)\n</tool_call>", response, flags=re.DOTALL)
        #matched_thought = re.search(r'<thought>\n(.*?)\n</thought>', response, flags=re.DOTALL)
        matched_thought = re.search(r'<thought>(.*?)</thought>', response, flags=re.DOTALL)
        
        if matched:
            action = matched.group(1)
            action = action.replace("<tool_call>", "").replace("</tool_call>", "")
            action = action.strip()
            action = json.loads(action) if action else None
        if matched_thought:
            
            thought = matched_thought.group(1)
            thought = thought.replace("<thought>", "").replace("</thought>", "")
            thought = thought.strip()
            thought = json.loads(thought) if thought else None
        else:
            print('11111111111111 没有匹配到')
    except Exception as e:
        logging.warning(f"parsing action failed, {response}, {e}")
    return action, thought


def _convert_action(
    action,
    screen_size,
    patch_size=14,
    merge_size=2,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
):
    action = action["arguments"]
    action_type = action["action"]
    w, h = screen_size
    resized_height, resized_width = smart_resize(
        height=h,
        width=w,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    logging.debug(
        f"screen size: {screen_size}, resized screen size: {[resized_width, resized_height]}"
    )
    rescale_w = w / resized_width
    rescale_h = h / resized_height

    match action_type:
        case "click":
            x, y = action["coordinate"]
            return {
                "action_type": "click",
                "x": int(x * rescale_w),
                "y": int(y * rescale_h),
            }
        case "long_press":
            x, y = action["coordinate"]
            return {
                "action_type": "long_press",
                "x": int(x * rescale_w),
                "y": int(y * rescale_h),
            }
        case "swipe":
            # # to execute_adb_action "scroll" (src/hammer_world/env/actuation.py)
            # start_x, start_y = action["coordinate"]
            # end_x, end_y = action["coordinate2"]
            # dx = end_x - start_x
            # dy = end_y - start_y
            # if abs(dx) < abs(dy):
            #     if dy > 0:
            #         direction = "up"
            #     else:
            #         direction = "down"
            # else:
            #     if dx > 0:
            #         direction = "left"
            #     else:
            #         direction = "right"
            # return {"action_type": "scroll", "direction": direction}

            # to execute_adb_action "swipe" (src/hammer_world/env/actuation.py)
            touch_xy = action["coordinate"]
            touch_xy = [int(touch_xy[0] * rescale_w), int(touch_xy[1] * rescale_h)]
            lift_xy = action["coordinate2"]
            lift_xy = [int(lift_xy[0] * rescale_w), int(lift_xy[1] * rescale_h)]
            return {
                "action_type": "swipe",
                "touch_xy": touch_xy,
                "lift_xy": lift_xy,
            }

        case "type":
            text = action["text"]
            return {"action_type": "input_text", "text": text}
        case "open":
            text = action["text"]
            return {"action_type": "open_app", "app_name": text}
        case "wait":
            return {"action_type": "wait"}
        case "terminate":
            status = action["status"]
            return {"action_type": "status", "goal_status": status}
        case _:
            return {"action_type": "unknown"}


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def get_chat_completion(messages, client=None, model_id="Qwen2.5-VL-72B-Instruct"):
    client = client or OpenAI()
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0,
        max_tokens=512,
        top_p=1,
        stream=False,
    )
    
    response = completion.choices[0].message.content
    print('111111111111',response)
    return response
