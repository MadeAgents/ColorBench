import ast
from gradio_client import Client

import re


class HammerEnvClient:
    def __init__(self, src: str):
        self.client = Client(src=src)
        self.task = None
        self.device_info = None

    @property
    def avaliable_devices(self):
        return self.get_avaliable_devices()

    def get_avaliable_devices(self):
        avaliable_devices = self.client.predict(api_name="/load_demo")["choices"]
        return [d[0] for d in avaliable_devices]

    def request_device(self, device: str = None):
        if device:
            assert device in self.avaliable_devices
        device_info = self.client.predict(_chatbot=[], device=device, api_name="/request_device")[2]
        device_info = device_info.get("value")
        self.device_info = get_device_info(device_info=device_info)
        return self.device_info

    def init_task(self, task: str):
        self.task = task
        result = self.client.predict(instruction=self.task, api_name="/init_task")
        # # chatbot
        # screenshot = result[0][-1][1]
        # device info
        screenshot = result[5]["value"]
        return get_screenshot(observation=screenshot)

    def step(self, action):
        """Actions
        - If you think the task has been completed, finish the task by using the status action with complete as goal_status: `{{"action_type": "status", "goal_status": "complete"}}`
        - If you think the task is not feasible (including cases like you don't have enough information or can not perform some necessary actions), finish by using the `status` action with infeasible as goal_status: `{{"action_type": "status", "goal_status": "infeasible"}}`
        - Answer user's question: `{{"action_type": "answer", "text": "<answer_text>"}}`
        - Click/tap on an element on the screen. Use the coordinates to indicate which element you want to click: `{{"action_type": "click", "x": <target_x>, "y": <target_y>}}`.
        - Long press on an element on the screen, similar with the click action above,use the coordinates to indicate which element you want to long press: `{{"action_type": "long_press", "x": <target_x>, "y": <target_y>}}`.
        - Type text into a text field (this action contains clicking the text field, typing in the text and pressing the enter, so no need to click on the target field to start), use the coordinates to indicate the target text field: `{{"action_type": "input_text", "text": <text_input>, "x": <target_x>, "y": <target_y>}}`
        - Press the Enter key: `{{"action_type": "keyboard_enter"}}`
        - Scroll the screen or a scrollable UI element in one of the four directions, use the same coordinates as above if you want to scroll a specific UI element, leave it empty when scrolling the whole screen: `{{"action_type": "scroll", "direction": <up, down, left, right>, "x": <optional_target_x>, "y": <optional_target_y>}}`
        - Open an app (nothing will happen if the app is not installed): `{{"action_type": "open_app", "app_name": <name>}}`
        - Wait for the screen to update: `{{"action_type": "wait"}}`
        """
        result = self.client.predict(
            _action_type=action["name"],
            _action_args=action["arguments"],
            _chatbot=[],
            api_name="/device_step",
        )
        # # chatbot
        # screenshot = result[0][-1][1]
        # device info
        screenshot = result[3]["value"]
        return get_screenshot(observation=screenshot)

    def save_task(self):
        self.client.predict(api_name="/save_task")

    def close(self):
        self.device_info = None
        _ = self.client.predict(_chatbot=[], api_name="/release_device")


def get_screenshot(observation):
    match = re.findall(
        pattern=r"""<img src="(.*?)" .*?/>""",
        string=observation,
        flags=re.DOTALL,
    )

    image_base64 = None
    if match:
        image_base64 = match[0]
    return image_base64


def get_device_info(device_info):
    if not device_info:
        return None
    matched = re.findall(pattern=r"""<p>设备名称：(.*?)</p>""", string=device_info)
    device_name = None
    if matched:
        device_name = matched[0]
    matched = re.findall(pattern=r"""<p>设备屏幕逻辑尺寸：(.*?)</p>""", string=device_info)
    screen_size = None
    if matched:
        screen_size = matched[0]
        screen_size = ast.literal_eval(screen_size)
    return {"device_name": device_name, "screen_size": screen_size}
