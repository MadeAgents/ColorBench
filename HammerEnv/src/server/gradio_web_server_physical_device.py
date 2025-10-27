#import android_world
import argparse
from absl import logging
from android_world.env.representation_utils import UIElement
from dataclasses import asdict
from dotenv import load_dotenv
from pathlib import Path
from typing import List

import json
import gradio as gr
import numpy as np
import os
import re
import uuid
import time


from hammer_world.env import json_action
from hammer_world.env.env_launcher import load_and_setup_env
from server.utils import (
    DeviceManager,
    get_action_param_prompt_grid,
    get_action_param_prompt_som,
    get_action_types,
    get_ip,
    image_to_base64,
    screenshot_to_grid_base64,
    screenshot_to_som_base64,
)

logging.set_verbosity("debug")

WORK_HOME = Path(__file__).parent.parent.parent
logging.debug(f"work home: {WORK_HOME}")

load_dotenv((WORK_HOME / ".env").as_posix())

_ADB_PATH = os.environ.get("ADB_PATH") or "adb"
logging.debug(f"adb path: {_ADB_PATH}")

SERV_DIR = Path(__file__).resolve().parent
with open(SERV_DIR / "css/block.css", "r") as f:
    block_css = f.read()

device_manager = DeviceManager()
use_som = True
uid = uuid.uuid4().hex
user_action_space = None
close_recents_at_start = False


class State:
    def __init__(self, device_name: str = None):
        self.device = None
        self.env = None
        self.obs = None
        self.instruction = None
        self.history = []
        try:
            self.device = device_manager.request_device(device_name=device_name)
            logging.info(f"Physical device: {self.device}")
            self.env = load_and_setup_env(device_name=self.device.device_name, adb_path=_ADB_PATH)
            self.obs = self.env.get_state(wait_to_stabilize=True)
        except Exception as e:
            logging.error(e)

    def is_available(self):
        return self.device and self.env

    def reset(self):
        if self.is_available():
            if close_recents_at_start:
                self.env.reset()
            self.obs = self.env.get_state(wait_to_stabilize=True)
        self.instruction = None
        self.history.clear()

    def release(self):
        if self.is_available():
            self.env.reset()
            self.env.close()
            device_manager.release_device(device_name=self.device.device_name)
        self.device = None
        self.env = None
        self.obs = None

    def step(self, action: json_action.JSONAction):
        self.env.execute_action(action=action)
        self.obs = self.env.get_state(wait_to_stabilize=True)

    @property
    def observation(self):
        if self.obs is None:
            self.obs = self.env.get_state(wait_to_stabilize=True)
        return self.obs

    @property
    def screenshot(self) -> np.ndarray:
        return self.observation.pixels

    @property
    def ui_elements(self) -> List[UIElement]:
        return self.observation.ui_elements


def _parse_action(action_type, action_args, ui_elements: List[UIElement]):
    """Ref: android_world/android_world/env/actuation.py"""
    action = {"action_type": action_type}
    # action arguments
    pattern = r"```json\s*({.*?})\s*```"
    match = re.search(pattern, action_args, re.DOTALL)
    if match:
        action_args = match.group(1)
    try:
        action_args = json.loads(action_args)
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing failed: {e}")
        return
    try:
        if action_type in [json_action.CLICK, json_action.DOUBLE_TAP]:
            if index := action_args.get("index"):
                ui_elem = ui_elements[index]
                x, y = ui_elem.bbox_pixels.center
            else:
                x, y = (action_args["x"], action_args["y"])
            action["x"], action["y"] = x, y
        elif action_type in [json_action.SCROLL, json_action.INPUT_TEXT]:
            if index := action_args.get("index"):
                ui_elem = ui_elements[index]
                x, y = ui_elem.bbox_pixels.center
            else:
                x, y = (action_args.get("x"), action_args.get("y"))
            if x and y:
                action["x"], action["y"] = x, y
            if action_type == json_action.INPUT_TEXT:
                action["text"] = action_args["text"]
            elif action_type == json_action.SCROLL:
                action["direction"] = action_args["direction"]
            else:
                pass
        elif action_type == json_action.SWIPE:
            touch_xy = action_args.get("touch_xy")
            lift_xy = action_args.get("lift_xy")
            index = action_args.get("index")
            direction = action_args.get("direction")
            if touch_xy and lift_xy:
                action["touch_xy"] = touch_xy
                action["lift_xy"] = lift_xy
            elif touch_xy and direction:
                action["touch_xy"] = touch_xy
                action["direction"] = action_args["direction"]
            elif index and direction:
                ui_elem = ui_elements[index]
                touch_xy = ui_elem.bbox_pixels.center
                action["touch_xy"] = touch_xy
                action["direction"] = action_args["direction"]
            else:
                if action_args["direction"] == "down":
                    action["direction"] = action_args["direction"]
                elif action_args["direction"] == "up":
                    touch_xy = [500,2300]
                    action["touch_xy"] = touch_xy
                    action["direction"] = action_args["direction"]
                else:
                    touch_xy = [500,1000]
                    action["touch_xy"] = touch_xy
                    action["direction"] = action_args["direction"]
        else:
            action.update(action_args)
    except Exception as e:
        logging.error(e)
        return
    return json_action.JSONAction(**action)


def _transform_action_space(action: json_action.JSONAction):
    assert user_action_space
    action_type = action.action_type

    for _action in user_action_space:
        if _action["name"] == action_type:
            template = _action["template"]
            break

    def _extract_variables(template):
        pattern = r"\{(\w+)\}"
        variables = re.findall(pattern, template)
        return variables

    variables = _extract_variables(template)
    variables = {v: getattr(action, v) for v in variables}
    user_action = template.format(**variables)
    return user_action


def _get_device_info_html(device_info, screenshot_base64):
    device_info_html = f"""<div>
        <p>设备名称：{device_info["device_name"]}</p>
        <p>设备屏幕物理尺寸：{device_info["device_screen_size"]}</p>
        <p>设备屏幕逻辑尺寸：{device_info["logical_screen_size"]}</p>
        <p><img src="{screenshot_base64}" alt="当前屏幕" style="max-width: 100%; height: auto;"/></p>
    </div>"""
    return device_info_html


def update_action_param_prompt(action_type):
    if use_som:
        prompt = get_action_param_prompt_som(action_type=action_type)
    else:
        prompt = get_action_param_prompt_grid(action_type=action_type)
    return gr.update(value=prompt, submit_btn=True, interactive=True)


def message_pair_to_chatbot(message_pair):
    return (
        message_pair[0],
        f"""<div class="chatbot-image-container"><img src="{message_pair[1]}" /></div>""",
    )


def request_device(_state: State, _chatbot: List, device: str, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"request_device. ip: {ip}, device: {device}")

    if _state is not None:
        _state.release()
        del _state
    _state = State(device_name=device)
    if not _state.is_available():
        return [None] + [gr.update() for _ in range(4)]

    _chatbot.clear()

    # observation
    screenshot = _state.screenshot
    screenshot_base64 = image_to_base64(screenshot)

    outputs = [_state]
    # chatbot
    outputs.append(_chatbot)
    # instruction textbox
    outputs.append(gr.update(interactive=True, submit_btn=True))
    # device info html
    device_info = {}
    device_info["device_name"] = _state.device.device_name
    device_info["device_screen_size"] = _state.env.device_screen_size
    device_info["logical_screen_size"] = _state.env.logical_screen_size
    outputs.append(
        gr.update(
            value=_get_device_info_html(
                device_info=device_info, screenshot_base64=screenshot_base64
            )
        )
    )
    # release button
    outputs.append(gr.update(interactive=True))
    return outputs


def init_task(instruction: str, _state: State, _chatbot: List, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"add_text. ip: {ip}, task: {instruction}")

    assert _state

    _state.reset()
    _chatbot.clear()
    # observation
    screenshot = _state.screenshot
    ui_elements = _state.ui_elements
    screenshot_base64 = image_to_base64(image=screenshot)

    _state.instruction = instruction
    _state.history.append(
        {
            "action": None,
            "user_action": None,
            "screenshot": screenshot_base64,
            "ui_elements": [asdict(ui) for ui in ui_elements],
        }
    )

    if use_som:
        screenshot_chatbot_base64 = screenshot_to_som_base64(
            screenshot=screenshot, ui_elements=ui_elements
        )
    else:
        screenshot_chatbot_base64 = screenshot_to_grid_base64(screenshot=screenshot)

    message_pair = ("", screenshot_chatbot_base64)
    _chatbot.append(message_pair_to_chatbot(message_pair=message_pair))

    # state
    outputs = [_state]
    # chatbot
    outputs.append(_chatbot)
    # instruction textbox
    outputs.append(gr.update(value=None))
    # task textbox
    outputs.append(gr.update(value=instruction, show_copy_button=True))
    # action type radio
    outputs.append(gr.update(value=None, interactive=True))
    # action param textbox
    outputs.append(gr.update(value=None))
    # device info html
    device_info = {}
    device_info["device_name"] = _state.device.device_name
    device_info["device_screen_size"] = _state.env.device_screen_size
    device_info["logical_screen_size"] = _state.env.logical_screen_size
    outputs.append(
        gr.update(
            value=_get_device_info_html(
                device_info=device_info, screenshot_base64=screenshot_base64
            )
        )
    )
    # clear button
    outputs.append(gr.update(interactive=True))
    # save button
    outputs.append(gr.update(interactive=True))
    # task result textbox
    #uid = uuid.uuid4().hex
    outputs.append(gr.update(value=uid))
    return outputs


def device_step(_action_type, _action_args, _state: State, _chatbot: List, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"device_step. ip: {ip}, action_type: {_action_type}, action_args: {_action_args}")
    ui_elements = _state.ui_elements
    message_pair = [f"action_type: {_action_type}, action_args: {_action_args}", None]

    action = _parse_action(
        action_type=_action_type, action_args=_action_args, ui_elements=ui_elements
    )
    if action is None:
        _chatbot.append(message_pair)
        return (
            # state
            gr.update(),
            # chatbot
            _chatbot,
            # action type radio
            gr.update(value=None),
            # action param textbox
            gr.update(value=None),
            # device info html
            gr.update(),
            # task result textbox
            gr.update(value=uid),
        )
    _state.step(action=action)
    screenshot = _state.screenshot
    ui_elements = _state.ui_elements
    screenshot_base64 = image_to_base64(screenshot)

    if use_som:
        screenshot_chatbot_base64 = screenshot_to_som_base64(
            screenshot=screenshot, ui_elements=ui_elements
        )
    else:
        screenshot_chatbot_base64 = screenshot_to_grid_base64(screenshot=screenshot)

    user_action = None
    if user_action_space:
        user_action = _transform_action_space(action)

    _state.history.append(
        {
            "action": action.json_str(),
            "user_action": user_action,
            "screenshot": screenshot_base64,
            "ui_elements": [asdict(ui) for ui in ui_elements],
        }
    )

    # state
    outputs = [_state]
    # chatbot
    message_pair = [f"action: {action.json_str()}", screenshot_chatbot_base64]
    _chatbot.append(message_pair_to_chatbot(message_pair=message_pair))
    outputs.append(_chatbot)
    # action type radio
    outputs.append(gr.update(value=None))
    # action param textbox
    outputs.append(gr.update(value=None))
    # device info html
    device_info = {}
    device_info["device_name"] = _state.device.device_name
    device_info["device_screen_size"] = _state.env.device_screen_size
    device_info["logical_screen_size"] = _state.env.logical_screen_size
    outputs.append(
        gr.update(
            value=_get_device_info_html(
                device_info=device_info, screenshot_base64=screenshot_base64
            )
        )
    )
    # task result textbox
    outputs.append(gr.update(value=uid))
    return outputs


def save_task(_state: State, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"save_task. ip: {ip}")

    instruction = _state.instruction
    history = _state.history

    #uid = uuid.uuid4().hex
    print('uid:',uid)
    os.makedirs(f"records/{uid}", exist_ok=True)
    with open(f"records/{uid}/{uid}.json", mode="w", encoding="utf-8") as f:
        json.dump({"instruction": instruction, "history": history}, f, ensure_ascii=False, indent=4)
    #os.makedirs("records", exist_ok=True)
    #with open(f"records/{uid}.json", mode="w", encoding="utf-8") as f:
    #    json.dump({"instruction": instruction, "history": history}, f, ensure_ascii=False, indent=4)
    return


def clear_task(_state: State, _chatbot: List, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"clear_task. ip: {ip}")
    global uid
    _state.reset()
    _chatbot.clear()

    screenshot = _state.screenshot
    screenshot_base64 = image_to_base64(screenshot)

    # state
    outputs = [_state]
    # chatbot
    outputs.append(_chatbot)
    # instruction textbox
    outputs.append(gr.update(value=None))
    # task textbox
    outputs.append(gr.update(value=None, show_copy_button=False))
    # action type radio
    outputs.append(gr.update(value=None, interactive=False))
    # action param textbox
    outputs.append(gr.update(value=None, interactive=False))
    # device info html
    device_info = {}
    device_info["device_name"] = _state.device.device_name
    device_info["device_screen_size"] = _state.env.device_screen_size
    device_info["logical_screen_size"] = _state.env.logical_screen_size
    outputs.append(
        gr.update(
            value=_get_device_info_html(
                device_info=device_info, screenshot_base64=screenshot_base64
            )
        )
    )
    # clear button
    outputs.append(gr.update(interactive=False))
    # save button
    outputs.append(gr.update(interactive=False))
    # task result textbox
    outputs.append(gr.update(value=uid))
    
    uid = uuid.uuid4().hex
    return outputs


def release_device(_state: State, _chatbot: List, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"release_device. ip: {ip}")

    if _state is None:
        pass
    else:
        _state.release()
        del _state
        _state = None
    _chatbot.clear()
    outputs = [_state]
    # chatbot
    outputs.append(_chatbot)
    # device radio
    outputs.append(gr.update(value=None))
    # instruction textbox
    outputs.append(gr.update(value=None, interactive=False, submit_btn=False))
    # task textbox
    outputs.append(gr.update(value=None))
    # action type radio
    outputs.append(gr.update(value=None, interactive=False))
    # action param textbox
    outputs.append(gr.update(value=None, interactive=False, submit_btn=False))
    # device info html
    outputs.append(gr.update(value=None))
    # clear button
    outputs.append(gr.update(interactive=False))
    # save button
    outputs.append(gr.update(interactive=False))
    # task result textbox
    outputs.append(gr.update(value=uid))
    # release button
    outputs.append(gr.update(interactive=False))
    return outputs


def _build_demo():
    notice_markdown = f"""# 📱 Android 移动设备动态交互环境"""
    state = gr.State()
    gr.Markdown(notice_markdown, elem_id="notice_markdown")
    with gr.Row():
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(
                type="tuples",
                label="Android Device",
                height=900,
                show_copy_button=True,
                show_copy_all_button=True,
            )
            instruction_textbox = gr.Textbox(
                lines=1, label="Instruction", submit_btn=False, interactive=False
            )
            action_type_radio = gr.Radio(
                choices=get_action_types(),
                label="Next Action Type",
                interactive=False,
            )
            action_param_textbox = gr.Textbox(
                lines=10,
                label="Next Action Parameters",
                submit_btn=False,
                interactive=False,
            )
        with gr.Column(scale=1):
            device_radio = gr.Radio(
                value=None,
                choices=[],
                label="Available Devices",
                interactive=True,
            )
            task_textbox = gr.Textbox(lines=2, label="Task", interactive=False)
            device_info_html = gr.HTML(
                value="<div>no device</div>",
                label="Device Info",
                container=True,
                show_label=True,
            )
            clear_btn = gr.Button("🔄 Clear (清除任务)", interactive=False)
            save_btn = gr.Button("💾 Save (保存任务)", interactive=False)
            task_result_textbox = gr.Textbox(
                label="Task Result",
                interactive=False,
            )
            release_btn = gr.Button("💨 Release (释放设备)", interactive=False)

    device_radio.select(
        fn=request_device,
        inputs=[state, chatbot, device_radio],
        outputs=[state, chatbot, instruction_textbox, device_info_html, release_btn],
    )

    instruction_textbox.submit(
        fn=init_task,
        inputs=[instruction_textbox, state, chatbot],
        outputs=[
            state,
            chatbot,
            instruction_textbox,
            task_textbox,
            action_type_radio,
            action_param_textbox,
            device_info_html,
            clear_btn,
            save_btn,
            task_result_textbox,
        ],
    )

    action_type_radio.change(
        fn=update_action_param_prompt,
        inputs=[action_type_radio],
        outputs=[action_param_textbox],
        show_progress=True,
    )
    action_param_textbox.submit(
        fn=device_step,
        inputs=[action_type_radio, action_param_textbox, state, chatbot],
        outputs=[
            state,
            chatbot,
            action_type_radio,
            action_param_textbox,
            device_info_html,
            task_result_textbox,
        ],
        show_progress=True,
    )
    clear_btn.click(
        fn=clear_task,
        inputs=[state, chatbot],
        outputs=[
            state,
            chatbot,
            instruction_textbox,
            task_textbox,
            action_type_radio,
            action_param_textbox,
            device_info_html,
            clear_btn,
            save_btn,
            task_result_textbox,
        ],
    )
    save_btn.click(fn=save_task, inputs=state).then(
        fn=clear_task,
        inputs=[state, chatbot],
        outputs=[
            state,
            chatbot,
            instruction_textbox,
            task_textbox,
            action_type_radio,
            action_param_textbox,
            device_info_html,
            clear_btn,
            save_btn,
            task_result_textbox,
        ],
    )
    release_btn.click(
        fn=release_device,
        inputs=[state, chatbot],
        outputs=[
            state,
            chatbot,
            device_radio,
            instruction_textbox,
            task_textbox,
            action_type_radio,
            action_param_textbox,
            device_info_html,
            clear_btn,
            save_btn,
            task_result_textbox,
            release_btn,
        ],
        show_progress=True,
    )
    return state, device_radio


def load_demo(request: gr.Request):
    available_devices = device_manager.get_available_devices()
    ip = get_ip(request)
    logging.info(f"load_demo. ip: {ip}. available devices: {available_devices}")

    # state
    outputs = [None]
    # device radio
    outputs.append(gr.update(choices=available_devices, value=None))
    return outputs


def build_demo():
    with gr.Blocks(css=block_css) as demo:
        state, device_radio = _build_demo()
        demo.load(
            fn=load_demo,
            outputs=[state, device_radio],
        )

    return demo


def _get_use_action_types():
    global user_action_space
    assert user_action_space

    action_types = [action["prompt"]["name"] for action in user_action_space]
    return action_types


def _get_use_action_param_prompt(action_type: str):
    global user_action_space
    assert user_action_space

    prompt = "```json\n{}\n```"
    for action in user_action_space:
        if action_type == action["name"]:
            prompt = action["prompt"]
            break
    return prompt


def main(args):
    global use_som
    use_som = args.use_som
    if args.user_action_space and os.path.isfile(args.user_action_space):
        global user_action_space
        global get_action_types
        global get_action_param_prompt_som
        global get_action_param_prompt_grid
        with open(args.user_action_space, mode="r", encoding="utf-8") as f:
            user_action_space = json.load(f)
            # get_action_types = _get_use_action_types
            get_action_param_prompt_som = _get_use_action_param_prompt
            get_action_param_prompt_grid = _get_use_action_param_prompt
    global close_recents_at_start
    close_recents_at_start = args.close_recents_at_start
    demo = build_demo()
    demo.queue(default_concurrency_limit=args.concurrency_limit).launch(
        share=False, server_name=args.server_name, server_port=args.server_port, show_error=True
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Android 移动设备动态交互环境")

    parser.add_argument("--concurrency-limit", type=int, default=10)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7880)

    parser.add_argument("--use-som", action="store_true", default=True)
    parser.add_argument("--user-action-space", type=str, default="./assets/user_action_space.json")

    parser.add_argument("--close-recents-at-start", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
