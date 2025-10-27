import os
from typing import Dict, Union
from PIL import Image, ImageColor, ImageDraw, ImageFont
from absl import logging
from android_world.agents.m3a_utils import validate_ui_element
from android_world.env.representation_utils import UIElement
from dataclasses import asdict
from threading import Lock

import base64
import dataclasses
import gradio as gr
import io
import json
import numpy as np
import subprocess

import platform
import subprocess

from hammer_world.env import json_action
from server.schema import (
    IMAGE_URL,
    TEXT,
    TYPE,
    URL,
)


@dataclasses.dataclass
class DeviceInfo:
    device_name: str = None
    occupied: bool = False


class DeviceManager:
    def __init__(self):
        self._devices: Dict[str, DeviceInfo] = {}
        self._lock = Lock()

    @property
    def devices(self):
        devices = get_devices()
        for d in devices:
            if d in self._devices:
                continue
            self._devices[d] = DeviceInfo(device_name=d)
        for d in list(self._devices.keys()):
            if d in devices:
                continue
            _ = self._devices.pop(d)
        return self._devices

    def _get_available_devices(self):
        devices = []
        for device in self.devices:
            if self.devices[device].occupied:
                continue
            devices.append(device)
        return devices

    def get_available_devices(self):
        with self._lock:
            devices = self._get_available_devices()
        return devices

    def request_device(self, device_name: str = None) -> DeviceInfo:
        with self._lock:
            available_devices = self._get_available_devices()
            if len(available_devices) == 0:
                raise RuntimeError("No available deives.")
            device_name = device_name or available_devices[0]
            if device_name:
                assert device_name in available_devices
                self.devices[device_name].occupied = True
        return self.devices[device_name]

    def release_device(self, device_name):
        with self._lock:
            if device_name in self.devices:
                self.devices[device_name].occupied = False


def get_devices():
    adb = os.environ.get("ADB_PATH", "adb")

    devices = []
    try:
        #result = subprocess.run(
        #    ["/bin/bash", "-c", f"{adb} devices"],
        #    check=True,
        #    capture_output=True,
        #    text=True,
        #)
        if platform.system() == "Windows":
            # Windows 系统直接执行命令
            result = subprocess.run(
                [adb, "devices"],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            # Linux/macOS 系统通过 bash 执行命令
            result = subprocess.run(
                ["/bin/bash", "-c", f"{adb} devices"],
                check=True,
                capture_output=True,
                text=True,
            )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error: {e.stderr}")
        return devices
    for device in result.stdout.strip().split("\n")[1:]:
        parts = device.strip().split("\t")
        if len(parts) == 2 and parts[1] == "device":
            devices.append(parts[0])
    return devices


def get_action_types():
    return list(json_action._ACTION_TYPES)


def get_action_param_prompt_som(action_type: str):
    prompt = ""
    if action_type in [
        json_action.CLICK,
        json_action.DOUBLE_TAP,
    ]:
        prompt = """请使用 JSON 格式输入屏幕坐标（整型）。
```json
{
    "index": 元素索引，整型
}
```
"""
    elif action_type == json_action.SWIPE:
        prompt = """请使用 JSON 格式。
```json
{
    "index": 元素索引，可选，整型,
    "direction": 滑动方向，字符串类型，取值范围："up", "down", "left", "right"
}
```
"""
    elif action_type == json_action.INPUT_TEXT:
        prompt = """请使用 JSON 格式。
```json
{
    "index": 元素索引，可选，整型,
    "text": "输入文本，字符串类型"
}
```
"""
    elif action_type == json_action.SCROLL:
        prompt = """请使用 JSON 格式。
```json
{
    "direction": 滑动方向，字符串类型，取值范围："up", "down", "left", "right"
}
```
"""
    else:
        prompt = """请使用 JSON 格式。
```json
{}
```
"""
    return prompt


def get_action_param_prompt_grid(action_type: str):
    prompt = ""
    if action_type in [
        json_action.CLICK,
        json_action.DOUBLE_TAP,
        json_action.LONG_PRESS,
    ]:
        prompt = """请使用 JSON 格式输入屏幕坐标（整型）。
```json
{
    "x": x, 整型,
    "y": y, 整型
}
```
"""
    elif action_type == json_action.SWIPE:
        prompt = """请使用 JSON 格式, 你可以使用direction，touch_xy+direction，touch_xy+lift_x三种操作方式。注意: up默认点击屏幕底部往上滑，down默认点击屏幕顶部往下滑，left和right默认点击屏幕中部往左/右滑；当方向无法满足你的需求时，可以尝试只使用touch_xy+lift_xy。
```json
{
    "touch_xy": [x, y], 可选, 整型,
    "lift_xy": [x, y], 可选, 整型,
    "direction": 滑动方向, 可选, 字符串类型, 取值范围: "up", "down", "left", "right"
}
```
"""
    elif action_type == json_action.INPUT_TEXT:
        prompt = """请使用 JSON 格式。
```json
{
    "x": x, 可选, 整型,
    "y": y, 可选, 整型,
    "text": "输入文本, 字符串类型"
}
```
"""
    elif action_type == json_action.SCROLL:
        prompt = """请使用 JSON 格式。
```json
{
    "direction": 滑动方向, 字符串类型, 取值范围: "up", "down", "left", "right"
}
```
"""
    else:
        prompt = """请使用 JSON 格式。
```json
{}
```
"""
    return prompt


def image_to_base64(image: Union[np.ndarray, Image.Image]):
    assert isinstance(image, (np.ndarray, Image.Image))
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    buffered.seek(0)
    image_base64 = base64.b64encode(buffered.getvalue())
    # in py3, b64encode() returns bytes
    return f"""data:image/png;base64,{image_base64.decode("utf-8")}"""


def device_state_to_content(ui_elements, screenshot_base64):
    ui_elements = [json.dumps(asdict(elem)) for elem in ui_elements]
    content = [
        {TYPE: TEXT, TEXT: f"""[{",".join(ui_elements)}]"""},
        {
            TYPE: IMAGE_URL,
            IMAGE_URL: {URL: screenshot_base64},
        },
    ]
    return content


def base64_to_image(image_base64):
    # Remove data URI prefix if present
    if image_base64.startswith("data:image"):
        image_base64 = image_base64.split(",")[1]
    # Decode the Base64 string
    image_base64 = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_base64))
    return image

def get_font(size=100):
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(script_dir, "fonts", "dejavu-fonts-ttf-2.37", "ttf")
    # 确保字体文件夹存在
    if not os.path.exists(font_dir):
        os.makedirs(font_dir)
        print(f"已创建字体目录: {font_dir}")
        print("请将字体文件（如DejaVuSans.ttf）放入此目录")
    
    # 尝试加载项目内的字体文件
    font_files = [
        os.path.join(font_dir, "DejaVuSans.ttf"),
        os.path.join(font_dir, "Arial.ttf"),
        os.path.join(font_dir, "SimHei.ttf"),  # 中文字体支持
    ]
    
    for font_path in font_files:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except OSError:
                continue
    
    # 尝试系统字体
    system_fonts = ["Arial", "SimHei", "DejaVuSans"]
    for font_name in system_fonts:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    
    # 最后手段：使用默认字体
    print("警告: 所有字体加载失败，使用系统默认字体")
    return ImageFont.load_default()

def screenshot_to_som_base64(
    screenshot: np.ndarray, ui_elements: list[UIElement], font_size: float = 50
) -> str:
    screenshot = Image.fromarray(screenshot)
    screen_size = screenshot.size

    overlay = Image.new("RGBA", screen_size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(im=screenshot)
    #font = ImageFont.truetype("DejaVuSans.ttf", 100)
    font = get_font(30)
    for index, ui_element in enumerate(ui_elements):
        if validate_ui_element(ui_element, screen_size):
            position = ui_element.bbox_pixels.center
            overlay_draw.text(xy=position, text=str(index), fill="red", font=font)
    screenshot = screenshot.convert("RGBA")
    screenshot = Image.alpha_composite(screenshot, overlay)
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    buffered.seek(0)
    image_base64 = base64.b64encode(buffered.getvalue())
    return f"""data:image/png;base64,{image_base64.decode("utf-8")}"""


def screenshot_to_grid_base64(
    screenshot: np.ndarray,
    grid_size: int = 100,
    color: str = "black",
    margin: int = 200,
    font_size: float = 50,
    mark_size: int = 20,
) -> str:
    screenshot = Image.fromarray(screenshot)
    w, h = screenshot.size
    if isinstance(color, str):
        try:
            color = ImageColor.getrgb(color)
            color = color + (128,)
        except ValueError:
            color = (255, 0, 0, 128)
    else:
        color = (255, 0, 0, 128)

    # plot grid
    overlay = Image.new(mode="RGBA", size=(w, h), color=(255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(im=overlay)
    grid_points = [(x, y) for x in range(0, w + 1, grid_size) for y in range(0, h + 1, grid_size)]
    for pt in grid_points:
        x, y = pt
        overlay_draw.ellipse(
            xy=((x - mark_size, y - mark_size), (x + mark_size, y + mark_size)),
            fill=color,
        )
    screenshot = screenshot.convert("RGBA")
    screenshot = Image.alpha_composite(screenshot, overlay)

    # plot axes
    color = color[:3] + (255,)
    overlay_size = w + margin, h + margin
    overlay = Image.new(mode="RGBA", size=overlay_size, color=(255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(im=overlay)
    #font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    #font = ImageFont.truetype("DejaVuSans.ttf", 100)
    font = get_font(30)
    for x in range(0, w + 1, grid_size):
        overlay_draw.text((x - mark_size, h), f"{x}", fill=color, font=font)
    for y in range(0, h + 1, grid_size):
        overlay_draw.text((w, y - mark_size), f"{y}", fill=color, font=font)
    overlay.paste(im=screenshot, box=(0, 0))
    screenshot = overlay

    # to base64
    buffered = io.BytesIO()
    screenshot.save(buffered, format="PNG")
    buffered.seek(0)
    image_base64 = base64.b64encode(buffered.getvalue())

    return f"""data:image/png;base64,{image_base64.decode("utf-8")}"""


def get_ip(request: gr.Request):
    if "cf-connecting-ip" in request.headers:
        ip = request.headers["cf-connecting-ip"]
    elif "x-forwarded-for" in request.headers:
        ip = request.headers["x-forwarded-for"]
        if "," in ip:
            ip = ip.split(",")[0]
    else:
        ip = request.client.host
    return ip
