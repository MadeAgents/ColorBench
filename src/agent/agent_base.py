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

from openai import OpenAI
import os
import base64
import json
import time
from PIL import Image
import logging
from openai import OpenAI
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_RETRIES = 5

def get_response(model, messages, api_key, base_url, temperature=0.1, top_k=5, top_p=0.9):

    client = OpenAI(api_key=api_key, base_url=base_url)
    retries = 0
    retry_delay = 2 
    while retries<= MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
            ).choices[0].message.content.strip()
            break
        except Exception as e:
            print(f"请求失败，重试中... 错误信息: {str(e)}")
            retries += 1
            time.sleep(retry_delay)
    
    if retries > MAX_RETRIES:
        print("请求多次失败，终止操作。")
        return None

    return response


class AgentBase(ABC):
    """
    通用 agent 基类。其他开发者可以继承该类并实现 agent_step 方法，
    以及根据需要重写 parse_user_input / parse_extract_response 等方法。
    """

    def __init__(self, agent_config: Dict[str, Any]):
        """
        agent_config: 必含键示例：{
            'model': str,
            'api_key': str,
            'base_url': str,
            'system_prompt': str,
            ...
        }
        """
        self.agent_config: Dict[str, Any] = agent_config
        self.model: str = agent_config.get('model', '')
        self.api_key: str = agent_config.get('api_key', '')
        self.base_url: str = agent_config.get('base_url', '')
        self.system_prompt: str = agent_config.get('system_prompt', '')
        self.task: Optional[str] = None
        self.history: List[str] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def set_task(self, task: str) -> None:
        """设置/重置当前任务与历史。子类可覆写以做额外初始化。"""
        self.task = task
        self.history = []

    def build_system_message(self, width: Optional[int] = None, height: Optional[int] = None) -> Dict[str, Any]:
        """构建系统消息结构（和现有代码格式兼容）。"""
        content = self.system_prompt
        if width is not None and height is not None:
            content = content.format(width=width, height=height)
        return {"role": "system", "content": [{"type": "text", "text": content}]}

    def call_model(self, messages: List[Dict[str, Any]], temperature: float = 0.1, max_tokens: int = 1024) -> Optional[str]:
        """
        调用语言模型的统一封装，默认使用文件内 get_response。
        返回字符串或 None（调用失败）。
        """
        try:
            return get_response(model=self.model, messages=messages, api_key=self.api_key, base_url=self.base_url, temperature=temperature)
        except Exception as e:
            self.logger.error(f"call_model error: {e}")
            return None
        
    @abstractmethod
    def agent_step(self, image_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        子类必须实现：根据 image_path（或其它上下文）生成下一步动作（dict）和动作描述（str）。
        返回 (action_dict, action_description) 或 (None, None) 代表失败/无需操作。
        动作描述用于记录输出历史，将会保存在checkpoint中，便于调试。
        """
        raise NotImplementedError

    def parse_user_input(self, input_str: str, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        解析模型或用户返回的字符串到可执行动作（dict）。
        支持的动作结构：
        {"action_type": "click", "x": int, "y": int}
        {"action_type": "long_press", "x": int, "y": int}
        {"action_type": "swipe", "direction": "up/down/left/right"}
        {"action_type": "type", "text": str}
        {"action_type": "system_button", "button": 'home/back'}
        {"action_type": "open", "app": str}
        {"action_type": "wait"}
        {"action_type": "complete", "status": "success/failure/answer"}
        """
        raise NotImplementedError

    def parse_extract_response(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        默认的 response 提取器：从 <action>...</action> 和 <thinking>...</thinking> 中提取内容。
        子类可覆写以适配不同格式。
        """
        try:
            match1 = re.search(r'<action>(.*)</action>', response, re.DOTALL)
            if match1:
                action = match1.group(1).strip()
            else:
                m = re.search(r'\{.*\}', response)
                action = m.group(0).strip() if m else None
        except Exception:
            action = None

        try:
            match2 = re.search(r'<thinking>\n(.*)\n</thinking>', response, re.DOTALL)
            action_description = match2.group(1).strip() if match2 else None
        except Exception:
            action_description = None

        return action, action_description



if __name__ == "__main__":

    image_path = ''

    with Image.open(image_path) as img:
        img_width, img_height = img.size
   
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            

    system_prompt = "YOUR_SYSTEM_PROMPT_HERE."

    msg = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "The user query: 打开美团"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}
            ],
        }
    ]

    response = get_response(model='', messages=msg, api_key='empty', base_url='')
    print(response)