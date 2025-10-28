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
from zai import ZhipuAiClient  # pip install zai-sdk
import os
import base64
import json
import time
from PIL import Image
from pathlib import Path
import logging
from openai import OpenAI
import re


PROMPT = """
# Role: 
You are a GUI Agent, and your primary task is to respond accurately to user requests or questions. In addition to directly answering the user's queries, you can also use tools or perform GUI operations directly until you fulfill the user's request or provide a correct answer. You should carefully read and understand the images and questions provided by the user, and engage in thinking and reflection when appropriate. 

# Background:
1. The user query: {instruction}.
2. The screen resolution: {img_width}x{img_height}.
3. Task Progress: You have done the following operation on the current device: {history}.

# Action Space: The following are all the actions you can take. You need to choose the action you want to perform. You do not need to provide precise coordinates, but you must clearly and accurately describe the action you are going to perform, including the necessary text content and icon descriptions.
1. "SWIPE[UP]": Swipe the screen up.
2. "SWIPE[DOWN]": Swipe the screen down.
3. "SWIPE[LEFT]": Swipe the screen left.
4. "SWIPE[RIGHT]": Swipe the screen right.
5. "CLICK[x,y]": Click the screen at the coordinates (x,y).
6. "LONG_PRESS[x,y]": Long press the screen at the coordinates (x,y).
7. "TYPE[TEXT]": Type the specified text into the activated input box.
8. "OPEN[app]": Open an app on the device at any page. You must open app with action OPEN[app] directly and use Chinese name, do not click the app icon to open it.
9. "WAIT[seconds]": Wait for the specified seconds.
10. "PRESS_BACK": Navigate to the previous screen.
11. "PRESS_HOME": Navigate to the home screen.
12. "TASK_COMPLETE[answer]": Mark the task as complete. If the instruction requires answering a question, provide the answer inside the brackets. If no answer is needed, use empty brackets "TASK_COMPLETE[]".\n'

# Output Format
1. Reason: the reason for the action and the memory. Your reason should include, but not limited to:- the content of the GUI, especially elements that are tightly related to the user goal- the step-by-step thinking process of how you come up with the new action. 
2. Action: If you choose to 'CLICK' or 'LONG_PRESS', do not provide precise coordinates. Clearly and accurately describe the icon you are going to click or press, including the necessary text content and icon descriptions.

Your answer should look like:
Reason: ...
Action: ...
"""

OCR_PROMPT = """You are a helpful goal-oriented assistant. You need to rewrite the descriptions of click or long-press actions into directly executable actions. Accurately understand descriptions of clicking icons and content, ignore the original coordinates, and output precise click coordinates in the specified response format.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type": "function", "function": {{"name_for_human": "mobile_use", "name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.
* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.
* You must open app with action open[app] in Chinese name directly, do not click the app icon to open it. 
* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.
* The screen's resolution is {width}x{height}.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.", "enum": ["click", "long_press"], "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}}}}, "required": ["action"], "type": "object"}}, "args_format": "Format the arguments as a JSON object."}}}}
</tools>

# Information
1. The click thought: {action_thought}
2. The click icon: {action_str} 

# Instruction
1. If there are coordinates in the click icon, please ignore them and focus on the textual thought of the click action. The coordinates may be inaccurate or incorrect.
2. Based on the click thought and click icon description, provide the exact coordinates of the specific icons or elements for interaction.

# Response Format
<action>
{{"name": mobile_use, "arguments": <args-json-object>}}
</action>
"""

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
def get_gpt_response(messages, temperature=0.1, top_k=5, top_p=0.9):

    client = OpenAI(api_key="", base_url="")
    retries = 0
    retry_delay = 2 
    while retries<= MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model='gpt-4o',
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
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

def get_glm_response(messages, temperature=0.1, top_k=5, top_p=0.9):
 
    client = ZhipuAiClient(api_key="")
    retries = 0
    retry_delay = 2  
    while retries<= MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model="glm-4.5V",
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
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


def get_qwen_response(messages, temperature=0.1, top_k=5, top_p=0.9):

    client = OpenAI(api_key="",base_url="")
    retries = 0
    retry_delay = 2  
    while retries<= MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model="qwen-vl-max-latest",
                messages=messages,
                temperature=temperature,
                max_tokens=2048,
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

def get_ocr_response(action_str, action_thought, img_width, img_height, image_path):
    client = OpenAI(api_key='', base_url='')
    retries = 0
    retry_delay = 2  
    image_base64 = encode_image_to_base64(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT.format(action_str=action_str, action_thought=action_thought, width=img_width, height=img_height)},
                {"type": "image_url", "image_url": {"url": image_base64}}
            ],
        }
    ]
    print(OCR_PROMPT.format(action_str=action_str, action_thought=action_thought, width=img_width, height=img_height))
    while retries<= MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model='qwen2.5-vl-7b-instruct',
                messages=messages,
                temperature=0.0,
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

response_map = {
    'gpt': get_gpt_response,
    'glm': get_glm_response,
    'qwen_max': get_qwen_response
}

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        ext = Path(image_path).suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif ext == ".png":
            mime_type = "image/png"
        else:
            mime_type = "image/jpeg"
        return f"data:{mime_type};base64,{encoded_string}"


def parse_mobile_response(response):
    pattern = r"Reason:(.*?)Action:(.*)"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None

    reason = match.group(1).strip()
    action = match.group(2).strip()

    if "<|begin_of_box|>" in action:
        action = action[
            action.index("<|begin_of_box|>") + len("<|begin_of_box|>") : action.rindex(
                "<|end_of_box|>"
            )
        ]

    return {
        "reason": reason,
        "action": action,
    }

class APIAgent:
    def __init__(self, model, agent_config=None):
        self.task = None
        self.model = model
        self.get_response = response_map[self.model]
        self.history = [] 
        
    def set_task(self, task):
        self.task = task  
        self.history = []  
        
    def parse_user_input(self, parsed, img_width, img_height, image_path):
        """解析用户输入的格式 action_type[param] 或 ANSWER[TEXT]"""
        try:
            input_str = parsed['action'].lower()
            if 'press_back' in input_str:
                result = {'action_type': 'system_button', 'button': 'back'}
                return result
            elif 'press_home' in input_str:
                result = {'action_type': 'system_button', 'button': 'home'}
                return result
            elif 'wait' in input_str:
                result = {'action_type': 'wait'}
                return result
            elif 'task_complete' in input_str:
                result = {'action_type': 'complete', 'status': 'success'}
                return result
            elif "swipe" in input_str:
                result = {'action_type': 'swipe'}
                if 'up' in input_str:
                    result['direction'] = 'up'
                elif 'down' in input_str:
                    result['direction'] = 'down'
                elif 'left' in input_str:
                    result['direction'] = 'left'
                elif 'right' in input_str:
                    result['direction'] = 'right'
                return result
            elif 'type' in input_str:
                text_match = re.search(r'\[(.*?)\]', input_str, re.DOTALL)
                text = text_match.group(1).replace('"','').replace("'",'').strip() if text_match else input_str
                result = {'action_type': 'type','text': text}
                return result
            elif 'open' in input_str or 'app' in input_str:
                app_match = re.search(r'\[(.*?)\]', input_str, re.DOTALL)
                app_name = app_match.group(1).replace('"','').replace("'",'').strip().strip() if app_match else input_str
                result = {'action_type': 'open','app': app_name}
                return result

            if 'click' in input_str:
                result = {'action_type': 'click'}
            elif 'long_press' in input_str:
                result = {'action_type': 'long_press'}

            action_str = parsed['action']
            action_thought = parsed['reason']
            grounding_action = get_ocr_response(action_str, action_thought, img_width, img_height, image_path)
            print(f"Raw OCR response: {grounding_action}")
            coordinates_match = re.search(r'"coordinate":\s*\[(\d+),\s*(\d+)\]', grounding_action, re.DOTALL)
            if coordinates_match:
                result['x'] = int(coordinates_match.group(1))
                result['y'] = int(coordinates_match.group(2))
            return result

        except Exception as e:
            logger.error(f"Error when parsing user input: {str(e)}")
            return {'action_type': 'wait', 'reason': 'Error in parsing user input, defaulting to wait.'}


    def scale_image(image_path, scale=0.25):
        """将图片缩放到指定比例，返回PIL Image对象"""
        try:
            with Image.open(image_path) as img:
                new_width = int(img.width * scale)
                new_height = int(img.height * scale)
                scaled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                return scaled_img
        except Exception as e:
            print(f"图片缩放出错: {str(e)}")
            return None

    def agent_step(self, image_path):
        """调用大模型获取操作建议"""
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            image_base64 = encode_image_to_base64(image_path)
        except Exception as e:
            logger.error(f"Error when reading or encoding image: {str(e)}")
            
        try:
            history_content = ''
            if self.history!= []:
                for i, step in enumerate(self.history, 1):  
                    history_content += f"Step {i}: Thought: {step['thought']}; Action: {step['action']}\n"
                    

            if history_content:
                history_content = history_content.rstrip('\n')

            prompt = PROMPT.format(
                instruction=self.task,
                img_width=img_width,
                img_height=img_height,
                history=history_content if history_content else '[First step and no prior actions taken.]'
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            logger.info(f"Current image path: {image_path}")
            response = self.get_response(messages=messages)
            logger.info(f"Raw Response:\n{response}")
            parsed = parse_mobile_response(response)
            step_info = f"Thought:{parsed['reason']} Action:{parsed['action']}"
            self.history.append(
                {
                    "thought": parsed['reason'],
                    "action": parsed['action']
                }
            ) 
            action = self.parse_user_input(parsed, img_width, img_height, image_path) 
            logger.info(f"Parsed action: {action}")
            return action, step_info

        except Exception as e:
            logger.info(f"Error occurred when calling vlm: {str(e)}")
            return None, None

if __name__ == "__main__":

    image_path = ''
    with Image.open(image_path) as img:
        img_width, img_height = img.size
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    instruction = "将第一个商品加入购物车"
    history_memory = "none"
    history = 'none'

    prompt = 'The user query: {instruction}.\nTask progress (You have done the following operation on the current device): {history}.\n'
    user_query = f"The user query: {instruction}.\nTask progress (You have done the following operation on the current device): {history}.\n"

    messages=[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"},
            },
            
        ],
    }
    ]
    print(prompt)
    