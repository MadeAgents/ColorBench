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

logger = logging.getLogger(__name__)

MAX_RETRIES = 5

def extract_numbers(s):
    """提取字符串中的所有连续数字串"""
    return re.findall(r'\d+', s)


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

def position_to_direction(x1, y1, x2, y2):
    """
    将坐标位置转换为方向
    :param x1: 起点x坐标
    :param y1: 起点y坐标
    :param x2: 终点x坐标
    :param y2: 终点y坐标
    :return: 方向字符串
    """
    x1
    if x1 == x2 and y1 == y2:
        logger.info("起点和终点相同，无法确定方向")
        return None
    elif abs(x1 - x2) > abs(y1 - y2):
        if x2 > x1:
            return "right"
        else:
            return "left"
    else:
        if y2 > y1:
            return "down"
        else:
            return "up"

"""
label_pattern = r'CLICK <point>\[\[(\d+),\s*(\d+)\]\]</point>'
            action_pattern = r'CLICK <point>\[\[(\d+),\s*(\d+)\]\]</point>'           
            label_match = re.match(label_pattern, label)
            action_match = re.match(action_pattern, action)
            if not label_match or not action_match:
                return 1, 0
"""

class AtlasAgent:
    def __init__(self, agent_config):
        self.agent_config = agent_config
        self.model = agent_config['model']
        self.api_key = agent_config['api_key']
        self.base_url = agent_config['base_url']
        self.system_prompt = agent_config['system_prompt']
        self.task = None
        self.history = []  
        
    def set_task(self, task):
        self.task = task  
        self.history = []  
        
    def parse_user_input(self, input_str, img_width, img_height):
        """解析用户输入的格式"""

        try:
            if 'PRESS_BACK' in input_str:
                result = {'action_type': 'system_button', 'button': 'back'}
                return result
            elif 'PRESS_HOME' in input_str:
                result = {'action_type': 'system_button', 'button': 'home'}
                return result
            elif 'WAIT' in input_str:
                result = {'action_type': 'wait'}
                return result
            elif 'COMPLETE' in input_str:
                result = {'action_type': 'complete', 'status': 'success'}
                return result
            
            input_lists = input_str.replace('[[', '[').replace(']]', ']').split('[')
            action_type = input_lists[0].strip().lower()  
            action_str = '[' + input_lists[1]  
            match_s = re.search(r'\[(.*)\]', action_str)  
            if match_s:
                action_s = match_s.group(1).strip()
            else:
                action_s = action_str.lstrip('[').split(']')[0].strip()
            logger.info(f'action: {action_type} and {action_s}')  

            if 'click' in action_type:
                positions = action_s.split(',')
                x = int(positions[0])*img_width//1000
                y = int(positions[1])*img_height//1000
                result = {
                    'action_type': 'click',
                    'x': x,
                    'y': y
                }
            elif 'long_press' in action_type:
                positions = action_s.split(',')
                x = int(positions[0])*img_width//1000
                y = int(positions[1])*img_height//1000
                result = {
                    'action_type': 'long_press',
                    'x': x,
                    'y': y
                }
            elif "scroll" in action_type:
                result = {'action_type': 'swipe'}
                action_s = action_s.lower()
                if 'up' in action_s:
                    result['direction'] = 'down'
                elif 'down' in action_s:
                    result['direction'] = 'up'
                elif 'left' in action_s:
                    result['direction'] = 'left'
                elif 'right' in action_s:
                    result['direction'] = 'right'

            elif 'type' in action_type:
                result = {'action_type': 'type','text': action_s.replace('\n','').strip()}
            elif 'open' in action_type or 'app' in action_type:
                result = {'action_type': 'open','app': action_s.replace('\n','').strip()}

                
            return result
        except Exception as e:
            logger.error(f"Error when parsing user input: {str(e)}, default to wait action.")
            return {'action_type': 'wait'}


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
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
            user_prompt = f"Current task instruction: {self.task}\n"
            if self.history!= []:
                history = ''.join([f'Step {si+1}: {content}; 'for si, content in enumerate(self.history)])
                user_prompt += f'\nAction History: {history}.\n'
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.system_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}},
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ]
            
            # logger.info(f"Vanilla Agent Prompt:\n {msg}")
            logger.info(f"Current image path: {image_path}")
            response = get_response(model=self.model,messages=msg,api_key=self.api_key, base_url=self.base_url)
            logger.info(f"Raw Response:\n {response}")

            action, action_thought = self.parse_extract_response(response)

            self.history.append(f'action:{action}, action_thought:{action_thought}')  
            action = self.parse_user_input(action, img_width, img_height)
            logger.info(f"Parsed action: {action}")
            return action, action_thought

        except Exception as e:
            logger.info(f"Error occurred when calling vlm: {str(e)}")
            return None, None

    def parse_extract_response(self, response):
        response = response.replace('Thoughts:', 'thoughts:').replace('Actions:', 'actions:')
        try:
            match1 = re.search(r'thoughts:(.*)actions:', response, re.DOTALL)
            if match1:
                action_thought = match1.group(1).replace('\n',' ').strip()
                logger.info(f"Extract action thought: {action_thought}")
            else:
                action_thought = None
                logger.warning(f"No action thought match found.")
        except Exception as e:
            logger.error(f"Error when extracting action thought: {str(e)}")
            action_thought = None
        try:
            match2 = re.search(r'actions:(.*)', response, re.DOTALL)
            if match2:
                action = match2.group(1).replace('\n',' ').strip()
                logger.info(f"Extract action: {action}")
            else:
                action = None
                logger.warning(f"No action match found.")            
        except Exception as e:
            logger.error(f"Error when extracting action: {str(e)}")
            action = None

        return action, action_thought