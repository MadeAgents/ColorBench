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

# atlas需要反归一化click坐标

def extract_numbers(s):
    """提取字符串中的所有连续数字串"""
    return re.findall(r'\d+', s)

def extract_before_heading(text):
    # 使用###分割字符串，取分割后的第一个部分
    parts = text.split("###", 1)  # 第二个参数1表示最多分割一次
    if len(parts) > 1:
        return parts[0].strip()  # 返回分割后的第一部分并去除前后空格
    return text.strip()  # 如果没有找到###，返回原字符串（去除空格）

def get_response(model, messages, api_key, base_url, temperature=0.1, top_k=5, top_p=0.9):
    # top_k越小越确定，top_p越大越多样(一般不会太大）

    client = OpenAI(api_key=api_key, base_url=base_url)
    retries = 0
    retry_delay = 2  # 初始重试延迟时间（秒）
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
        self.history = []  # 任务记忆结构
        
    def set_task(self, task):
        self.task = task  # 任务查询
        self.history = []  # 任务记忆结构
        
    def parse_user_input(self, input_str, img_width, img_height):
        # {"name": <function-name>, "arguments": <args-json-object>
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
            action_type = input_lists[0].strip().lower()  # 提取动作类型并去除多余空格
            action_str = '[' + input_lists[1]  # 提取动作部分
            match_s = re.search(r'\[(.*)\]', action_str)  # 提取最外层的[]
            if match_s:
                action_s = match_s.group(1).strip()
            else:
                action_s = action_str.lstrip('[').split(']')[0].strip()
            logger.info(f'action: {action_type} and {action_s}')  # 得到动作部分

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
        # 展示使用的，可以保存使用？倒也不必
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
            # 读取并编码图片
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
            user_prompt = f"Current task instruction: {self.task}\n"
            if self.history!= []:
                history = ''.join([f'Step {si+1}: {content}; 'for si, content in enumerate(self.history)])
                user_prompt += f'\nAction History: {history}.\n'
                # user_prompt += f'\nHistory action descriptions of task progress (You have done the following operation on the current device): {history}.\n'
                # if enable_think:
                #     user_prompt += f'\nBefore answering, explain your reasoning step-by-step in {think_tag_begin}{think_tag_end} tags, and insert them before the <tool_call></tool_call> XML tags.'
                # user_prompt += '\nAfter answering, summarize your action in <thinking></thinking> tags, and insert them after the <tool_call></tool_call> XML tags.'
                # if self.model == 'qwen':
                # user_prompt += '\nAttention! Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. You must use the Chinese name of the app to open an app. You can open the specified app (in Chinese name) at any page.'  # open: Open an app on the device.
                # user_prompt += '\n\nResponse as the following format:\nThoughts: Clearly outline your reasoning process for current step.\nActions: Specify the actual actions you will take based on your reasoning. You should follow action format when generating.'
            
            # {
            #         "role": "system",
            #         "content": [
            #             {"type": "text", "text": self.system_prompt.format(width=img_width, height=img_height)},
            #         ],
            #     },
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
            # print(self.model, self.base_url)
            response = get_response(model=self.model,messages=msg,api_key=self.api_key, base_url=self.base_url)
            logger.info(f"Raw Response:\n {response}")

            action, action_thought = self.parse_extract_response(response)

            self.history.append(f'action:{action}, action_thought:{action_thought}')  # 更新任务记忆结构
            action = self.parse_user_input(action, img_width, img_height)
            logger.info(f"Parsed action: {action}")
            return action, action_thought

        except Exception as e:
            logger.info(f"Error occurred when calling vlm: {str(e)}")
            return None, None

    def parse_extract_response(self, response):
        # 使用###分割字符串，取分割后的第一个部分
        # 从答案中提取出Thought和Action，Thought作为action_description，Action作为action
        response = response.replace('Thoughts:', 'thoughts:').replace('Actions:', 'actions:')
        # print(f"修改后: {response}")
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