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

class Qwen3Agent:
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
        # {"name": <function-name>, "arguments": <args-json-object>
        """解析用户输入的格式 action_type[param] 或 ANSWER[TEXT]"""
        try:
            input_str = input_str.replace("{{", "{")
            match_s = re.search(r'"arguments": (\{.*\})\}', input_str) 
            if match_s:
                action_s = match_s.group(1).strip()
            else:
                action_s = input_str
            logger.info(f'action_s: {action_s}') 

            action = json.loads(action_s)
            logger.info(f"JSON action: {action} and Type {type(action)}")

            action_type = action.get('action', 'None')
            if action_type=='None':
                logger.warning("No action found, use name key to find.")
                action_type = action.get('name', 'None')
                action.pop('name') 
            else:
                action.pop('action')  
            if action_type=='None':
                action_type = 'wait' 
                action = action['arguments']
            
            params = action
            
            result = {'action_type': action_type}

            if action_type in ['click','long_press']:
                coordinate = params.get('coordinate', None)
                result['x'] = int(coordinate[0])*img_width//1000
                result['y'] = int(coordinate[1])*img_height//1000

            elif action_type == 'swipe':
                coordinate1 = params['coordinate']
                coordinate2 = params['coordinate2']
                if coordinate1 and coordinate2:
                    result['direction'] = position_to_direction(
                        coordinate1[0], coordinate1[1],
                        coordinate2[0], coordinate2[1]
                    )
            elif action_type == 'system_button':
                result['button'] = params.get('button', '').lower() 
            elif action_type in ['type']:
                result['text'] = params.get('text', '')
            elif action_type == 'open':
                result['app'] = params.get('text', '')
                if result['app'] == '':
                    result['app'] = params.get('app', '')
            elif action_type == 'terminate':
                result['action_type'] = 'complete'
                result['status'] = params.get('status', '')

                
            return result
        except Exception as e:
            logger.error(f"Error when parsing user input: {str(e)}")
            return None


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
        except Exception as e:
            logger.error(f"Error when reading or encoding image: {str(e)}")
            
        try:
            user_prompt = f"The user query: {self.task}.\n"
            if self.history!= []:
                history = ''.join([f'Step {si+1}: {content}\n'for si, content in enumerate(self.history)])
                print(history)
                user_prompt += f'\nTask progress (You have done the following operation on the current device): {history}.\n'
                user_prompt += '\nAttention! You must open app with action open[app] directly, do not click the app icon to open it. You can open the specified app(in Chinese name) at any page.'
                
            print
            msg = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}
                    ],
                }
            ]
           
            # logger.info(f"Vanilla Agent Prompt:\n {msg}")
            logger.info(f"Current image path: {image_path}")
            response = get_response(model=self.model,messages=msg,api_key=self.api_key, base_url=self.base_url)
            logger.info(f"Raw Response:\n {response}")

            action, action_description, thought = self.parse_extract_response(response)

            self.history.append(f'Thought:{thought} Action:{action_description}') 
            action = self.parse_user_input(action, img_width, img_height)  
            logger.info(f"Parsed action: {action}")
            return action, f'Thought:{thought} Action:{action_description}'

        except Exception as e:
            logger.info(f"Error occurred when calling vlm: {str(e)}")
            return None, None

    def parse_extract_response(self, response):
        try:
            match1 = re.search(r'<tool_call>\n(.*)\n</tool_call>', response, re.DOTALL)
            if match1:
                action = match1.group(1).strip()
            else:
                action = re.search(r'\{.*\}', response).group(0).strip() 
            logger.info(f"Extract action: {action}")
        except Exception as e:
            logger.error(f"Error when extracting action: {str(e)}")
            action = None
        try:
            description = response.split('<tool_call>\n')[0]
            thought = description.split('Action:')[0].split('Thought:')[1].replace('\n','').strip()
            action_description = description.split('Action:')[1].replace('\n','').strip()
            logger.info(f"extract thought: {thought}\n extract description: {action_description}")
           
        except Exception as e:
            logger.error(f"Error when extracting action description: {str(e)}")
            thought = None
            action_description = None

        return action, action_description, thought

if __name__ == "__main__":
    
    system_prompt = ""
    instruction = ""

    stage2_history = ''
    history = [
        "I opened the X app from the home screen.",
    ]
    for idx, his in enumerate(history):
        stage2_history += 'Step ' + str(idx + 1) + ': ' + str(his.replace('\n', '').replace('"', '')) + '; '

    user_query = f"The user query: {instruction}.\nTask progress (You have done the following operation on the current device): {history}.\n"

    messages=[
    {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_query},
            {
                "type": "image_url",
                # "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
            
        ],
    }
    ]   
    response = get_response(model='qwen',messages=messages,api_key='empty', base_url='')
    print(response)
