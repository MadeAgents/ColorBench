from openai import OpenAI
import os
import base64
import json
import time
from PIL import Image
import logging
from openai import OpenAI
import re
# from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor


logger = logging.getLogger(__name__)

MAX_RETRIES = 5

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

class Qwen3Agent:
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
        """解析用户输入的格式 action_type[param] 或 ANSWER[TEXT]"""
        try:
            input_str = input_str.replace("{{", "{")
            match_s = re.search(r'"arguments": (\{.*\})\}', input_str)  # 提取最外层的{}
            if match_s:
                action_s = match_s.group(1).strip()
            else:
                action_s = input_str
            logger.info(f'action_s: {action_s}')  # 得到动作部分
            # 分离动作类型和内容
            # 先使用json函数将str转换为dict
            # action_s = action_s.rstrip('}')+'}'  # 替换单引号为双引号, 补全缺失的括号
            # logger.info(f'action_s: {action_s}')

            action = json.loads(action_s)
            logger.info(f"JSON action: {action} and Type {type(action)}")

            action_type = action.get('action', 'None')
            if action_type=='None':
                logger.warning("No action found, use name key to find.")
                action_type = action.get('name', 'None')
                action.pop('name')  # 移除'name'键并返回其值
            else:
                action.pop('action')  # 移除'action'键并返回其值
            if action_type=='None':
                action_type = 'wait'  # 默认等待
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
                result['button'] = params.get('button', '').lower()  # 统一转为小写
                # 检查是否有from参数
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
        except Exception as e:
            logger.error(f"Error when reading or encoding image: {str(e)}")
            
        try:
            user_prompt = f"The user query: {self.task}.\n"
            if self.history!= []:
                history = ''.join([f'Step {si+1}: {content}\n'for si, content in enumerate(self.history)])
                print(history)
                user_prompt += f'\nTask progress (You have done the following operation on the current device): {history}.\n'
                # user_prompt += f'\nHistory action descriptions of task progress (You have done the following operation on the current device): {history}.\n'
                # if enable_think:
                #     user_prompt += f'\nBefore answering, explain your reasoning step-by-step in {think_tag_begin}{think_tag_end} tags, and insert them before the <tool_call></tool_call> XML tags.'
                # user_prompt += '\nAfter answering, summarize your action in <thinking></thinking> tags, and insert them after the <tool_call></tool_call> XML tags.'
                # if self.model == 'qwen':
                user_prompt += '\nAttention! You must open app with action open[app] directly, do not click the app icon to open it. You can open the specified app(in Chinese name) at any page.'  # open: Open an app on the device.
                # user_prompt += '\n\nResponse as the following format:\n<action>\n{"name": "mobile_use", "arguments": <args-json-object>}\n</action>\n<thinking>\n[action description]\n</thinking>'
                
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
            # print(msg)
            
            # logger.info(f"Vanilla Agent Prompt:\n {msg}")
            logger.info(f"Current image path: {image_path}")
            # print(self.model, self.base_url)
            response = get_response(model=self.model,messages=msg,api_key=self.api_key, base_url=self.base_url)
            logger.info(f"Raw Response:\n {response}")

            action, action_description, thought = self.parse_extract_response(response)

            self.history.append(f'Thought:{thought} Action:{action_description}')  # 更新任务记忆结构
            # \nAction:{action}
            action = self.parse_user_input(action, img_width, img_height)  # 解析动作
            logger.info(f"Parsed action: {action}")
            return action, f'Thought:{thought} Action:{action_description}'

        except Exception as e:
            logger.info(f"Error occurred when calling vlm: {str(e)}")
            return None, None

    def parse_extract_response(self, response):
        # 使用###分割字符串，取分割后的第一个部分
        # 从答案中提取出动作和动作描述

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
            # match2 = re.search(r'<thinking>\n(.*)\n</thinking>', response, re.DOTALL)
            # if match2:
            #     action_description = match2.group(1).strip()
            #     logger.info(f"Extract action description: {action_description}")
            # else:
            #     action_description = None
            #     logger.info(f"No action description match found.")

        except Exception as e:
            logger.error(f"Error when extracting action description: {str(e)}")
            thought = None
            action_description = None

        return action, action_description, thought

if __name__ == "__main__":
    
    system_prompt = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"mobile_use\", \"description\": \"Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.\", \"parameters\": {\"properties\": {\"action\": {\"description\": \"The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.\", \"enum\": [\"click\", \"long_press\", \"swipe\", \"type\", \"answer\", \"system_button\", \"wait\", \"terminate\"], \"type\": \"string\"}, \"coordinate\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.\", \"type\": \"array\"}, \"coordinate2\": {\"description\": \"(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.\", \"type\": \"array\"}, \"text\": {\"description\": \"Required only by `action=type` and `action=answer`.\", \"type\": \"string\"}, \"time\": {\"description\": \"The seconds to wait. Required only by `action=long_press` and `action=wait`.\", \"type\": \"number\"}, \"button\": {\"description\": \"Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`\", \"enum\": [\"Back\", \"Home\", \"Menu\", \"Enter\"], \"type\": \"string\"}, \"status\": {\"description\": \"The status of the task. Required only by `action=terminate`.\", \"type\": \"string\", \"enum\": [\"success\", \"failure\"]}}, \"required\": [\"action\"], \"type\": \"object\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\nRules:\n- Output exactly in the order: Thought, Action, <tool_call>.\n- Be brief: one sentence for Thought, one for Action.\n- Do not output anything else outside those three parts.\n- If finishing, use action=terminate in the tool call.For each function call, there must be an \"action\" key in the \"arguments\" which denote the type of the action."
    # The operation history can be orgnized by Step x: [action]; Step x+1: [action]...
    instriction = "Search for Musk in X and go to his homepage to open the first post."

    stage2_history = ''
    history = [
        "I opened the X app from the home screen.",
    ]
    for idx, his in enumerate(history):
        stage2_history += 'Step ' + str(idx + 1) + ': ' + str(his.replace('\n', '').replace('"', '')) + '; '

    user_query = f"The user query: {instriction}.\nTask progress (You have done the following operation on the current device): {history}.\n"

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
            # {
            #     "type": "image_url",
            #     # "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            # },
            
        ],
    }
    ]   
    # response = get_response(model='qwen',messages=messages,api_key='empty', base_url='')
    # print(response)
    print(system_prompt)
