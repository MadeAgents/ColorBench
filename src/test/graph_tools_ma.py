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

import os
import math
import json
import shutil
from openai import OpenAI
import base64
import random
import logging
from collections import defaultdict
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

random.seed(42)   

APP_MAP = {
    "bilibili": "哔哩哔哩",
    "高德": "高德地图",
    "WPS": "WPS Office",
    "58": "58同城",
    "去哪儿": "去哪儿旅行",
    "携程": "携程旅行",
    "红果": "红果免费短剧",
    "红果短剧": "红果免费短剧",
    "小布": "小布助手",
    "百度浏览器": "百度",
    "JD": "京东",
}

CLICK_APP_MAP = {
    "微信":[78,191,248,421],
    "QQ":[327,193,499,426],
    "淘宝":[576,472,755,705],
    "哔哩哔哩":[329,474,501,707],
    "抖音":[78,479,250,702],
    "京东":[834,479,1004,707],
    "美团":[78,756,250,988],
    "小红书":[585,758,750,981],
    "百度":[834,758,1004,988],
    "高德地图":[329,1042,501,1273],
    "拼多多":[332,1328,490,1558],
    "携程旅行":[80,1323,250,1553],
    "大众点评":[834,1326,1004,1556],
}

def euclidean_distance(x1, y1, x2, y2):
    """计算两点之间的欧式距离"""
    return math.sqrt((x1 - x2)**2 + (y1 - y2)** 2)


def position_to_direction(x1, y1, x2, y2):
    """
    将坐标位置转换为方向
    :param x1: 起点x坐标
    :param y1: 起点y坐标
    :param x2: 终点x坐标
    :param y2: 终点y坐标
    :return: 方向字符串
    """
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
                     

class Graph_DataSet:
    def __init__(self, graph_config):
        self.graph_json_file = graph_config['graph_file']
        if not self.graph_json_file:
            logger.error("未提供图的JSON文件路径")
            return
        elif not os.path.exists(self.graph_json_file):
            logger.error(f"图的JSON文件路径不存在: {self.graph_json_file}")
            return
        try:
            with open(self.graph_json_file, 'r', encoding='utf-8') as f:
                self.graph_data = json.load(f)
                logger.info(f"成功加载图数据，节点数: {len(self.graph_data)}")
        except json.JSONDecodeError as e:
            logger.error(f"加载图数据时出错: {str(e)}")
            self.graph_data = None

        self.query = None  
        self.trajectory = []  
        self.history_stack = [] 
        self.home_page = graph_config['root_node']
        self.apps = defaultdict(list) 
        for screenshot, actions in self.graph_data[self.home_page].items():
            app = actions[0].get('app', None)
            self.apps[app].append(screenshot)
        logger.info(f"已识别的应用入口:\n {self.apps}")
        logger.info(f"应用数量为: {len(self.apps)}")

    def set_task(self, query):
        self.query = query 
        self.trajectory = [{
            'id': 0,
            'screenshot': self.home_page,  
            'action': None  
        }] 
        self.history_stack = [self.home_page]  

    def clear_task(self, task_json):
        self.query = ''
        self.trajectory = [] 
        self.history_stack = [] 

    def save_trajectory(self, output_dir, use_time, save_image=False, config_name = None, parent_dir = None, task_id = None):
        """
        保存轨迹到指定目录
        :param output_dir: 基础输出目录
        :param use_time: 使用时间
        :param save_image: 是否保存图片
        :param config_name: 配置名称（用于创建主文件夹）
        :param parent_dir: 父目录
        :param task_id: 任务ID（可选）
        """
        safe_query = self.query.replace('/','_').replace(':','_').replace('*','_').replace('?','_').replace('"','_').replace('<','_').replace('>','_').replace('|','_')
        
        # store structure：output_dir/config_name/safe_query/
        if config_name:
            main_folder = os.path.join(output_dir, config_name)
            task_folder = os.path.join(main_folder, safe_query)
        else:
            task_folder = os.path.join(output_dir, safe_query)
        
        os.makedirs(task_folder, exist_ok=True)
        
        # save trajectory JSON
        output_json = os.path.join(task_folder, 'trajectory.json')
        trajectory_data = {
            'task_id': task_id,
            'query': self.query,
            'trajectory': self.trajectory,
            'step_count': len(self.trajectory),
            'use_time': use_time,
            'config_name': config_name
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, ensure_ascii=False, indent=2)
        
        # save images with action points if needed
        if save_image and parent_dir:
            for i, step in enumerate(self.trajectory):
                try:
                    screenshot_path = os.path.join(parent_dir, step.get('screenshot'))
                    if step.get('action') and isinstance(step['action'], dict):
                        action = '_'.join(str(item).replace('/','').replace(' ','') for item in step['action'].values())
                    else:
                        action = f"step_{i}"
                    
                    if screenshot_path and os.path.exists(screenshot_path):
                        if (step.get('action') and 
                            isinstance(step['action'], dict) and 
                            step['action'].get('action_type') in ['click', 'long_press']):
                            try:
                                with Image.open(screenshot_path) as img:
                                    draw = ImageDraw.Draw(img)
                                    x, y = int(step['action']['x']), int(step['action']['y'])
                                    r = 10  
                                    draw.ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3], fill="#FFFFFF")
                                    draw.ellipse([x - r, y - r, x + r, y + r], fill="#FF0000", width=3)
                                    img_output_path = os.path.join(task_folder, f'trajectory_{i}_{action}.png')
                                    img.save(img_output_path)
                            except Exception as e:
                                logger.warning(f"Error when drawing action point for step {i} for {self.query}: {e}")
                                shutil.copy(screenshot_path, os.path.join(task_folder, f'trajectory_{i}_{action}.png'))
                        else:
                            shutil.copy(screenshot_path, os.path.join(task_folder, f'trajectory_{i}_{action}.png'))
                except Exception as e:
                    logger.warning(f"Error when saving image for step {i} for {self.query}: {e}")
                    
        logger.info(f"Saved trajectory to {task_folder}")


    def check_jump_condition(self, parsed_input):
        """检查用户输入是否符合跳转条件，返回(target_node, jump_message, answer_text)"""

        answer_text = ""

        if not parsed_input:
            return None, "Error Format，Please use following format.(waitin for implement)", answer_text

        if parsed_input['action_type'] == 'complete':
            answer_text = 'complete'
            return None, "Complete the task", answer_text
        
        current_node_id = self.trajectory[-1]['screenshot']
        logger.info(f"Current node: {current_node_id}")

        if parsed_input['action_type'] == 'system_button' and parsed_input['button'] == 'back':
            if len(self.history_stack) >= 2:
                self.history_stack.pop()
                previous_page = self.history_stack.pop()
                jump_message = f"返回上一页: {previous_page} (from: {current_node_id})"
                return previous_page, jump_message, answer_text
            else:
                return None, "Already in the home page, cannot go back", answer_text

        if parsed_input['action_type'] == 'system_button' and parsed_input['button'] == 'home':
            return self.home_page, f"Successfully back to home", answer_text

        if current_node_id == self.home_page and parsed_input['action_type'] == 'click':
            tmp_app = self.parse_app_from_click(parsed_input)
            if tmp_app:
                parsed_input = {
                    'action_type': 'open',
                    'app': tmp_app
                }
        if parsed_input['action_type'] == 'open':
            app = APP_MAP.get(parsed_input['app'], parsed_input['app'])
            try:
                app_home = random.choice(self.apps[app])
                return app_home, f"Successfully transform to {app}: {app_home}", answer_text
            except Exception as e:
                logger.info(f"Wrong when open app {app}, abort the task!")
                return current_node_id, f"Failed to transform to {app}: {e}. Stay still", f"Failed to transform to {app}"

        outgoing_edges = self.graph_data.get(current_node_id, {})
        logger.info(f"The number of outgoing edges: {len(outgoing_edges)}")
        
        if not outgoing_edges:

            return current_node_id, f"Current node '{current_node_id}' has no outgoing edges", answer_text
        
        edges = []
        messages = []
        wait_edges = []
        for edge, actions in outgoing_edges.items():
            for action in actions:
                if action.get('action_type', '').lower() == 'wait':
                    wait_edges.append(edge)

                action_type = parsed_input['action_type']
                if action.get('action_type', '').lower() != action_type:
                    continue
                match = False
                
                # 1. click: check position or bbox
                if action_type in ['click','long_press']:
                    if 'bbox' in action and 'x' in parsed_input and 'y' in parsed_input:
                        x1, y1, x2, y2 = action['bbox']
                        if int(x1) <= int(parsed_input['x']) <= int(x2) and int(y1) <= int(parsed_input['y']) <= int(y2):
                            match = True
                            messages.append(f"click match with {action} (in bbox: {action['bbox']})")
                    elif 'x' in action and 'y' in action and 'x' in parsed_input and 'y' in parsed_input:
                        distance = euclidean_distance(
                            parsed_input['x'], parsed_input['y'],
                            action['x'], action['y']
                        )
                        threshold = 1080 * 0.14
                        if distance < threshold:
                            match = True
                            messages.append(f"click match with {action} (距离: {distance:.2f} < 阈值: {threshold:.2f})")
                        else:
                            action['match'] = False
                            logger.info(f"click not match with {action}  (距离: {distance:.2f} >= 阈值: {threshold:.2f})")
                            pass
                
                # 2. swipe: check direction
                elif action_type == 'swipe':
                    if action['direction'] == parsed_input['direction']:
                        match = True
                        messages.append(f"滑动方向匹配: swipe {action['direction']}")

                # 3. input: check text content
                elif action_type == 'type':
                    if 'text' in action and 'text' in parsed_input:
                        action['text'] = action['text'].lower().replace('，', ',').replace('。', ',').replace(' ', '').replace('/',',').replace(':',',').replace('*',',').replace('?',',').replace('"',',').replace('“',',').replace('”',',').replace('<',',').replace('>',',').replace('|',',').lower()  
                        gt_text = action['text'].split(',')
                        vlm_text = parsed_input['text'].lower().replace('，', ',').replace('。', ',').replace(' ', '').replace('/',',').replace(':',',').replace('*',',').replace('?',',').replace('"',',').replace('“',',').replace('”',',').replace('<',',').replace('>',',').replace('|',',').lower()  
                        vlm_text = vlm_text.split(',')  
                        flag = True
                        for text in gt_text:
                            if text not in parsed_input['text']:
                                flag = False
                                break
                        if not flag:
                            flag = True
                            for text in vlm_text:
                                if text not in action['text']:
                                    flag = False
                                    break
                        if flag:
                            match = True
                            messages.append(f"Input text match: {parsed_input['text']}")
                        if parsed_input['text'].lower() in action['text'].lower() or action['text'].lower() in parsed_input['text'].lower():
                            # 如果文字有包含关系
                            match = True
                            messages.append(f"Input text match: {parsed_input['text']}")

                # 4. others: only check action_type
                else:
                    match = True
                    messages.append(f"动作类型匹配: {action_type}")
                
                if match:
                    edges.append(edge)
                    break
        
        if edges:
            n = random.randint(0, len(edges)-1)
            return edges[n], f"跳转成功: {messages[n]}", answer_text

        if wait_edges:
            n = random.randint(0, len(wait_edges)-1)
            return wait_edges[n], f"没有匹配的动作条件，自动选择wait", answer_text

        return current_node_id, f"没有找到匹配的动作条件: {parsed_input}，停留在原地", answer_text


    def step(self, user_input, action_description=None, action_step_info=None):
        """
        执行一步动作，更新图和轨迹
        :param action: 执行的动作
        返回：下一个节点；答案
        """
        if self.trajectory == []:
            logger.info("未设置任务，请先调用 set_task() 方法")
            return None, None

        # global graph_data
        if self.graph_data is None:
            return None, "未加载到JSON数据，无法进行跳转判断", answer_text

        self.trajectory[-1]['action'] = user_input 
        if action_description:
            self.trajectory[-1]['action_description'] = action_description
        if action_step_info:
            for key in action_step_info:
                if key != 'step_number' and key != 'screenshot':
                    self.trajectory[-1][key] = action_step_info[key]

        target_node, jump_message, answer_text = self.check_jump_condition(user_input)

        if answer_text:
            self.trajectory[-1]['answer'] = answer_text
            return None, answer_text  

        elif target_node is not None:
            self.history_stack.append(target_node)
            self.trajectory.append({
                'id': len(self.trajectory),
                'screenshot': target_node
            })
            logger.info(f"跳转到新节点: {target_node}, 跳转信息: {jump_message}")
            return target_node, None
        else:
            logger.info(f"出现错误，当前节点: {self.trajectory[-1]['screenshot']}, 跳转信息: {jump_message}")
            return None, None

    def parse_app_from_click(self, parsed_input):
        for app, bbox in CLICK_APP_MAP.items():
            x1, y1, x2, y2 = bbox
            if int(x1) <= int(parsed_input['x']) <= int(x2) and int(y1) <= int(parsed_input['y']) <= int(y2):
                return app
        return None


def get_qwen_response(query, imagebase64):
    prompt = f"""你需要完成以下任务：{query}
        请分析提供的图片，**图片尺寸为1080*2374**，判断为了完成上述任务在当前应该执行什么操作。
        请根据图片内容，以action_type[param]的格式返回操作指令，例如：
        - 点击操作: click[x,y]（x和y是坐标）
        - 系统按钮: system_button[back] 或 system_button[home]
        如果需要返回回答或与用户交互获取更多信息，可以使用ANSWER[TEXT]格式。
        **注意**，当你判断已经完成任务，请返回ANSWER[COMPLETE]。
        只返回操作指令，不要其他内容。
        
        ### 示例 ### 
        "click[156,2067]###reasoning: 点击全部，进入筛选页面"
        "click[300,620]###reasoning: 点击激活搜索框"
        "wait###reasoning: 等待广告结束"
        "type[where is shenzhen]###reasoning: 在已激活的搜索框中输入内容where is shenzhen"
        "system_button[home]###reasoning: 返回主屏幕"
        "system_button[back]###reasoning: 返回上一页"
        "ANSWER[COMPLETE]###reasoning: 用户任务已全部完成"
    """
            
    msg = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{imagebase64}"}}
            ],
        }
    ]
    
    response = client.chat.completions.create(
        model="qwen2.5-vl-72b-instruct",
        messages=msg,
        temperature=0,
        max_tokens=1024,
    ).choices[0].message.content.strip()

    return response

def extract_before_heading(text):
    parts = text.split("###", 1) 
    if len(parts) > 1:
        return parts[0].strip()  
    return text.strip() 

if __name__=='__main__':
    graph_json_file = ''
    output_dir = ''
    example_task = '在小红书，先查看个人主页，再点击左上角设置按钮，返回个人主页，最后返回推荐首页' 


    client = OpenAI(
        base_url="http://your-api-endpoint/v1",
        api_key="empty",
    )

    graph_dataset = Graph_DataSet(graph_json_file)
    graph_dataset.set_task(example_task)
    complete = False

    image_path = ''
    max_step = 20
    current_step = 0
    while not complete and current_step < max_step:
        image_path = os.path.join('', image_path)
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        response = get_qwen_response(example_task, encoded_string)
        logger.info(f"Qwen2.5-vl-72b-instruct 返回结果: {response}")

        action_str = extract_before_heading(response)
        image_path, answer = graph_dataset.step(action_str)
        if answer:
            logger.info(f"任务结束，回答为: {answer}")
            complete = True
        elif image_path is None:
            logger.info("出现错误，无法继续执行任务")
            complete = True
        current_step += 1

    graph_dataset.save_trajectory(output_dir)
    logger.info(f"任务轨迹已保存到: {os.path.join(output_dir, example_task)}")




