# 更换思路，使用已经实验过的任务轨迹，统计模型在特定任务上的每一步可能的动作数
from openai import OpenAI
import os
import json
import time
import logging
import colorlog
from pathlib import Path
from dotenv import load_dotenv
import yaml
from zai import ZhipuAiClient



def setup_logging(log_file_path):
    """配置日志系统"""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'
        }
    ))
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel("INFO")
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

model = "glm"
api_key = "empty"
base_url = 'your_base_url'
log_file_path = f'./check_number_from_trajectory.log'
setup_logging(log_file_path)
logger = logging.getLogger(__name__)
MAX_RETRIES = 3

def get_glm_response(messages, temperature=0.1, top_k=5, top_p=0.9):

    client = ZhipuAiClient(api_key="a29023ba80c242d8ad34d6b98e4c1cae.ktWzqwnAUsFMaIsA")
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

def get_response(model, messages, api_key, base_url, temperature=0.1):
    """Get response from LLM with retry mechanism"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    retries = 0
    retry_delay = 2
    
    while retries <= MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
            ).choices[0].message.content.strip()
            return response
        except Exception as e:
            logger.warning(f"Request failed, retrying... Error: {str(e)}")
            retries += 1
            time.sleep(retry_delay)
    
    logger.error("Request failed after multiple retries.")
    return None

def caculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0

    return iou

def check(action, actions):
    # 检查atcion是否已经在actions中
    for existing_action in actions:
        if type(existing_action) == dict and type(action) == dict:
            if 'action_type' in existing_action and 'action_type' in action:
                if existing_action['action_type'] == action['action_type']:
                    if existing_action['action_type'] in ['click', 'long_click']:
                        if 'x' in existing_action and 'y' in existing_action and 'x' in action and 'y' in action:
                            ex, ey = int(existing_action['x']), int(existing_action['y'])
                            x, y = int(action['x']), int(action['y'])
                            if abs(ex - x) <= 100 and abs(ey - y) <= 100:
                                return True
                    elif existing_action['action_type'] == 'swipe':
                        if 'direction' in existing_action and 'direction' in action:
                            if existing_action['direction'] == action['direction']:
                                return True
                    else:
                        # 其他动作类型，直接比较整个字典
                        if existing_action == action:
                            return True
        elif type(existing_action) == list and type(action) == list:
            # 都是bbox的情况，计算iou
            if caculate_iou(existing_action, action) > 0.9:
                return True

def main():
    logger.info("Progress Start!")

    logger.info(f"Using model: {model}")
    checkpoint_path = "./checkpoints"
    image_folder = "./final_graph_images_919"
    graph_json = "./final_graph_0914.json"
    with open(graph_json, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    # 记录graph_data中每个节点的bbox
    node_bboxes = {}
    for source,targets in graph_data.items():
        node_bboxes[source] = []
        for target,relations in targets.items():
            for action in relations:
                if 'bbox' in action and action['bbox'] not in node_bboxes[source]:
                    node_bboxes[source].append(action['bbox'])
    logger.info("获取到每个节点的bbox信息成功")
    logger.info(f"有最多的bbox的节点是有{max([len(bboxes) for bboxes in node_bboxes.values()])}个bbox")
    
    all_subdirs = [d for d in os.listdir(checkpoint_path) if os.path.isdir(os.path.join(checkpoint_path, d))]
    counts_dict = {}  
    for subdir in all_subdirs:
        subdir_path = os.path.join(checkpoint_path, subdir)
        all_tasks = [d for d in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, d))]
        for task in all_tasks:
            if task not in counts_dict:
                counts_dict[task] = {}
            if subdir.startswith("tasks"):
                task_folder = os.path.join(subdir_path, task.replace('/','_').replace(':','_').replace('*','_').replace('?','_').replace('"','_').replace('<','_').replace('>','_').replace('|','_'))
            else:
                task_folder = os.path.join(subdir_path,task.replace('/','_'))
            
            if not os.path.exists(task_folder):
                logger.warning(f"Task folder {task_folder} does not exist, skipping...")
                continue
            # 读取trajectory文件
            trajectory_file = os.path.join(task_folder, 'trajectory.json')
            if not os.path.exists(trajectory_file):
                logger.warning(f"Trajectory file {trajectory_file} does not exist, skipping...")
                continue
            with open(trajectory_file, 'r', encoding='utf-8') as f:
                trajectory = json.load(f)
            if subdir.startswith("tasks"):
                trajectory = trajectory.get('trajectory', [])

            # 遍历轨迹的每一步
            for step in trajectory:
                screenshot = step.get('screenshot', None)
                if screenshot is None:
                    continue
                if screenshot not in counts_dict[task]:
                    counts_dict[task][screenshot] = []  
                action = step.get('action', None)
                if action is None:
                    continue
                if action['action_type'] == 'click' or action['action_type'] == 'long_click':
                    if 'x' in action and 'y' in action:
                        x, y = int(action['x']), int(action['y'])
                        bboxes = node_bboxes.get(screenshot, set())
                        
                        if bboxes == set():
                            flag = False
                            for existing_action in counts_dict[task][screenshot]:
                                if 'action_type' in existing_action and existing_action['action_type'] in ['click', 'long_click'] and 'x' in existing_action and 'y' in existing_action:
                                    ex, ey = int(existing_action['x']), int(existing_action['y'])
                                    if abs(ex - x) <= 100 and abs(ey - y) <= 100:
                                        flag=True
                                        break
                            if not flag:
                                counts_dict[task][screenshot].append(action)
                            continue

                        add_flag = False
                        for bbox in bboxes:
                            x1, y1, x2, y2 = bbox  
                            if int(x1) <= x <= int(x2) and int(y1) <= y <= int(y2):
                                add_flag = True
                                bbox_flag = False
                                for existing_action in counts_dict[task][screenshot]:
                                    if type(existing_action) == list:
                                        if caculate_iou(existing_action, bbox) > 0.9:
                                            bbox_flag = True
                                            break
                                if not bbox_flag:
                                    counts_dict[task][screenshot].append(bbox)
                                    break
                        if not add_flag:
                            flag = False
                            for existing_action in counts_dict[task][screenshot]:
                                if 'action_type' in existing_action and existing_action['action_type'] in ['click', 'long_click'] and 'x' in existing_action and 'y' in existing_action:
                                    ex, ey = int(existing_action['x']), int(existing_action['y'])
                                    # 欧氏距离小于100认为是重复
                                    if (abs(ex - x) <= 150 and abs(ey - y) <= 150) :
                                        flag=True
                                        break
                            if not flag:
                                counts_dict[task][screenshot].append(action)

                elif action['action_type'] == 'type' or action['action_type'] == 'open' or action['action_type'] == 'system_button' or action['action_type'] == 'complete' or action['action_type'] == 'wait':
                    continue  
                elif action['action_type'] == 'swipe':
                    # direction = action.get('direction', None)
                    for existing_action in counts_dict[task][screenshot]:
                        if 'action_type' in existing_action and existing_action['action_type'] == 'swipe' and 'direction' in existing_action and 'direction' in action:
                            if existing_action['direction'] == action['direction']:
                                break
                else:
                    if action not in counts_dict[task][screenshot]:
                        counts_dict[task][screenshot].append(action)


    screenshot_actions = {}
    screenshot_counts = {}  # screen:counts
    for task, screens in counts_dict.items():
        for screen, actions in screens.items():
            if screen not in screenshot_counts:
                screenshot_actions[screen] = actions
            else:
                for action in actions:
                    flag = check(action, actions)
                    if not flag:
                        screenshot_actions[screen].append(action)
    
    for screen, actions in screenshot_actions.items():
        screenshot_counts[screen] = len(actions)

    # 把结果存入json文件
    output_file = "screenshot_action_counts.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(screenshot_counts, f, ensure_ascii=False, indent=4)
    logger.info(f"Action counts per screenshot saved to {output_file}")
    
            
    preference_counts = list(screenshot_counts.values())
    # 输出统计结果
    output_csv = "screenshot_preference_counts.csv"
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("screenshot,preference_count\n")
        for screen, count in screenshot_counts.items():
            f.write(f"{screen},{count}\n")
    logger.info(f"Preference counts per screenshot saved to {output_csv}")

    logger.info(f"Total images processed: {len(preference_counts)}")
    logger.info(f"Max preference counts: {max(preference_counts) if preference_counts else 0}")

    max_screens = [screen for screen, count in screenshot_counts.items() if count == max(preference_counts)]
    logger.info(f"Max preference screens: {max_screens}")
    logger.info(f"这个最大的屏幕对应的动作有: {screenshot_actions[max_screens[0]]}")
    logger.info(f"Min preference counts: {min(preference_counts) if preference_counts else 0}")

    zero_count = sum(1 for count in preference_counts if count == 0)
    logger.info(f"Number of screens with zero preference counts: {zero_count}")
    logger.info(f"Average preference counts: {sum(preference_counts)/len(preference_counts) if preference_counts else 0}")


if __name__ == "__main__":
    main()