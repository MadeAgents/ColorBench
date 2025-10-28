import os
import json
import time
import logging
import colorlog
from pathlib import Path
import yaml



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
                        if existing_action == action:
                            return True
        elif type(existing_action) == list and type(action) == list:
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

            trajectory_file = os.path.join(task_folder, 'trajectory.json')
            if not os.path.exists(trajectory_file):
                logger.warning(f"Trajectory file {trajectory_file} does not exist, skipping...")
                continue
            with open(trajectory_file, 'r', encoding='utf-8') as f:
                trajectory = json.load(f)
            if subdir.startswith("tasks"):
                trajectory = trajectory.get('trajectory', [])

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

    output_file = "screenshot_action_counts.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(screenshot_counts, f, ensure_ascii=False, indent=4)
    logger.info(f"Action counts per screenshot saved to {output_file}")
    
            
    preference_counts = list(screenshot_counts.values())

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