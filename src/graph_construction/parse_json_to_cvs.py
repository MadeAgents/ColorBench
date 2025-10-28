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

import json
import pandas as pd
import numpy as np
import pathlib as Path
import os


json_file = 'xxx.json'
save_file = 'xxx.csv'


def json_to_adjacency_csv(json_file_path, output_csv_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    screenshots = []
    screenshot_to_node = {}  
    
    for node_id, node_info in data['nodes'].items():
        for i, screenshot in enumerate(node_info['screenlists']):
            try:
                screenshot_path = screenshot['screenshot_path']
                screenshots.append({
                    'screenshot': screenshot_path,
                    'node_id': node_id,
                })
            except:
                print('错误出现在：',node_id, screenshot)
    
    print(f"总共找到 {len(screenshots)} 张截图")
    
    n_screenshots = len(screenshots)
    adjacency_matrix = np.full((n_screenshots, n_screenshots), '', dtype=object)
    
    node_id_to_screenshots = {}
    for shot in screenshots:
        node_id = shot['node_id']
        if node_id not in node_id_to_screenshots:
            node_id_to_screenshots[node_id] = []
        node_id_to_screenshots[node_id].append(shot['screenshot'])
    screenshot_id_to_index = {shot['screenshot']: i for i, shot in enumerate(screenshots)}
    
    for node_id, node_info in data['nodes'].items():
        source_screenshots = node_id_to_screenshots.get(node_id, [])
        for edge in node_info['ui_element_edge_list']:
            target_node_id = str(edge['target_node'])
            if target_node_id == '-1':
                continue
            if target_node_id == node_id:
                continue

            action_type = edge['action_type']
            action_param = edge.get('action_parameter', {})
            if action_type.lower() in ['click', 'long_press']: 
                new_action = {
                    "action_type": action_type,
                    "x": action_param['x'],
                    "y": action_param['y']
                }
            elif action_type == 'swipe':
                new_action = {
                    "action_type": action_type,
                    "start": action_param['start'],
                    "lift": action_param['lift']
                }
            elif action_type == 'system_button':
                button = action_param.get('text', '')
                if button=='back':
                    continue 
                new_action = {
                    "action_type": action_type,
                    "button": action_param.get('text', '')
                }
            elif action_type in ['input_text','answer','type']:
                new_action = {
                    "action_type": action_type,
                    "text": action_param.get('text', '')
                }
            elif action_type == 'status':
                new_action = {
                    "action_type": action_type,
                    "status": action_param.get('status', '')
                }
            elif action_type == 'open':
                new_action = {
                    "action_type": action_type,
                    "app": action_param.get('app', '')
                }
            else:
                new_action = {
                    "action_type": action_type
                }
                
            target_screenshots = node_id_to_screenshots.get(target_node_id, [])
            
            for source_screenshot in source_screenshots:
                for target_screenshot in target_screenshots:
                    source_idx = screenshot_id_to_index[source_screenshot]
                    target_idx = screenshot_id_to_index[target_screenshot]
                    existing_action = adjacency_matrix[source_idx, target_idx]
                    if existing_action:
                        flag = False
                        if action_type.lower() in ['click', 'long_press']: 
                            for old_action in existing_action:
                                if old_action['action_type'] == new_action['action_type'] and (abs(old_action.get('x')-new_action.get('x'))<=50) and (abs(old_action.get('y')-new_action.get('y'))<=50):
                                    flag = True 
                                    break
                        if not flag:
                            adjacency_matrix[source_idx, target_idx].append(new_action)
                    else:
                        adjacency_matrix[source_idx, target_idx] = [new_action]
    
    screenshot_labels = [shot['screenshot'] for shot in screenshots]
    df = pd.DataFrame(adjacency_matrix, 
                     index=screenshot_labels, 
                     columns=screenshot_labels).replace('', 0)
                        
    df.to_csv(output_csv_path, encoding='utf-8')

    screenshots_df = pd.DataFrame(screenshots)
    info_csv_path = output_csv_path.replace('.csv', '_screenshot_info.csv')
    screenshots_df.to_csv(info_csv_path, index=False, encoding='utf-8')
    
    print(f"邻接矩阵已保存到: {output_csv_path}")
    print(f"截图信息已保存到: {info_csv_path}")
    print(f"矩阵大小: {n_screenshots} x {n_screenshots}")
    
    non_empty_actions = np.count_nonzero(adjacency_matrix != '')
    print(f"总共有 {non_empty_actions} 个有效的动作连接")
    

if __name__ == "__main__":
    json_to_adjacency_csv(json_file, save_file)