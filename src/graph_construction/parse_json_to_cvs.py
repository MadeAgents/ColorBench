import json
import pandas as pd
import numpy as np
import pathlib as Path
import os


json_file = '/home/notebook/code/personal/S9060045/demonstration_based_learning/data/results/redbook.json'
save_file = '/home/notebook/code/personal/S9060045/demonstration_based_learning/data/results/csv/redbook_adjacency_matrix.csv'


def json_to_adjacency_csv(json_file_path, output_csv_path):
    # 提取所需字段
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取所有截图信息
    screenshots = []
    screenshot_to_node = {}  # 截图路径到节点ID的映射
    
    # 遍历所有节点，收集截图信息
    for node_id, node_info in data['nodes'].items():
        for i, screenshot in enumerate(node_info['screenlists']):
            try:
                screenshot_path = screenshot['screenshot_path']
                # 创建唯一的截图标识符
                screenshots.append({
                    'screenshot': screenshot_path,
                    'node_id': node_id,
                })
            except:
                print('错误出现在：',node_id, screenshot)
    
    print(f"总共找到 {len(screenshots)} 张截图")
    
    # 创建邻接矩阵
    n_screenshots = len(screenshots)
    adjacency_matrix = np.full((n_screenshots, n_screenshots), '', dtype=object)
    
    # 建立节点ID到其截图的映射
    node_id_to_screenshots = {}
    for shot in screenshots:
        node_id = shot['node_id']
        if node_id not in node_id_to_screenshots:
            node_id_to_screenshots[node_id] = []
        node_id_to_screenshots[node_id].append(shot['screenshot'])
    # 创建截图ID到表格索引的映射
    screenshot_id_to_index = {shot['screenshot']: i for i, shot in enumerate(screenshots)}
    
    
    # 填充邻接矩阵
    for node_id, node_info in data['nodes'].items():
        source_screenshots = node_id_to_screenshots.get(node_id, [])
        
        # 遍历该节点的所有边
        for edge in node_info['ui_element_edge_list']:
            target_node_id = str(edge['target_node'])
            # 如果目标节点是-1，跳过
            if target_node_id == '-1':
                continue
            if target_node_id == node_id:
                continue

            action_type = edge['action_type']
            action_param = edge.get('action_parameter', {})
            # [{"action_type": "click", "x": 292, "y": 510}, {"action_type": "click", "x": 292, "y": 316}, {"action_type": "click", "x": 293, "y": 704}
            # 构建动作描述
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
                    continue  # 如果是返回键，跳过
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
                
            # 获取目标节点的截图索引
            target_screenshots = node_id_to_screenshots.get(target_node_id, [])
            
            # 从源节点的所有截图到目标节点的所有截图都添加这个动作
            for source_screenshot in source_screenshots:
                for target_screenshot in target_screenshots:
                    # 如果已经有动作了，用分号分隔多个动作
                    source_idx = screenshot_id_to_index[source_screenshot]
                    target_idx = screenshot_id_to_index[target_screenshot]
                    # 返回的是一个字典列表
                    existing_action = adjacency_matrix[source_idx, target_idx]
                    if existing_action:
                        # 如果已经有了，检查一下click有没有重复的阈值给到50
                        flag = False
                        if action_type.lower() in ['click', 'long_press']: 
                            for old_action in existing_action:
                                if old_action['action_type'] == new_action['action_type'] and (abs(old_action.get('x')-new_action.get('x'))<=50) and (abs(old_action.get('y')-new_action.get('y'))<=50):
                                    flag = True  # 重复了
                                    break
                        if not flag:
                            adjacency_matrix[source_idx, target_idx].append(new_action)
                    else:
                        adjacency_matrix[source_idx, target_idx] = [new_action]
    
    # 创建DataFrame
    screenshot_labels = [shot['screenshot'] for shot in screenshots]
    # 没有值的地方填0
    df = pd.DataFrame(adjacency_matrix, 
                     index=screenshot_labels, 
                     columns=screenshot_labels).replace('', 0)
                        
    # 保存为CSV
    df.to_csv(output_csv_path, encoding='utf-8')
    
    # 创建截图信息表
    screenshots_df = pd.DataFrame(screenshots)
    info_csv_path = output_csv_path.replace('.csv', '_screenshot_info.csv')
    screenshots_df.to_csv(info_csv_path, index=False, encoding='utf-8')
    
    print(f"邻接矩阵已保存到: {output_csv_path}")
    print(f"截图信息已保存到: {info_csv_path}")
    print(f"矩阵大小: {n_screenshots} x {n_screenshots}")
    
    # 统计非空动作数量
    non_empty_actions = np.count_nonzero(adjacency_matrix != '')
    print(f"总共有 {non_empty_actions} 个有效的动作连接")
    

if __name__ == "__main__":
    json_to_adjacency_csv(json_file, save_file)