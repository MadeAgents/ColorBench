import pandas as pd
import ast
import json
import os
from collections import defaultdict


def check_matrix(adjacency_matrix:pd.DataFrame):
    deleted_indices = []
    rows = adjacency_matrix.index.tolist()
    for i in range(len(rows)):
        if all(adjacency_matrix.iloc[:, i] == ''):
            print(f"警告: 截图 {rows[i]['screenshot']} 没有入边")
            deleted_indices.append(i)
    for index in sorted(deleted_indices, reverse=True):
        adjacency_matrix = adjacency_matrix.drop(index, axis=0)
        adjacency_matrix = adjacency_matrix.drop(index, axis=1)
    print(f"删除了 {len(deleted_indices)} 个没有入边的截图节点")
    return adjacency_matrix

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
        print("起点和终点相同，无法确定方向")
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
        

def csv_to_json(file_path, output_json_path=None):
    """
    将邻接矩阵CSV文件转换为JSON格式，自动尝试多种编码解决读取问题
    
    参数:
        file_path: CSV文件路径
        output_json_path: 输出JSON文件路径，默认为与CSV同名的JSON文件
    """
    try:
        encodings = [ 'UTF-8', 'GB2312', 'GBK']  # 'ANSI', 'ISO-8859-1'
        df = None
        
        for encoding in encodings:
            try:
                print(f"尝试使用编码 {encoding} 读取文件...")
                df = pd.read_csv(file_path, index_col=0, encoding=encoding, dtype=str)
                print(f"成功使用编码 {encoding} 读取文件，大小为: {df.shape}")
                df = check_matrix(df)
                if df is None:
                    print(f"错误: 编码 {encoding} 读取的文件不符合预期格式")
                    continue
                print(f"成功检查并删除没有入边的节点，检查后大小为: {df.shape}")
                break 
            except UnicodeDecodeError:
                continue  
        
        if df is None:
            print("错误: 尝试所有编码均无法正确读取文件")
            return None
        
        print("开始将CSV转换为JSON格式...")
        json_data = convert_to_json(df)
        print("成功将CSV转换为JSON格式")
        if output_json_path is None:
            output_json_path = os.path.splitext(file_path)[0] + '.json'  
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"已成功将CSV转换为JSON格式并保存到 {output_json_path}")
        
        return json_data
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return None

def convert_to_json(df):
    """将DataFrame格式的邻接矩阵转换为JSON结构"""
    
    try:
        graph = defaultdict(lambda: defaultdict(list))
        all_nodes = set(df.index)
        for col in df.columns:
            all_nodes.add(col)
        # {source_node:{target_node:[actions]}}
        graph = {
            node_id: {} for node_id in all_nodes
        }  
    except Exception as e:
        print(f"初始化图结构时出错: {e}")
        return {}

    # 收集所有边
    self_loop_count = 0
    for source in df.index:
        for target in df.columns:
            value = df.loc[source, target]
            try:
                if pd.notna(value) and str(value).strip() not in ['0', 'nan', 'NaN', '']:
                    if source == target:
                        self_loop_count += 1
                        # print(f"警告: 跳过自环边 {source} -> {target}")
                        continue  # 跳过自环
                    try: 
                        if isinstance(value, str):
                            edges_list = ast.literal_eval(value.strip())
                            if not isinstance(edges_list, list):
                                # print(f"警告: 解析边 {source} -> {target} 时，值不是列表格式，已转换为单元素列表")
                                edges_list = [edges_list]
                        else:
                            edges_list = value if isinstance(value, list) else [value]
      
                        normalized_actions = []
                        for action in edges_list:
                            if action['action_type'] == 'system_button':
                                # print(f"警告: 遇到 system_button 动作 {action} from {source} to {target}, 已跳过")
                                continue
                            if isinstance(action, dict):
                                if action['action_type'] not in ['click', 'long_press','open','wait','swipe','input_text','type','input']:
                                    print(f"警告: 遇到不支持的动作 {action} from {source} to {target}, 已转换为通用格式")
                                    normalized_actions.append({"action_type": "custom", "value": str(action)})

                                if action['action_type'] == 'input_text' or action['action_type'] == 'input':
                                    action['action_type'] = 'type' 
                                elif action['action_type'] == 'click':
                                    # If there are other click actions that are within 50 pixels, we won't add this one.
                                    if 'x' in action and 'y' in action:
                                        x, y = action['x'], action['y']
                                        too_close = False
                                        for existing_action in normalized_actions:
                                            if existing_action['action_type'] == 'click' and 'x' in existing_action and 'y' in existing_action:
                                                ex, ey = existing_action['x'], existing_action['y']
                                                if abs(ex - x) <= 20 and abs(ey - y) <= 20:
                                                    too_close = True
                                                    break
                                        if too_close:
                                            print(f"警告: 跳过相近的 click 动作 {action} from {source} to {target}")
                                            continue
                                elif action['action_type'] == 'swipe':
                                    if 'direction' not in action:
                                        try:
                                            action['direction'] = position_to_direction(action['x1'], action['y1'], action['x2'], action['y2'])
                                            print(f"警告: swipe 动作缺少 direction 字段，已根据坐标计算 direction 为 {action['direction']}")
                                        except Exception as e:
                                            print(f"计算 {source} to {target} 的坐标时direction: {e}")
                                            action['direction'] = 'unknown'
                                normalized_actions.append(action)
                            else:
                                print(f"警告: 错误的动作 {action}，已转换为通用格式")
                                normalized_actions.append({"action_type": "custom", "value": str(action)})
                            
                        graph[source][target] = normalized_actions

                    except (SyntaxError, ValueError, TypeError) as e:
                        print(f"转换为JSON时解析值出错 {value}: {e}")
                        graph[source][target] = [{"action_type": "error", "value": str(value)}]
            except Exception as e:
                print(f"处理边 {source} -> {target} 时出错: {e}")
                print(f"出错的值: {value}")
                continue


    
    print(f"总共跳过了 {self_loop_count} 条自环边")
    return graph

if __name__ == "__main__":
    # 输入CSV文件路径
    csv_file_path = ""
    output_json_path = ""
    
    json_data = csv_to_json(csv_file_path, output_json_path)
