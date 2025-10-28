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

import gradio as gr
import pandas as pd
import os
from PIL import Image
import re
import ast
from datetime import datetime

BASE_RECORD_PATH = os.path.join(".", "graph_images")
os.makedirs(BASE_RECORD_PATH, exist_ok=True)


def resize_image(img, scale_factor=0.3):
    if img is None:
        return None
        
    width, height = img.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    min_dimension = 50
    new_width = max(new_width, min_dimension)
    new_height = max(new_height, min_dimension)
    
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def get_image_path(node_name):
    if not node_name:
        print("节点名称为空，无法获取图片路径")
        return None

    possible_paths = [os.path.join(BASE_RECORD_PATH, node_name)]
    
    if not os.path.splitext(node_name)[1].lower() in ['.png', '.jpg', '.jpeg']:
        possible_paths.extend([
            os.path.join(BASE_RECORD_PATH, f"{node_name}.png"),
            os.path.join(BASE_RECORD_PATH, f"{node_name}.jpg"),
            os.path.join(BASE_RECORD_PATH, f"{node_name}.jpeg")
        ])
    
    possible_paths.append(node_name)
    if not os.path.splitext(node_name)[1].lower() in ['0', '.png', '.jpg', '.jpeg']:
        possible_paths.extend([
            f"{node_name}.png",
            f"{node_name}.jpg",
            f"{node_name}.jpeg"
        ])
    
    for path in possible_paths:
        normalized_path = os.path.normpath(path)
        abs_path = os.path.abspath(normalized_path)
        
        if os.path.exists(abs_path) and os.path.isfile(abs_path):
            return abs_path
    
    return None

def load_image(node_name):
    if not node_name:
        return None
        
    img_path = get_image_path(node_name)
    if img_path:
        try:
            img = Image.open(img_path)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                background.paste(img, img.split()[-1])
                img = background
            return resize_image(img, scale_factor=0.3)
        except Exception as e:
            print(f"加载图片时出错 {img_path}: {str(e)}")
    return None

def load_adjacency_matrix(file_path):
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'iso-8859-1', 'utf-16']

    if not file_path.startswith(BASE_RECORD_PATH):
        file_name = os.path.basename(file_path)
        print(f"文件名: {file_name}")
        unified_path = os.path.join(BASE_RECORD_PATH, file_name)
        print(f"尝试从统一路径加载文件: {unified_path}")
        if os.path.exists(unified_path):
            file_path = unified_path
    
    try:
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    file_path, 
                    index_col=0, 
                    encoding=encoding,
                    dtype=str
                )
                if not df.columns.equals(df.index):
                    return None, "邻接矩阵的行名和列名不一致，请检查文件格式"
                return df, f"成功使用编码 {encoding} 加载文件"
            except UnicodeDecodeError:
                continue
        
        return None, "所有尝试的编码格式都无法正确解码文件，请检查文件编码"
        
    except FileNotFoundError:
        return None, f"错误：找不到文件 {file_path}"
    except Exception as e:
        return None, f"加载文件时发生错误：{str(e)}"


def get_node_relations(matrix, node_name):
    if node_name not in matrix.index:
        return None, None, f"错误：节点 '{node_name}' 不存在于矩阵中"
    
    def is_zero(value):
        if pd.isna(value) or value is None:
            return True
        str_val = str(value).strip().lower()
        return str_val in ['0', '', 'none', 'nan', 'null']
    
    pointing_to = {}
    for col in matrix.columns:
        value = matrix.loc[node_name, col]
        if not is_zero(value):
            pointing_to[col] = value.strip()
    
    pointed_by = {}
    for row in matrix.index:
        value = matrix.loc[row, node_name]
        if not is_zero(value):
            pointed_by[row] = value.strip()
    
    return pointing_to, pointed_by, None


def get_edges_between_nodes(matrix, node1, node2):
    edges = []
    edge_details = []
    
    if node1 not in matrix.index or node2 not in matrix.index:
        return edges, edge_details, "节点不存在于邻接矩阵中"
    
    value1 = matrix.loc[node1, node2]
    if pd.notna(value1) and str(value1).strip().lower() not in ['0', '', 'none', 'nan', 'null']:
        try:
            edge_data = ast.literal_eval(str(value1).strip())
            if isinstance(edge_data, list):
                for i, edge in enumerate(edge_data):
                    edges.append(f"{node1} -> {node2}: 边 {i+1}")
                    edge_details.append(str(edge))
            else:
                edges.append(f"{node1} -> {node2}: {value1.strip()}")
                edge_details.append(value1.strip())
        except:
            edges.append(f"{node1} -> {node2}: {value1.strip()}")
            edge_details.append(value1.strip())
    
    return edges, edge_details, None if edges else "两个节点之间没有边"


def process_query(file, node_name):
    if file is None:
        return "请先上传邻接矩阵CSV文件", "", "", None, [], [], [], [], [], []
    
    if file is not None:
        file_name = os.path.basename(file.name)
        unified_file_path = os.path.join(BASE_RECORD_PATH, file_name)
        
        if not os.path.exists(unified_file_path):
            with open(unified_file_path, 'wb') as f_out:
                f_out.write(file.read())
            file.name = unified_file_path
    
    matrix, message = load_adjacency_matrix(file.name)
    if matrix is None:
        return message, "", "", None, [], [], [], [], [], []
    
    node = node_name.strip()
    if not node:
        return "请输入节点名称", "", "", None, [], [], [], [], [], []
    
    pointing_to, pointed_by, error = get_node_relations(matrix, node)
    if error:
        return error, "", "", None, [], [], [], [], [], []
    
    current_node_img = load_image(node)
    
    if current_node_img is None:
        status_msg = f"成功查询节点 '{node}' 的关系，但未找到该节点的图片"
    else:
        status_msg = f"成功查询节点 '{node}' 的关系及图片"
    
    upstream_gallery_data = []
    upstream_full_data = []
    upstream_info = []
    for i, (n, condition) in enumerate(pointed_by.items()):
        img = load_image(n)
        if img is not None:
            info_text = f"{n} [{condition}] (索引: {i})"
            upstream_gallery_data.append((img, f"图片 {i}"))
            upstream_full_data.append((img, info_text, n, i))
            upstream_info.append(info_text)
    
    downstream_gallery_data = []
    downstream_full_data = []
    downstream_info = []
    for i, (n, condition) in enumerate(pointing_to.items()):
        img = load_image(n)
        if img is not None:
            info_text = f"{n} [{condition}] (索引: {i})"
            downstream_gallery_data.append((img, f"图片 {i}"))
            downstream_full_data.append((img, info_text, n, i))
            downstream_info.append(info_text)
    
    pointing_str = ", ".join([f"{k}({v})" for k, v in pointing_to.items()]) if pointing_to else "无"
    pointed_by_str = ", ".join([f"{k}({v})" for k, v in pointed_by.items()]) if pointed_by else "无"
    
    return (status_msg, 
            pointing_str, 
            pointed_by_str,
            current_node_img,
            upstream_gallery_data,
            downstream_gallery_data,
            upstream_full_data,
            downstream_full_data,
            "\n".join(upstream_info),
            "\n".join(downstream_info)
           )


def select_upstream_image(upstream_full_data, index, selected_indices):
    if index is None or index < 0:
        return selected_indices
    
    valid_indices = [item[3] for item in upstream_full_data]
    if index not in valid_indices:
        return selected_indices
    
    if index in selected_indices:
        return [i for i in selected_indices if i != index]
    else:
        return selected_indices + [index]

def select_downstream_image(downstream_full_data, index, selected_indices):
    if index is None or index < 0:
        return selected_indices
    
    valid_indices = [item[3] for item in downstream_full_data]
    if index not in valid_indices:
        return selected_indices
    
    if index in selected_indices:
        return [i for i in selected_indices if i != index]
    else:
        return selected_indices + [index]


def add_manual_image(manual_image_input, upstream_full_data, downstream_full_data, 
                     upstream_selected, downstream_selected, manual_selected):
    if not manual_image_input.strip():
        return manual_selected, "请输入图片名称或路径"
    
    image_name = manual_image_input.strip()
    img_path = get_image_path(image_name)
    if not img_path:
        return manual_selected, f"图片 '{image_name}' 不存在，请检查路径是否正确"
    
    all_auto_selected = []
    for idx in upstream_selected:
        for item in upstream_full_data:
            if item[3] == idx:
                all_auto_selected.append(item[2])
    for idx in downstream_selected:
        for item in downstream_full_data:
            if item[3] == idx:
                all_auto_selected.append(item[2])
    
    if image_name in all_auto_selected:
        return manual_selected, f"图片 '{image_name}' 已在自动选中列表中"
    
    if image_name in manual_selected:
        new_manual = [img for img in manual_selected if img != image_name]
        return new_manual, f"已从手动选中列表中移除: {image_name}"
    else:
        new_manual = manual_selected + [image_name]
        return new_manual, f"已添加到手动选中列表: {image_name}"


def get_selected_names(upstream_full_data, downstream_full_data, 
                      upstream_selected, downstream_selected, manual_selected):
    selected_names = []
    
    for idx in upstream_selected:
        for item in upstream_full_data:
            if item[3] == idx:
                selected_names.append(item[2])
    
    for idx in downstream_selected:
        for item in downstream_full_data:
            if item[3] == idx:
                selected_names.append(item[2])
    
    selected_names.extend(manual_selected)
    
    if not selected_names:
        return "未选中任何图片"
    
    return "选中的图片名称：\n" + "\n".join(selected_names)


def update_selected_images(upstream_full_data, downstream_full_data, 
                          upstream_selected, downstream_selected, manual_selected):
    selected_names = []
    
    for idx in upstream_selected:
        for item in upstream_full_data:
            if item[3] == idx:
                selected_names.append(item[2])
    
    for idx in downstream_selected:
        for item in downstream_full_data:
            if item[3] == idx:
                selected_names.append(item[2])
    
    selected_names.extend(manual_selected)
    
    if not selected_names:
        return []
    
    images = []
    for name in selected_names:
        img = load_image(name)
        if img:
            images.append((img, name))
    
    return images


def get_available_nodes(file):
    if file is None:
        return "请先上传文件以查看可用节点"
    
    file_name = os.path.basename(file.name)
    print(f"文件名: {file_name}")
    unified_file_path = os.path.join(BASE_RECORD_PATH, file_name)
    print(f"尝试从统一路径加载文件: {unified_file_path}")
    if os.path.exists(unified_file_path):
        file.name = unified_file_path
    
    matrix, message = load_adjacency_matrix(file.name)
    if matrix is None:
        print(message)
        return "无法加载文件，无法获取节点列表"
    
    return "可用节点：\n" + ", ".join(matrix.index.tolist())


def save_with_backup(matrix, original_file_name):
    try:
        os.makedirs(BASE_RECORD_PATH, exist_ok=True)
        backup_dir = os.path.join(BASE_RECORD_PATH, "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        file_name = os.path.basename(original_file_name)
        name, ext = os.path.splitext(file_name)
        
        main_path = os.path.join(BASE_RECORD_PATH, file_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"{name}_{timestamp}{ext}")
        
        matrix.to_csv(main_path, encoding='utf-8-sig')  
        matrix.to_csv(backup_path, encoding='utf-8-sig')
        
        return main_path, backup_path
        
    except Exception as e:
        raise Exception(f"保存文件时出错: {str(e)}")

def merge_edges(edge_strings):
    merged = []
    for edge_str in edge_strings:
        try:
            edge_data = ast.literal_eval(edge_str)
            if isinstance(edge_data, list):
                merged.extend(edge_data)
            elif isinstance(edge_data, dict):
                merged.append(edge_data)
            else:
                merged.append(edge_str)
        except:
            merged.append(edge_str)
    
    unique_edges = []
    seen = set()
    for edge in merged:
        edge_str = str(edge)
        if edge_str not in seen:
            seen.add(edge_str)
            unique_edges.append(edge)
    
    return unique_edges


def merge_nodes(file, upstream_full_data, downstream_full_data, 
               upstream_selected, downstream_selected, manual_selected):
    if file is None:
        return "请先上传邻接矩阵CSV文件", None, [], [], []
    
    file_name = os.path.basename(file.name)
    unified_file_path = os.path.join(BASE_RECORD_PATH, file_name)
    
    if not os.path.exists(unified_file_path):
        with open(unified_file_path, 'wb') as f_out:
            f_out.write(file.read())
    
    selected_names = []
    for idx in upstream_selected:
        for item in upstream_full_data:
            if item[3] == idx:
                selected_names.append(item[2])
    
    for idx in downstream_selected:
        for item in downstream_full_data:
            if item[3] == idx:
                selected_names.append(item[2])
    
    selected_names.extend(manual_selected)
    
    seen = set()
    nodes_to_merge = []
    for name in selected_names:
        if name not in seen:
            seen.add(name)
            nodes_to_merge.append(name)
    
    if len(nodes_to_merge) < 2:
        return "至少需要选择两个节点进行合并", None, [], [], []
    
    new_node_name = nodes_to_merge[0]
    matrix, message = load_adjacency_matrix(unified_file_path)
    if matrix is None:
        return message, None, [], [], []
    
    for node in nodes_to_merge:
        if node not in matrix.index:
            return f"节点 '{node}' 不存在于邻接矩阵中", None, [], [], []
    
    new_matrix = matrix.copy()
    pointing_to_merged = {}
    for merged_node in nodes_to_merge:
        for node in new_matrix.index:
            if node in nodes_to_merge:
                continue
            value = new_matrix.loc[node, merged_node]
            if pd.notna(value) and str(value).strip().lower() not in ['0', '', 'none', 'nan', 'null']:
                if node not in pointing_to_merged:
                    pointing_to_merged[node] = []
                pointing_to_merged[node].append(value.strip())
    
    merged_pointing_to = {}
    for merged_node in nodes_to_merge:
        for node in new_matrix.columns:
            if node in nodes_to_merge:
                continue
            value = new_matrix.loc[merged_node, node]
            if pd.notna(value) and str(value).strip().lower() not in ['0', '', 'none', 'nan', 'null']:
                if node not in merged_pointing_to:
                    merged_pointing_to[node] = []
                merged_pointing_to[node].append(value.strip())
    
    nodes_to_drop = [node for node in nodes_to_merge if node != new_node_name]
    new_matrix = new_matrix.drop(index=nodes_to_drop, columns=nodes_to_drop, errors='ignore')
    
    for node, values in pointing_to_merged.items():
        merged_edges = merge_edges(values)
        new_value = str(merged_edges)
        new_matrix.loc[node, new_node_name] = new_value
    
    for node, values in merged_pointing_to.items():
        merged_edges = merge_edges(values)
        new_value = str(merged_edges)
        new_matrix.loc[new_node_name, node] = new_value
    
    internal_relations = []
    for from_node in nodes_to_merge:
        for to_node in nodes_to_merge:
            if from_node != to_node:
                value = matrix.loc[from_node, to_node]
                if pd.notna(value) and str(value).strip().lower() not in ['0', '', 'none', 'nan', 'null']:
                    internal_relations.append(value.strip())
    
    if internal_relations:
        merged_internal_edges = merge_edges(internal_relations)
        new_matrix.loc[new_node_name, new_node_name] = str(merged_internal_edges)
    
    try:
        main_path, backup_path = save_with_backup(new_matrix, file_name)
        
        result_msg = (f"成功合并节点！\n"
                      f"被合并的节点: {', '.join(nodes_to_merge)}\n"
                      f"新节点名称: {new_node_name}\n"
                      f"已更新文件: {os.path.abspath(main_path)}\n"
                      f"备份文件已保存到: {os.path.abspath(backup_path)}")
        return result_msg, main_path, [], [], []
    except Exception as e:
        return f"保存文件时出错: {str(e)}", None, [], [], []


def delete_node(file, node_to_delete):
    if file is None:
        return "请先上传邻接矩阵CSV文件", None
    
    file_name = os.path.basename(file.name)
    unified_file_path = os.path.join(BASE_RECORD_PATH, file_name)
    
    if not os.path.exists(unified_file_path):
        with open(unified_file_path, 'wb') as f_out:
            f_out.write(file.read())
    
    if not node_to_delete.strip():
        return "请输入要删除的节点名称", None
    
    node_to_delete = node_to_delete.strip()
    matrix, message = load_adjacency_matrix(unified_file_path)
    if matrix is None:
        return message, None
    
    if node_to_delete not in matrix.index:
        return f"节点 '{node_to_delete}' 不存在于邻接矩阵中", None
    
    new_matrix = matrix.drop(index=node_to_delete, columns=node_to_delete, errors='ignore')
    
    try:
        main_path, backup_path = save_with_backup(new_matrix, file_name)
        
        result_msg = (f"成功删除节点！\n"
                      f"已删除节点: {node_to_delete}\n"
                      f"已更新文件: {os.path.abspath(main_path)}\n"
                      f"备份文件已保存到: {os.path.abspath(backup_path)}")
        return result_msg, main_path
    except Exception as e:
        return f"保存文件时出错: {str(e)}", None


def add_new_node(file, new_node_name, upstream_relations, downstream_relations):
    """
    新增节点功能（智能解析版）
    支持边信息包含逗号（如字典/列表），仅分割真正的关系间隔
    """
    if file is None:
        return "请先上传邻接矩阵CSV文件", None
    
    new_node = new_node_name.strip()
    if not new_node:
        return "请输入新节点的名称", None
    
    file_name = os.path.basename(file.name)
    unified_file_path = os.path.join(BASE_RECORD_PATH, file_name)
    if not os.path.exists(unified_file_path):
        with open(unified_file_path, 'wb') as f_out:
            f_out.write(file.read())
    

    def smart_split_relations(input_str):
        """
        分割格式为"节点1:边信息1,节点2:边信息2"的字符串
        自动忽略边信息（如字典、列表）内部的逗号
        """
        if not input_str.strip():
            return []
            
        parts = []
        current_part = []
        bracket_level = 0  
        quote_state = None  
        
        for c in input_str:
            if c in ['"', "'"]:
                if quote_state == c:
                    quote_state = None  
                elif quote_state is None:
                    quote_state = c 
                current_part.append(c)
                continue
            
            if quote_state is None:  
                if c in ['{', '[']:
                    bracket_level += 1
                elif c in ['}', ']']:
                    bracket_level = max(0, bracket_level - 1)

            if c == ',' and quote_state is None and bracket_level == 0:
                parts.append(''.join(current_part).strip())
                current_part = []
            else:
                current_part.append(c)

        if current_part:
            parts.append(''.join(current_part).strip())

        return [p for p in parts if p]

    upstream = {}
    if upstream_relations.strip():
        relations = smart_split_relations(upstream_relations)
        for rel in relations:
            colon_positions = [i for i, c in enumerate(rel) if c == ':']
            if not colon_positions:
                return f"上游关系格式错误：'{rel}' 缺少冒号，正确格式应为'节点:边信息'", None
            
            first_colon = colon_positions[0]
            node = rel[:first_colon].strip()
            edge = rel[first_colon+1:].strip()
            
            upstream[node] = edge
    
    downstream = {}
    if downstream_relations.strip():
        relations = smart_split_relations(downstream_relations)
        for rel in relations:
            colon_positions = [i for i, c in enumerate(rel) if c == ':']
            if not colon_positions:
                return f"下游关系格式错误：'{rel}' 缺少冒号，正确格式应为'节点:边信息'", None
            
            first_colon = colon_positions[0]
            node = rel[:first_colon].strip()
            edge = rel[first_colon+1:].strip()
            
            downstream[node] = edge
    
    if not upstream and not downstream:
        return "上游节点和下游节点至少需要输入一个", None
    
    matrix, message = load_adjacency_matrix(unified_file_path)
    if matrix is None:
        return message, None
    
    if new_node in matrix.index:
        return f"节点 '{new_node}' 已存在，请使用不同的名称", None
    
    all_nodes = set(matrix.index)
    for node in upstream:
        if node not in all_nodes:
            return (f"上游节点 '{node}' 不存在于邻接矩阵中！\n"
                    f"请检查节点名称是否与矩阵中的一致（注意路径分隔符、大小写）", None)
    for node in downstream:
        if node not in all_nodes:
            return (f"下游节点 '{node}' 不存在于邻接矩阵中！\n"
                    f"请检查节点名称是否与矩阵中的一致（注意路径分隔符、大小写）", None)
    
    new_matrix = matrix.copy()
    new_matrix.loc[new_node] = "0"
    new_matrix[new_node] = "0"
    
    for node, edge in upstream.items():
        new_matrix.loc[node, new_node] = edge
    for node, edge in downstream.items():
        new_matrix.loc[new_node, node] = edge
    
    try:
        main_path, backup_path = save_with_backup(new_matrix, file_name)
        result_msg = (f"成功添加新节点！\n新节点名称: {new_node}\n"
                      f"上游节点: {', '.join(upstream.keys()) if upstream else '无'}\n"
                      f"已更新文件: {os.path.abspath(main_path)}")
        return result_msg, main_path
    except Exception as e:
        return f"保存文件时出错: {str(e)}", None

def get_edges_handler(file, node1, node2):
    """修复版：兼容文件对象和字符串路径，解决'name'属性错误"""
    try:
        print(f"尝试获取节点 {node1} 到 {node2} 的边")
        
        if file is None:
            print("未上传邻接矩阵文件")
            return [], [], "请先上传邻接矩阵CSV文件"
        
        if isinstance(file, str):
            unified_file_path = file
        else:
            file_name = os.path.basename(file.name)
            unified_file_path = os.path.join(BASE_RECORD_PATH, file_name)
            
            if not os.path.exists(unified_file_path):
                with open(unified_file_path, 'wb') as f_out:
                    f_out.write(file.read())
        
        node1 = node1.strip()
        node2 = node2.strip()
        
        if not node1 or not node2:
            print("节点名称为空")
            return [], [], "请输入两个节点的名称"
        
        matrix, message = load_adjacency_matrix(unified_file_path)
        if matrix is None:
            print(f"加载矩阵失败: {message}")
            return [], [], f"加载文件失败: {message}"
        
        if node1 not in matrix.index:
            print(f"节点 {node1} 不存在")
            return [], [], f"节点 '{node1}' 不存在于邻接矩阵中"
        if node2 not in matrix.index:
            print(f"节点 {node2} 不存在")
            return [], [], f"节点 '{node2}' 不存在于邻接矩阵中"
        
        edges, edge_details, message = get_edges_between_nodes(matrix, node1, node2)
        
        print(f"找到 {len(edges)} 条边: {edges}")
        
        if edges:
            display_msg = f"成功找到 {len(edges)} 条边：{edge_details}"
            return edges, edge_details, display_msg
        else:
            return [], [], message if message else "两个节点之间没有边"
            
    except Exception as e:
        error_msg = f"获取边时发生错误: {str(e)}"
        print(error_msg)
        return [], [], error_msg
    
def remove_specific_edge(edges_state, edge_details_state, edge_to_remove):
    if not edge_to_remove.strip():
        return edges_state, edge_details_state, "请输入要删除的边内容"
    
    try:
        user_input = ast.literal_eval(edge_to_remove.strip())
        input_str = str(user_input)
        
        new_edge_details = []
        new_edges = []
    
        all_edges = []
        for detail in edge_details_state:
            try:
                data = ast.literal_eval(detail)
                if isinstance(data, list):
                    all_edges.extend(data)
                else:
                    all_edges.append(data)
            except:
                all_edges.append(detail)
    
        filtered_edges = [edge for edge in all_edges if str(edge) != input_str]
     
        for i, edge in enumerate(filtered_edges):
            new_edge_details.append(str(edge))
            if edges_state:
                base_edge = edges_state[0].split(":")[0]
                new_edges.append(f"{base_edge}: 边 {i+1}")
        
        if len(new_edge_details) == len(edge_details_state):
            return edges_state, edge_details_state, "未找到匹配的边，请检查输入格式"
        
        display_msg = f"成功删除指定边，剩余 {len(new_edge_details)} 条边：{new_edge_details}" if new_edge_details else \
                      "已删除所有边，当前没有边"
        
        return new_edges, new_edge_details, display_msg
        
    except Exception as e:
        error_msg = f"删除边时发生错误: {str(e)}，请检查输入格式是否正确"
        print(error_msg)
        return edges_state, edge_details_state, error_msg


def remove_all_edges(edges_state, edge_details_state):
    if not edges_state or not edge_details_state:
        return [], [], "没有可删除的边"
    
    return [], [], "已成功删除所有边"


def delete_edges(file, node1, node2, edges_state, edge_details_state):
    if file is None:
        return "请先上传邻接矩阵CSV文件", None
    
    file_name = os.path.basename(file.name)
    unified_file_path = os.path.join(BASE_RECORD_PATH, file_name)
    
    if not os.path.exists(unified_file_path):
        with open(unified_file_path, 'wb') as f_out:
            f_out.write(file.read())
    
    node1 = node1.strip()
    node2 = node2.strip()
    if not node1 or not node2:
        return "请输入两个节点的名称", None
    
    matrix, message = load_adjacency_matrix(unified_file_path)
    if matrix is None:
        return message, None
    
    if node1 not in matrix.index:
        return f"节点 '{node1}' 不存在于邻接矩阵中", None
    if node2 not in matrix.index:
        return f"节点 '{node2}' 不存在于邻接矩阵中", None
    
    new_matrix = matrix.copy()
    original_count = len(edge_details_state)
    
    if edge_details_state:
        try:
            remaining_edges = [ast.literal_eval(detail) for detail in edge_details_state]
            new_matrix.loc[node1, node2] = str(remaining_edges)
        except:
            new_matrix.loc[node1, node2] = str(edge_details_state)
    else:
        new_matrix.loc[node1, node2] = "0"
    
    updated_count = len(edge_details_state)
    deleted_count = original_count - updated_count
    
    try:
        main_path, backup_path = save_with_backup(new_matrix, file_name)
        
        result_msg = (f"成功更新 {deleted_count} 条边！\n"
                      f"涉及节点: {node1} -> {node2}\n"
                      f"更新后剩余 {updated_count} 条边\n"
                      f"已更新文件: {os.path.abspath(main_path)}\n"
                      f"备份文件已保存到: {os.path.abspath(backup_path)}")
        return result_msg, main_path
    except Exception as e:
        return f"保存文件时出错: {str(e)}", None

def add_edge(file, from_node, to_node, action_type, x, y, button, text, other):
    """添加边功能，同时自动创建带来源标识的反向边"""
    if file is None:
        return "请先上传邻接矩阵CSV文件", None
    
    file_name = os.path.basename(file.name)
    unified_file_path = os.path.join(BASE_RECORD_PATH, file_name)
    
    if not os.path.exists(unified_file_path):
        with open(unified_file_path, 'wb') as f_out:
            f_out.write(file.read())
    
    from_node = from_node.strip()
    to_node = to_node.strip()
    if not from_node or not to_node:
        return "请输入源节点和目标节点的名称", None
    
    action_data = {"action_type": action_type}
    if action_type == "click":
        if x is None or y is None:
            return "点击动作需要X和Y坐标", None
        action_data["x"] = int(x)
        action_data["y"] = int(y)
    elif action_type == "system_button":
        if not button.strip():
            return "系统按钮动作需要按钮名称", None
        action_data["button"] = button.strip()
    elif action_type == "input_text":
        if not text.strip():
            return "输入文本动作需要输入内容", None
        action_data["text"] = text.strip()
    else: 
        if not other.strip():
            return "其他动作需要详细信息", None
        action_data["details"] = other.strip()
    
    back_action_data = {
        "action_type": "system_button",
        "button": "back",
        "from": from_node  
    }
    
    matrix, message = load_adjacency_matrix(unified_file_path)
    if matrix is None:
        return message, None
    
    if from_node not in matrix.index:
        return f"源节点 '{from_node}' 不存在于邻接矩阵中", None
    if to_node not in matrix.index:
        return f"目标节点 '{to_node}' 不存在于邻接矩阵中", None
    
    new_matrix = matrix.copy()
    
    current_forward_value = new_matrix.loc[from_node, to_node]
    try:
        if pd.isna(current_forward_value) or str(current_forward_value).strip().lower() in ['0', '', 'none', 'nan', 'null']:
            forward_edges = [action_data]
        else:
            forward_edges = ast.literal_eval(str(current_forward_value))
            if not isinstance(forward_edges, list):
                forward_edges = [forward_edges]
            forward_edges.append(action_data)
    except:
        forward_edges = [action_data]
    new_matrix.loc[from_node, to_node] = str(forward_edges)
    
    # update back edge automatically（to_node → from_node）
    # current_back_value = new_matrix.loc[to_node, from_node]
    # try:
    #     if pd.isna(current_back_value) or str(current_back_value).strip().lower() in ['0', '', 'none', 'nan', 'null']:
    #         back_edges = [back_action_data]
    #     else:
    #         back_edges = ast.literal_eval(str(current_back_value))
    #         if not isinstance(back_edges, list):
    #             back_edges = [back_edges]
    #         if not any(edge.get("button") == "back" and edge.get("from") == from_node for edge in back_edges):
    #             back_edges.append(back_action_data)
    # except:
    #     back_edges = [back_action_data]
    # new_matrix.loc[to_node, from_node] = str(back_edges)
    
    try:
        main_path, backup_path = save_with_backup(new_matrix, file_name)
        
        result_msg = (f"成功添加边及反向边！不加反向边了\n"
                      f"正向边: {from_node} → {to_node}: {action_data}\n"
                      f"反向边: {to_node} → {from_node}: {back_action_data}\n"
                      f"已更新文件: {os.path.abspath(main_path)}")
        return result_msg, main_path
    except Exception as e:
        return f"保存文件时出错: {str(e)}", None


with gr.Blocks(title="邻接矩阵节点关系工具", theme=gr.themes.Soft()) as demo:
    upstream_full_data = gr.State([])
    downstream_full_data = gr.State([])
    upstream_selected = gr.State([])
    downstream_selected = gr.State([])
    manual_selected = gr.State([])
    merged_file = gr.State(None)
    edges_state = gr.State([]) 
    edge_details_state = gr.State([])  
    
    gr.Markdown("## 邻接矩阵节点关系工具")
    gr.Markdown("支持节点关系查询、新增、合并、删除与边添加/删除功能")
    gr.Markdown(f"**文件存储路径：{os.path.abspath(BASE_RECORD_PATH)}**")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="上传邻接矩阵CSV文件", file_types=[".csv"])
            node_input = gr.Textbox(label="输入节点名称", placeholder="请输入要查询的节点名称")
            query_btn = gr.Button("查询关系", variant="primary")
            nodes_output = gr.Textbox(label="节点列表", lines=4)
            
            # 新增节点功能区域
            gr.Markdown("### 新增节点")
            new_node_name = gr.Textbox(label="新节点名称", placeholder="输入要新增的节点名称")
            upstream_relations = gr.Textbox(
                label="上游节点及边", 
                placeholder="格式: 节点1:边信息1,节点2:边信息2（其他节点指向新节点）"
            )
            downstream_relations = gr.Textbox(
                label="下游节点及边", 
                placeholder="格式: 节点1:边信息1,节点2:边信息2（新节点指向其他节点）"
            )
            add_node_btn = gr.Button("新增节点", variant="primary")
            add_node_status = gr.Textbox(label="新增节点结果", lines=3)
            
            # 手动输入图片区域
            gr.Markdown("### 手动输入图片")
            manual_image_input = gr.Textbox(label="图片名称或路径", placeholder="输入要选中的图片名称或路径")
            add_manual_btn = gr.Button("添加/移除图片", variant="secondary")
            manual_status = gr.Textbox(label="操作结果", lines=1)
            
            # 选中图片名称展示区域
            gr.Markdown("### 选中的图片名称")
            selected_names_output = gr.Textbox(label="结果", lines=5)
            get_selected_btn = gr.Button("刷新选中图片名称", variant="secondary")
            
            # 节点合并功能区域
            gr.Markdown("### 节点合并")
            merge_btn = gr.Button("合并节点", variant="stop")
            merge_status = gr.Textbox(label="合并结果", lines=3)
            
            # 节点删除功能区域
            gr.Markdown("### 节点删除")
            node_to_delete_input = gr.Textbox(label="要删除的节点名称", placeholder="输入要删除的节点名称")
            delete_btn = gr.Button("删除节点", variant="destructive")
            delete_status = gr.Textbox(label="删除结果", lines=3)
            
            # 边添加功能区域
            gr.Markdown("### 边添加")
            with gr.Row():
                from_node_input = gr.Textbox(label="源节点", placeholder="输入源节点名称")
                to_node_input = gr.Textbox(label="目标节点", placeholder="输入目标节点名称")
            with gr.Row():
                action_type_input = gr.Dropdown(
                    label="动作类型",
                    choices=["click", "system_button", "input_text", "other"],
                    value="click"
                )
            with gr.Row():
                x_input = gr.Number(label="X坐标", visible=True)
                y_input = gr.Number(label="Y坐标", visible=True)
            button_input = gr.Textbox(label="按钮名称", placeholder="如：back", visible=False)
            text_input = gr.Textbox(label="输入文本", placeholder="输入的文本内容", visible=False)
            other_input = gr.Textbox(label="其他信息", placeholder="其他动作的详细信息", visible=False)
            
            add_edge_btn = gr.Button("添加边", variant="primary")
            add_edge_status = gr.Textbox(label="添加边结果", lines=3)
            
            # 边删除功能区域
            gr.Markdown("### 边删除")
            with gr.Row():
                node1_input = gr.Textbox(label="节点1", placeholder="输入第一个节点名称")
                node2_input = gr.Textbox(label="节点2", placeholder="输入第二个节点名称")
            get_edges_btn = gr.Button("获取节点间的边", variant="secondary")
            edge_status = gr.Textbox(label="边查询结果", lines=3)
            
            # 手动输入要删除的边
            gr.Markdown("#### 手动删除特定边")
            edge_to_remove_input = gr.Textbox(
                label="要删除的边", 
                placeholder="例如: {'action_type': 'click', 'x': 158, 'y': 399}"
            )
            remove_specific_edge_btn = gr.Button("删除指定边", variant="secondary")
            
            # 一键删除所有边
            remove_all_edges_btn = gr.Button("一键删除所有边", variant="destructive")
            
            delete_edge_btn = gr.Button("保存边修改", variant="destructive")
            delete_edge_status = gr.Textbox(label="边修改结果", lines=3)
            
            # 选中图片栏
            gr.Markdown("### 选中的图片栏")
            selected_images_output = gr.Gallery(label="已选中的图片", show_label=True)
            
        with gr.Column(scale=2):
            status_output = gr.Textbox(label="状态信息", lines=2)
            
            gr.Markdown("### 当前节点图片")
            current_node_img = gr.Image(label="当前节点", type="pil")
            
            with gr.Tab("上游节点（指向当前节点）"):
                gr.Markdown("### 上游节点列表")
                upstream_output = gr.Textbox(label="上游节点", lines=3)
                gr.Markdown("### 上游节点图片")
                upstream_gallery = gr.Gallery(
                    label="上游节点图片", 
                    show_label=True, 
                    elem_id="upstream-gallery",
                    columns=[3], 
                    rows=[2], 
                    object_fit="contain", 
                    height="auto"
                )
                gr.Markdown("### 上游节点信息")
                upstream_info_text = gr.Textbox(
                    label="上游节点详细信息", 
                    lines=5, 
                    interactive=True
                )
            
            with gr.Tab("下游节点（当前节点指向）"):
                gr.Markdown("### 下游节点列表")
                downstream_output = gr.Textbox(label="下游节点", lines=3)
                gr.Markdown("### 下游节点图片")
                downstream_gallery = gr.Gallery(
                    label="下游节点图片", 
                    show_label=True, 
                    elem_id="downstream-gallery",
                    columns=[3], 
                    rows=[2], 
                    object_fit="contain", 
                    height="auto"
                )
                gr.Markdown("### 下游节点信息")
                downstream_info_text = gr.Textbox(
                    label="下游节点详细信息", 
                    lines=5, 
                    interactive=True
                )
    
    # 设置事件
    file_input.change(
        fn=get_available_nodes,
        inputs=[file_input],
        outputs=[nodes_output]
    )
    
    query_btn.click(
        fn=process_query,
        inputs=[file_input, node_input],
        outputs=[
            status_output, 
            downstream_output, 
            upstream_output,
            current_node_img,
            upstream_gallery,
            downstream_gallery,
            upstream_full_data,
            downstream_full_data,
            upstream_info_text,
            downstream_info_text
        ]
    ).then(
        fn=lambda: ([], [], []),
        outputs=[upstream_selected, downstream_selected, manual_selected]
    ).then(
        fn=lambda: [],
        outputs=[selected_images_output]
    )
    
    # 新增节点事件
    add_node_btn.click(
        fn=add_new_node,
        inputs=[file_input, new_node_name, upstream_relations, downstream_relations],
        outputs=[add_node_status, merged_file]
    ).then(
        fn=lambda: ([], [], []),
        outputs=[upstream_selected, downstream_selected, manual_selected]
    ).then(
        fn=lambda: [],
        outputs=[selected_images_output]
    ).then(
        fn=lambda merged_file, original_file: 
            load_adjacency_matrix(merged_file)[1] if merged_file else 
            get_available_nodes(original_file),
        inputs=[merged_file, file_input],
        outputs=[nodes_output]
    )
    
    # 上游图片选择事件
    gr.HTML("""<script>
    function setupUpstreamSelection() {
        const gallery = document.getElementById('upstream-gallery');
        if (gallery) {
            gallery.addEventListener('click', function(e) {
                const items = gallery.querySelectorAll('.gallery-item');
                for (let i = 0; i < items.length; i++) {
                    if (items[i].contains(e.target)) {
                        items[i].classList.toggle('selected');
                        const trigger = gradioApp().getElementById('upstream-select-trigger');
                        trigger.setAttribute('data-index', i);
                        trigger.click();
                        return;
                    }
                }
            });
        }
    }
    document.addEventListener('DOMContentLoaded', setupUpstreamSelection);
    </script>""")
    
    # 添加选中样式
    gr.HTML("""<style>
    .gallery-item.selected {
        border: 3px solid #4CAF50 !important;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
    }
    .gallery-item {
        transition: all 0.3s ease;
        border: 3px solid transparent;
        margin: 5px;
        padding: 5px;
    }
    </style>""")
    
    upstream_select_trigger = gr.Button(visible=False, elem_id="upstream-select-trigger")
    upstream_select_trigger.click(
        fn=lambda trigger: int(trigger.getAttribute('data-index')),
        inputs=[upstream_select_trigger],
        outputs=[gr.Number(visible=False, label="上游索引")]
    ).then(
        fn=select_upstream_image,
        inputs=[upstream_full_data, gr.Number(visible=False), upstream_selected],
        outputs=[upstream_selected]
    ).then(
        fn=update_selected_images,
        inputs=[upstream_full_data, downstream_full_data, upstream_selected, downstream_selected, manual_selected],
        outputs=[selected_images_output]
    )
    
    # 下游图片选择事件
    gr.HTML("""<script>
    function setupDownstreamSelection() {
        const gallery = document.getElementById('downstream-gallery');
        if (gallery) {
            gallery.addEventListener('click', function(e) {
                const items = gallery.querySelectorAll('.gallery-item');
                for (let i = 0; i < items.length; i++) {
                    if (items[i].contains(e.target)) {
                        items[i].classList.toggle('selected');
                        const trigger = gradioApp().getElementById('downstream-select-trigger');
                        trigger.setAttribute('data-index', i);
                        trigger.click();
                        return;
                    }
                }
            });
        }
    }
    document.addEventListener('DOMContentLoaded', setupDownstreamSelection);
    </script>""")
    
    downstream_select_trigger = gr.Button(visible=False, elem_id="downstream-select-trigger")
    downstream_select_trigger.click(
        fn=lambda trigger: int(trigger.getAttribute('data-index')),
        inputs=[downstream_select_trigger],
        outputs=[gr.Number(visible=False, label="下游索引")]
    ).then(
        fn=select_downstream_image,
        inputs=[downstream_full_data, gr.Number(visible=False), downstream_selected],
        outputs=[downstream_selected]
    ).then(
        fn=update_selected_images,
        inputs=[upstream_full_data, downstream_full_data, upstream_selected, downstream_selected, manual_selected],
        outputs=[selected_images_output]
    )
    
    # 手动添加图片事件
    add_manual_btn.click(
        fn=add_manual_image,
        inputs=[manual_image_input, upstream_full_data, downstream_full_data,
                upstream_selected, downstream_selected, manual_selected],
        outputs=[manual_selected, manual_status]
    ).then(
        fn=update_selected_images,
        inputs=[upstream_full_data, downstream_full_data, upstream_selected, downstream_selected, manual_selected],
        outputs=[selected_images_output]
    )
    
    # 获取选中图片名称
    get_selected_btn.click(
        fn=get_selected_names,
        inputs=[
            upstream_full_data, 
            downstream_full_data,
            upstream_selected,
            downstream_selected,
            manual_selected
        ],
        outputs=[selected_names_output]
    )
    
    # 合并节点事件
    merge_btn.click(
        fn=merge_nodes,
        inputs=[file_input, upstream_full_data, downstream_full_data,
                upstream_selected, downstream_selected, manual_selected],
        outputs=[merge_status, merged_file, upstream_selected, downstream_selected, manual_selected]
    ).then(
        fn=lambda: [],
        outputs=[selected_images_output]
    ).then(
        fn=lambda merged_file, original_file: 
            load_adjacency_matrix(merged_file)[1] if merged_file else 
            get_available_nodes(original_file),
        inputs=[merged_file, file_input],
        outputs=[nodes_output]
    )
    
    # 删除节点事件
    delete_btn.click(
        fn=delete_node,
        inputs=[file_input, node_to_delete_input],
        outputs=[delete_status, merged_file]
    ).then(
        fn=lambda: [],
        outputs=[selected_images_output]
    ).then(
        fn=lambda: ([], [], []),
        outputs=[upstream_selected, downstream_selected, manual_selected]
    ).then(
        fn=lambda merged_file, original_file: 
            load_adjacency_matrix(merged_file)[1] if merged_file else 
            get_available_nodes(original_file),
        inputs=[merged_file, file_input],
        outputs=[nodes_output]
    )
    
    # 处理动作类型切换显示不同输入框
    def toggle_action_inputs(action_type):
        if action_type == "click":
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        elif action_type == "system_button":
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
        elif action_type == "input_text":
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        else:  # other
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    
    action_type_input.change(
        fn=toggle_action_inputs,
        inputs=[action_type_input],
        outputs=[x_input, y_input, button_input, text_input, other_input]
    )
    
    # 添加边事件
    add_edge_btn.click(
        fn=add_edge,
        inputs=[file_input, from_node_input, to_node_input, action_type_input, 
                x_input, y_input, button_input, text_input, other_input],
        outputs=[add_edge_status, merged_file]
    ).then(
        fn=lambda merged_file, original_file, n1, n2: 
            get_edges_handler(merged_file, n1, n2) if merged_file else 
            get_edges_handler(original_file, n1, n2),
        inputs=[merged_file, file_input, from_node_input, to_node_input],
        outputs=[edges_state, edge_details_state, edge_status]
    ).then(
        fn=lambda: [],
        outputs=[selected_images_output]
    ).then(
        fn=lambda: ([], [], []),
        outputs=[upstream_selected, downstream_selected, manual_selected]
    ).then(
        fn=lambda merged_file, original_file: 
            load_adjacency_matrix(merged_file)[1] if merged_file else 
            get_available_nodes(original_file),
        inputs=[merged_file, file_input],
        outputs=[nodes_output]
    )
    
    # 获取两个节点之间的边
    get_edges_btn.click(
        fn=get_edges_handler,
        inputs=[file_input, node1_input, node2_input],
        outputs=[edges_state, edge_details_state, edge_status]
    )
    
    # 删除特定边事件
    remove_specific_edge_btn.click(
        fn=remove_specific_edge,
        inputs=[edges_state, edge_details_state, edge_to_remove_input],
        outputs=[edges_state, edge_details_state, edge_status]
    )
    
    # 一键删除所有边事件
    remove_all_edges_btn.click(
        fn=remove_all_edges,
        inputs=[edges_state, edge_details_state],
        outputs=[edges_state, edge_details_state, edge_status]
    )
    
    # 保存边修改
    delete_edge_btn.click(
        fn=delete_edges,
        inputs=[file_input, node1_input, node2_input, edges_state, edge_details_state],
        outputs=[delete_edge_status, merged_file]
    ).then(
        fn=lambda merged_file, original_file, n1, n2: 
            get_edges_handler(merged_file, n1, n2) if merged_file else 
            get_edges_handler(original_file, n1, n2),
        inputs=[merged_file, file_input, node1_input, node2_input],
        outputs=[edges_state, edge_details_state, edge_status]
    ).then(
        fn=lambda: [],
        outputs=[selected_images_output]
    ).then(
        fn=lambda: ([], [], []),
        outputs=[upstream_selected, downstream_selected, manual_selected]
    ).then(
        fn=lambda merged_file, original_file: 
            load_adjacency_matrix(merged_file)[1] if merged_file else 
            get_available_nodes(original_file),
        inputs=[merged_file, file_input],
        outputs=[nodes_output]
    )
    
    # 按Enter键查询
    node_input.submit(
        fn=process_query,
        inputs=[file_input, node_input],
        outputs=[
            status_output, 
            downstream_output, 
            upstream_output,
            current_node_img,
            upstream_gallery,
            downstream_gallery,
            upstream_full_data,
            downstream_full_data,
            upstream_info_text,
            downstream_info_text
        ]
    ).then(
        fn=lambda: ([], [], []),
        outputs=[upstream_selected, downstream_selected, manual_selected]
    ).then(
        fn=lambda: [],
        outputs=[selected_images_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True, server_port=7681)
