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

"""
This file is to construct a graph for demonstration-based learning when traversing the trajectory 
"""
from FlagEmbedding import BGEM3FlagModel, FlagModel
import torch
import os
import json
import logging
import traceback
import datetime
import colorlog
import argparse
from pathlib import Path
from dotenv import load_dotenv
from src.graph_construction.graph import Graph
load_dotenv()

model = FlagModel("./models--BAAI--bge-large-zh-v1.5/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：", use_fp16=True)

if torch.cuda.is_available():
    torch.device('cuda:0')
else:
    torch.device('cpu')

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

def log_error_simple(error_message, log_file):
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] ERROR: {error_message}\n")
        f.write(f"详细错误信息: {traceback.format_exc()}\n")
        f.write("-" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on smartphone assistant tasks"
    )

    parser.add_argument(
        "--input_folder",
        required=None,
        help="Path to the input JSONL file containing test data",
    )   
    parser.add_argument(
        "--output_file",
        default=None,
        help="Path to save results.",
    )  

    log_file_path = './results/construct_graph.log'
    setup_logging(log_file_path)
    logger = logging.getLogger(__name__)
    logger.info("Progress Start!")

    args = parser.parse_args()

    # If output file is not specified, create one based on input filename
    if args.input_folder is None:
        input_folder = Path('./examples/trajectories')
    if args.output_file is None:
        output_path = Path('./results/graph.json')
    else:
        output_path = Path(args.output_file)
    
    # Ensure output directory exists
    output_dir = output_path.parent
    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Input folder path: {input_folder}")
    logger.info(f"Output file: {output_path}")

    graph = Graph(app='美团')
    task_dirs = os.listdir(input_folder) 

    for task_dir in task_dirs:
        if task_dir.endswith('.json'):
            continue

        task_json = os.path.join(input_folder, task_dir, f'{task_dir}.json')  
        logger.info(f"Processing task directory: {task_dir}")
        try:
            with open(task_json, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
                logger.info(f"成功加载 {task_dir} 的轨迹数据")
        except Exception as e:
            logger.info(f"错误: 加载 {task_dir} 的json文件时出错: {str(e)}")
            continue
        
        query = task_data['task']   
        logger.info(f"Processing task: {query} with id {task_dir}")
        new_trajectory = True
        last_edge = -1
        last_node = -1
        for i, step_data in enumerate(task_data['trajectory']):
            # reconstruct data
            logger.info(f"Start Updating graph with step {i+1} for task {query}")
            step_data['screenshot'] = os.path.join(input_folder, task_dir, f'observation_{i}.png')  
            step_data['query'] = query
            last_edge, last_node = graph.update(data=step_data, new_trajectory= new_trajectory, last_node=last_node, last_edge=last_edge, threshold=0.85, step=i) 
            new_trajectory = False
            logger.info(f"Updated graph with step {i+1} for task {query}")

        graph.save_graph(save_path=output_path)

    logger.info(f"Graph saved to {output_path}")

if __name__ == "__main__":
    main()