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
import time
import json
import logging
import colorlog
import argparse
import datetime
import time
from pathlib import Path
from dotenv import load_dotenv
import yaml
from src.agent import VanillaAgent
from src.agent.agent_atlas import AtlasAgent
from src.agent.agent_tars import TarsAgent
from src.agent.agent_tars_dpo import TarsDPOAgent
from src.agent.agent_qwen3 import Qwen3Agent
from src.agent.agent_api import APIAgent
from src.test.graph_tools import Graph_DataSet

load_dotenv()

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

def load_yaml(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading YAML file {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on smartphone assistant tasks"
    )
    parser.add_argument(
        "--config",
        default='./config/default.yaml',
        help="Path to the config YAML file.",
    )
    parser.add_argument(
        "--model",
        default='qwen',
        help="Model of agent you want to use.",
    )  
    parser.add_argument(
        "--api",
        default='gpt',  
        help="API model to use.",
    )

    args = parser.parse_args()
    tmp_time = datetime.datetime.now().strftime("%m%d_%H%M") 
    config_name = f'vanilla_{args.model}_{tmp_time}'

    config = load_yaml(args.config)
    parent_dir = config['path']['image_folder']
    graph_json_file = config['graph']['graph_file']
    output_dir = config['path']['output_folder']
    os.makedirs(output_dir, exist_ok=True)

    log_file_path = f'./log/{config_name}.log'
    setup_logging(log_file_path)
    logger = logging.getLogger(__name__)
    logger.info("Progress Start!")

    graph_dataset = Graph_DataSet(config['graph'])
    if 'qwen3' in args.model:
        agent = Qwen3Agent(config['agent'])
    elif 'qwen' in args.model:
        agent = VanillaAgent(config['agent'])
    elif 'owl' in args.model:
        agent = VanillaAgent(config['agent'])
    elif 'atlas' in args.model:
        agent = AtlasAgent(config['agent'])
    elif 'tars_dpo' in args.model:
        agent = TarsDPOAgent(config['agent'])
    elif 'tars' in args.model:
        agent = TarsAgent(config['agent'])
    elif 'api' in args.model:
        agent = APIAgent(model=args.api)
    else:
        # use your own agent
        raise ValueError(f"Unsupported agent model: {args.model}")


    task_json = config['tasks']['tasks_file']
    with open(task_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for task_item in data:
        task = task_item['query']
        graph_dataset.set_task(task)
        agent.set_task(task)
        complete = False

        image_path = graph_dataset.home_page
        max_step = config['tasks']['max_steps']
        current_step = 0
        start_time = time.time()
        logger.info(f"----------开始执行任务{task}----------")
        while not complete and current_step < max_step:
            image_path = os.path.join(parent_dir, image_path)
            action, action_description = agent.agent_step(image_path) 
            image_path, answer = graph_dataset.step(action, action_description=action_description)
            if answer:
                logger.info(f"任务结束，回答为: {answer}")
                complete = True
            elif image_path is None:
                logger.warning("出现错误，无法继续执行任务")
                complete = True
            current_step += 1

        # save trajectory
        use_time = time.time() - start_time
        logger.info(f"任务 '{task}' 执行结束, 总步数: {current_step}, 用时: {use_time:.2f} 秒")
        graph_dataset.save_trajectory(output_dir, use_time, save_image=False, config_name=config_name, parent_dir = parent_dir)
        logger.info(f"任务轨迹已保存")


if __name__ == "__main__":
    main()