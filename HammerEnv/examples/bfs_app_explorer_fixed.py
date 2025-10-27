#!/usr/bin/env python3
"""
修复版BFS应用探索器 - 解决API认证问题
"""

# 在导入任何模块之前设置环境变量
import os
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = "http://innoday-demo.oppo.test/v1"

from absl import logging
from datetime import datetime
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional, Tuple
from queue import Queue
import threading
import hashlib
import re

import argparse
import json
import os
import uuid
import time

# 现在导入其他模块
from hammer_agent.qwen_agent import Operator as QwenOperator
from server.client import HammerEnvClient
import sys

from server.utils import (
    DeviceManager,
    get_action_param_prompt_grid,
    get_action_param_prompt_som,
    get_action_types,
    get_ip,
    image_to_base64,
    screenshot_to_grid_base64,
    screenshot_to_som_base64,
)

# 初始化设备管理器
device_manager = DeviceManager()
available_devices = device_manager.get_available_devices()
device_count = len(available_devices)
print(f"可利用的设备数量: {device_count}")
print(f"可利用的设备列表: {available_devices}")

logging.set_verbosity("debug")
load_dotenv(os.path.join(os.getcwd(), ".env"))

APP_NAME = "xiaohongshu"  # 小红书
MODEL_NAME = "Qwen2.5-VL-72B-Instruct"  # 使用正确的模型名称


@dataclass
class ClickableElement:
    """可点击元素的数据结构"""
    element_name: str
    element_type: str
    coordinates: Tuple[int, int]
    description: str
    confidence: float = 0.0


@dataclass
class TrajectoryNode:
    """轨迹节点数据结构"""
    path: List[str]  # 轨迹路径
    current_page: str  # 当前页面描述
    clickable_elements: List[ClickableElement]  # 当前页面可点击元素
    depth: int  # 当前深度
    status: str  # 状态: pending/exploring/completed/failed
    page_signature: str  # 页面签名，用于去重
    timestamp: str = ""


@dataclass
class ExplorationConfig:
    """探索配置"""
    max_depth: int = 3  # 最大探索深度
    max_trajectories: int = 50  # 最大轨迹数量
    app_name: str = "小红书"  # 目标应用
    output_dir: str = "trajectories"  # 输出目录
    delay_between_actions: float = 2.0  # 操作间延迟
    max_retries: int = 3  # 最大重试次数
    model_name: str = "Qwen2.5-VL-72B-Instruct"  # AI模型名称
    reset_environment_per_task: bool = True  # 是否每个任务后重置环境
    reset_delay: float = 1.0  # 重置延迟时间


class BFSEplorer:
    """广度优先搜索应用探索器"""
    
    def __init__(self, config: ExplorationConfig, server_url: str):
        self.config = config
        self.server_url = server_url
        self.exploration_queue = Queue()
        self.visited_pages: Set[str] = set()
        self.completed_trajectories: List[TrajectoryNode] = []
        self.client: Optional[HammerEnvClient] = None
        self.operator: Optional[QwenOperator] = None
        self.device_info = None
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 页面分析提示词
        self.page_analysis_prompt = """
            You are a page analysis expert. Please analyze the current page screenshot and identify all clickable UI elements.

            Please carefully analyze the page and find all possible clickable elements, including:
            - Buttons
            - Links
            - Icons
            - Menu items
            - Input fields
            - Other interactive elements
            
            ***Please exclude the following time-varying or dynamically pushed content from the identification of clickable elements:***
            1. Content pushed on the homepage (e.g., news feeds, recommendation lists, dynamically updated modules)
            2. Various advertisements (e.g., pop-up ads, banner ads, in-feed ads)
            
            Please return the results strictly in the following JSON format, without adding any other text:
            {
                "clickable_elements": [
                    {
                        "element_name": "Settings",
                        "element_type": "button",
                        "coordinates": [x, y],
                        "description": "Settings button",
                        "confidence": 0.9
                    }
                ]
            }
            
            Important requirements:
            1. Return only JSON format, do not add any explanatory text
            2. Ensure the JSON format is completely correct, without any syntax errors
            3. No trailing commas
            4. Coordinates must be a numeric array [x, y]
            5. Confidence must be a number between 0 and 1
            6. If no clickable elements are found, return an empty array: {"clickable_elements": []}
            """

    def initialize_device(self):
        """初始化设备连接"""
        try:
            self.client = HammerEnvClient(src=self.server_url)
            self.device_info = self.client.request_device()
            logging.info(f"设备连接成功: {self.device_info}")
            
            # 检查device_info结构
            if not self.device_info:
                logging.error("device_info为空")
                return False
                
            if "screen_size" not in self.device_info:
                logging.error(f"device_info缺少screen_size键: {self.device_info}")
                return False
                
            logging.info(f"屏幕尺寸: {self.device_info['screen_size']}")
            
            # 创建Operator
            self.operator = QwenOperator(
                device_client=self.client, 
                max_steps=10
            )
            return True
        except Exception as e:
            logging.error(f"设备初始化失败: {e}")
            import traceback
            logging.error(f"详细错误信息: {traceback.format_exc()}")
            return False

    def test_api_connection(self):
        """测试API连接"""
        try:
            logging.info("测试API连接...")
            
            # 执行一个简单的测试任务
            result = self.operator.run(task="测试API连接")
            
            if result and "trajectory" in result:
                logging.info("✅ API连接测试成功")
                return True
            else:
                logging.warning("❌ API连接测试失败")
                return False
                
        except Exception as e:
            logging.error(f"API连接测试失败: {e}")
            return False

    def analyze_clickable_elements(self, screenshot_base64: str, ui_elements: List) -> List[ClickableElement]:
        """使用AI分析页面可点击元素"""
        try:
            # 使用operator.run来分析页面
            analysis_task = f"请分析当前页面，识别所有可以点击的UI元素。{self.page_analysis_prompt}"
            
            # 执行分析任务
            result = self.operator.run(task=analysis_task)
            #logging.info(f"AI分析结果: {result}")
            
            # 从结果中提取响应
            if result and "trajectory" in result:
                # 获取最后一次响应
                last_response = None
                for step in result["trajectory"]:
                    if "response" in step:
                        last_response = step["response"]
                
                if last_response:
                    logging.info(f"AI响应内容: {last_response}")
                    
                    # 提取JSON部分
                    json_match = re.search(r'\{.*\}', last_response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        logging.info(f"提取的JSON字符串: {json_str}")
                        
                        try:
                            analysis_result = json.loads(json_str)
                            logging.info(f"解析后的JSON: {analysis_result}")
                            
                            clickable_elements = []
                            for element_data in analysis_result.get("clickable_elements", []):
                                element = ClickableElement(
                                    element_name=element_data.get("element_name", ""),
                                    element_type=element_data.get("element_type", "unknown"),
                                    coordinates=tuple(element_data.get("coordinates", [0, 0])),
                                    description=element_data.get("description", ""),
                                    confidence=element_data.get("confidence", 0.0)
                                )
                                clickable_elements.append(element)
                            
                            logging.info(f"成功解析 {len(clickable_elements)} 个可点击元素")
                            return clickable_elements
                            
                        except json.JSONDecodeError as e:
                            logging.error(f"JSON解析失败: {e}")
                            logging.error(f"问题JSON字符串: {json_str}")
                            
                            # 尝试多种修复方法
                            fixed_json = json_str
                            
                            # 修复方法1: 移除尾随逗号
                            fixed_json = re.sub(r',\s*}', '}', fixed_json)
                            fixed_json = re.sub(r',\s*]', ']', fixed_json)
                            
                            # 修复方法2: 修复可能的引号问题
                            fixed_json = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_json)
                            
                            # 修复方法3: 修复可能的单引号
                            fixed_json = fixed_json.replace("'", '"')
                            
                            # 修复方法4: 移除可能的注释
                            fixed_json = re.sub(r'//.*$', '', fixed_json, flags=re.MULTILINE)
                            fixed_json = re.sub(r'/\*.*?\*/', '', fixed_json, flags=re.DOTALL)
                            
                            # 修复方法5: 修复可能的换行问题
                            fixed_json = re.sub(r'\n\s*', ' ', fixed_json)
                            
                            # 修复方法6: 修复不完整的JSON结构
                            # 检查是否缺少闭合符
                            if fixed_json.count('[') > fixed_json.count(']'):
                                fixed_json += ']'
                                logging.info("添加了缺失的数组闭合符 ]")
                            if fixed_json.count('{') > fixed_json.count('}'):
                                fixed_json += '}'
                                logging.info("添加了缺失的对象闭合符 }")
                            
                            # 修复方法7: 处理可能的截断问题
                            # 如果JSON以逗号结尾，移除它
                            fixed_json = re.sub(r',\s*$', '', fixed_json)
                            
                            # 如果JSON以不完整的对象结尾，尝试修复
                            if fixed_json.endswith(','):
                                fixed_json = fixed_json.rstrip(',')
                                if fixed_json.count('[') > fixed_json.count(']'):
                                    fixed_json += ']'
                                if fixed_json.count('{') > fixed_json.count('}'):
                                    fixed_json += '}'
                            
                            logging.info(f"修复后的JSON字符串: {fixed_json}")
                            
                            try:
                                analysis_result = json.loads(fixed_json)
                                logging.info("通过修复JSON格式成功解析")
                                
                                clickable_elements = []
                                for element_data in analysis_result.get("clickable_elements", []):
                                    element = ClickableElement(
                                        element_name=element_data.get("element_name", ""),
                                        element_type=element_data.get("element_type", "unknown"),
                                        coordinates=tuple(element_data.get("coordinates", [0, 0])),
                                        description=element_data.get("description", ""),
                                        confidence=element_data.get("confidence", 0.0)
                                    )
                                    clickable_elements.append(element)
                                
                                logging.info(f"成功解析 {len(clickable_elements)} 个可点击元素")
                                return clickable_elements
                                
                            except json.JSONDecodeError as e2:
                                logging.error(f"修复后仍然无法解析JSON: {e2}")
                                
                                # 尝试更激进的修复方法
                                try:
                                    # 方法7: 尝试提取数组部分（包括不完整的）
                                    array_match = re.search(r'"clickable_elements"\s*:\s*\[(.*?)(?:\]|$)', fixed_json, re.DOTALL)
                                    if array_match:
                                        array_content = array_match.group(1)
                                        logging.info(f"提取的数组内容: {array_content}")
                                        
                                        # 如果数组为空，直接返回空列表
                                        if not array_content.strip():
                                            return []
                                        
                                        # 尝试解析数组中的每个元素
                                        elements = []
                                        
                                        # 更智能的元素分割：查找完整的对象
                                        # 使用栈来匹配大括号
                                        brace_count = 0
                                        current_element = ""
                                        i = 0
                                        
                                        while i < len(array_content):
                                            char = array_content[i]
                                            current_element += char
                                            
                                            if char == '{':
                                                brace_count += 1
                                            elif char == '}':
                                                brace_count -= 1
                                                if brace_count == 0:
                                                    # 找到一个完整的元素
                                                    try:
                                                        elem_data = json.loads(current_element.strip())
                                                        element = ClickableElement(
                                                            element_name=elem_data.get("element_name", ""),
                                                            element_type=elem_data.get("element_type", "unknown"),
                                                            coordinates=tuple(elem_data.get("coordinates", [0, 0])),
                                                            description=elem_data.get("description", ""),
                                                            confidence=elem_data.get("confidence", 0.0)
                                                        )
                                                        elements.append(element)
                                                        logging.info(f"成功解析元素: {element.element_name}")
                                                    except Exception as elem_e:
                                                        logging.warning(f"解析元素失败: {elem_e}, 元素内容: {current_element}")
                                                    
                                                    current_element = ""
                                            i += 1
                                        
                                        if elements:
                                            logging.info(f"通过智能元素级解析成功解析 {len(elements)} 个元素")
                                            return elements
                                        else:
                                            # 如果智能解析失败，尝试简单的正则表达式方法
                                            element_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', array_content)
                                            for elem_str in element_matches:
                                                try:
                                                    elem_data = json.loads(elem_str)
                                                    element = ClickableElement(
                                                        element_name=elem_data.get("element_name", ""),
                                                        element_type=elem_data.get("element_type", "unknown"),
                                                        coordinates=tuple(elem_data.get("coordinates", [0, 0])),
                                                        description=elem_data.get("description", ""),
                                                        confidence=elem_data.get("confidence", 0.0)
                                                    )
                                                    elements.append(element)
                                                except:
                                                    continue
                                            
                                            if elements:
                                                logging.info(f"通过简单元素级解析成功解析 {len(elements)} 个元素")
                                                return elements
                                
                                except Exception as e3:
                                    logging.error(f"元素级解析也失败: {e3}")
                                
                                # 最后的尝试：返回空列表而不是崩溃
                                logging.warning("所有JSON解析方法都失败，返回空列表")
                                return []
                    else:
                        logging.warning("未找到JSON格式的响应")
                        return []
            
            logging.warning("无法从AI响应中提取可点击元素")
            
            # 备用方案：尝试从文本中提取一些基本信息
            try:
                logging.info("尝试从文本中提取基本信息...")
                # 查找可能的坐标信息
                coord_matches = re.findall(r'\[(\d+),\s*(\d+)\]', last_response)
                if coord_matches:
                    logging.info(f"从文本中找到 {len(coord_matches)} 个坐标")
                    clickable_elements = []
                    for i, (x, y) in enumerate(coord_matches):
                        element = ClickableElement(
                            element_name=f"元素{i+1}",
                            element_type="unknown",
                            coordinates=(int(x), int(y)),
                            description=f"从文本中提取的坐标 [{x}, {y}]",
                            confidence=0.5
                        )
                        clickable_elements.append(element)
                    logging.info(f"成功提取 {len(clickable_elements)} 个元素")
                    return clickable_elements
            except Exception as e:
                logging.error(f"文本提取也失败: {e}")
            
            return []
                
        except Exception as e:
            logging.error(f"分析可点击元素失败: {e}")
            return []

    def execute_trajectory(self, trajectory_path: List[str]) -> bool:
        """执行轨迹到达目标页面"""
        try:
            # 将轨迹路径转换为任务描述
            task_description = " -> ".join(trajectory_path)
            logging.info(f"执行轨迹: {task_description}")
            
            # 添加调试信息
            logging.info(f"使用模型: {self.config.model_name}")
            logging.info(f"设备信息: {self.device_info}")
            logging.info(f"Operator类型: {type(self.operator)}")
            
            # 使用operator.run执行整个轨迹
            result = self.operator.run(task=task_description)
            
            if result and "trajectory" in result:
                logging.info(f"轨迹执行完成，共执行 {len(result['trajectory'])} 步")
                
                # 添加详细的轨迹分析
                for i, step in enumerate(result['trajectory']):
                    logging.info(f"步骤 {i+1}:")
                    logging.info(f"  响应: {step.get('response', 'N/A')}")
                    logging.info(f"  动作: {step.get('action', 'N/A')}")
                
                # 保存轨迹结果到实例变量，供后续使用
                self.last_trajectory_result = result
                return True
            else:
                logging.warning("轨迹执行失败")
                return False
                    
        except Exception as e:
            logging.error(f"执行轨迹失败: {e}")
            return False

    def explore_page(self, trajectory_node: TrajectoryNode) -> List[TrajectoryNode]:
        """探索单个页面，返回新的轨迹节点"""
        try:
            # 执行轨迹到达目标页面
            if not self.execute_trajectory(trajectory_node.path):
                logging.error(f"无法执行轨迹: {trajectory_node.path}")
                return []
            
            # 分析当前页面的可点击元素
            clickable_elements = self.analyze_clickable_elements("", [])
            
            # 如果AI分析失败，使用默认的点击元素
            if not clickable_elements:
                logging.warning("AI分析失败，使用默认点击元素")
                clickable_elements = [
                    ClickableElement(
                        element_name="默认点击",
                        element_type="button",
                        coordinates=(500, 1000),  # 屏幕中心偏上
                        description="默认点击位置",
                        confidence=0.3
                    ),
                    ClickableElement(
                        element_name="返回",
                        element_type="button", 
                        coordinates=(100, 100),  # 左上角
                        description="可能的返回按钮",
                        confidence=0.3
                    )
                ]
            
            # 生成页面签名（使用轨迹路径作为签名）
            page_signature = " -> ".join(trajectory_node.path)
            
            # 检查是否已访问过此页面
            if page_signature in self.visited_pages:
                logging.info(f"页面已访问过，跳过: {trajectory_node.path}")
                return []
            
            # 标记页面为已访问
            self.visited_pages.add(page_signature)
            
            # 生成新的轨迹节点
            new_nodes = []
            for element in clickable_elements:
                if trajectory_node.depth < self.config.max_depth:
                    new_path = trajectory_node.path + [f"点击{element.element_name}"]
                    new_node = TrajectoryNode(
                        path=new_path,
                        current_page=f"{trajectory_node.current_page} -> {element.element_name}",
                        clickable_elements=[],
                        depth=trajectory_node.depth + 1,
                        status="pending",
                        page_signature="",
                        timestamp=datetime.now().isoformat()
                    )
                    new_nodes.append(new_node)
            
            # 更新当前节点状态
            trajectory_node.status = "completed"
            trajectory_node.clickable_elements = clickable_elements
            trajectory_node.page_signature = page_signature
            
            return new_nodes
            
        except Exception as e:
            logging.error(f"探索页面失败: {e}")
            trajectory_node.status = "failed"
            return []

    def save_trajectory(self, trajectory_node: TrajectoryNode):
        """保存轨迹到文件夹"""
        try:
            # 生成文件名称（基于最后一个动作的名词）
            if trajectory_node.path:
                last_action = trajectory_node.path[-1]
                # 提取动作中的名词
                file_name = self.extract_noun_from_action(last_action)
            else:
                file_name = "unknown_action"
            
            # 文件夹名与JSON文件名一致
            folder_name = file_name
            
            # 创建文件夹路径
            folder_path = os.path.join(self.config.output_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            # 生成步骤数据并保存截图
            steps_data = []
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            
            # 检查是否有轨迹执行结果
            if hasattr(self, 'last_trajectory_result') and self.last_trajectory_result:
                trajectory_result = self.last_trajectory_result
                trajectory_steps = trajectory_result.get('trajectory', [])
                
                # 使用轨迹执行过程中的截图
                for i, action in enumerate(trajectory_node.path):
                    # 解析动作并生成对应的JSON格式
                    action_json = self.parse_action_to_json(action)
                    action_description = self.generate_action_description(action, i)
                    screenshot_name = f"Screenshot_{timestamp}_{i}.jpg"
                    
                    # 从轨迹结果中获取对应的截图
                    if i < len(trajectory_steps):
                        step_screenshot = trajectory_steps[i].get('observation', '')
                        screenshot_saved = self.save_screenshot_from_base64(folder_path, screenshot_name, step_screenshot)
                    else:
                        screenshot_saved = False
                    
                    step_data = {
                        "action": action_json,
                        "action_description_v1": action_description,
                        "screenshot": screenshot_name,
                        "ui_elements": []
                    }
                    steps_data.append(step_data)
                
                # 添加最后的complete动作和对应的截图
                final_screenshot_name = f"Screenshot_{timestamp}_{len(trajectory_node.path)}.jpg"
                if len(trajectory_steps) > 0:
                    # 使用最后一步的截图作为complete动作的截图
                    final_step_screenshot = trajectory_steps[-1].get('observation', '')
                    final_screenshot_saved = self.save_screenshot_from_base64(folder_path, final_screenshot_name, final_step_screenshot)
                else:
                    final_screenshot_saved = False
                
                final_step_data = {
                    "action": json.dumps({"action_type": "complete", "status": "success"}, ensure_ascii=False),
                    "action_description_v1": "任务完成，所有操作已成功执行。",
                    "screenshot": final_screenshot_name,
                    "ui_elements": []
                }
                steps_data.append(final_step_data)
            else:
                # 如果没有轨迹结果，使用默认方式
                logging.warning("没有找到轨迹执行结果，使用默认截图保存方式")
                for i, action in enumerate(trajectory_node.path):
                    action_json = self.parse_action_to_json(action)
                    action_description = self.generate_action_description(action, i)
                    screenshot_name = f"Screenshot_{timestamp}_{i}.jpg"
                    
                    step_data = {
                        "action": action_json,
                        "action_description_v1": action_description,
                        "screenshot": screenshot_name,
                        "ui_elements": []
                    }
                    steps_data.append(step_data)
            
            # 保存JSON文件
            json_filepath = os.path.join(folder_path, f"{file_name}.json")
            with open(json_filepath, "w", encoding="utf-8") as f:
                json.dump(steps_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"轨迹已保存: {json_filepath}")
            logging.info(f"文件夹: {folder_path}")
            logging.info(f"文件名: {file_name}")
            logging.info(f"步骤数: {len(steps_data)} (包含complete动作)")
            
        except Exception as e:
            logging.error(f"保存轨迹失败: {e}")
    
    def save_screenshot(self, folder_path: str, screenshot_name: str, step_index: int) -> bool:
        """保存截图到指定文件夹"""
        try:
            # 尝试从operator获取当前截图
            if hasattr(self.operator, 'device_client') and self.operator.device_client:
                # 获取当前屏幕截图 - 使用step方法获取最新截图
                screenshot_base64 = self.operator.device_client.step({
                    "name": "wait",
                    "arguments": "{}"
                })
                
                if screenshot_base64:
                    # 解码base64图片数据
                    import base64
                    from PIL import Image
                    import io
                    
                    # 移除base64前缀（如果有）
                    if screenshot_base64.startswith('data:image'):
                        screenshot_base64 = screenshot_base64.split(',')[1]
                    
                    # 解码base64数据
                    image_data = base64.b64decode(screenshot_base64)
                    
                    # 创建PIL图像对象
                    image = Image.open(io.BytesIO(image_data))
                    
                    # 如果是RGBA模式，转换为RGB模式（JPEG不支持透明度）
                    if image.mode == 'RGBA':
                        # 创建白色背景
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        # 将RGBA图像合成到白色背景上
                        background.paste(image, mask=image.split()[-1])  # 使用alpha通道作为mask
                        image = background
                    elif image.mode != 'RGB':
                        # 其他模式也转换为RGB
                        image = image.convert('RGB')
                    
                    # 保存图片
                    screenshot_path = os.path.join(folder_path, screenshot_name)
                    image.save(screenshot_path, 'JPEG', quality=85)
                    
                    logging.info(f"截图已保存: {screenshot_path}")
                    return True
                else:
                    logging.warning(f"无法获取第{step_index + 1}步的截图")
                    return False
            else:
                logging.warning("无法访问设备客户端获取截图")
                return False
                
        except Exception as e:
            logging.error(f"保存截图失败: {e}")
            return False
    
    def save_screenshot_from_base64(self, folder_path: str, screenshot_name: str, screenshot_base64: str) -> bool:
        """从base64数据保存截图到指定文件夹"""
        try:
            if not screenshot_base64:
                logging.warning(f"截图数据为空: {screenshot_name}")
                return False
            
            # 解码base64图片数据
            import base64
            from PIL import Image
            import io
            
            # 移除base64前缀（如果有）
            if screenshot_base64.startswith('data:image'):
                screenshot_base64 = screenshot_base64.split(',')[1]
            
            # 解码base64数据
            image_data = base64.b64decode(screenshot_base64)
            
            # 创建PIL图像对象
            image = Image.open(io.BytesIO(image_data))
            
            # 如果是RGBA模式，转换为RGB模式（JPEG不支持透明度）
            if image.mode == 'RGBA':
                # 创建白色背景
                background = Image.new('RGB', image.size, (255, 255, 255))
                # 将RGBA图像合成到白色背景上
                background.paste(image, mask=image.split()[-1])  # 使用alpha通道作为mask
                image = background
            elif image.mode != 'RGB':
                # 其他模式也转换为RGB
                image = image.convert('RGB')
            
            # 保存图片
            screenshot_path = os.path.join(folder_path, screenshot_name)
            image.save(screenshot_path, 'JPEG', quality=85)
            
            logging.info(f"截图已保存: {screenshot_path}")
            return True
                
        except Exception as e:
            logging.error(f"保存截图失败: {e}")
            return False
    
    def capture_step_screenshot(self, step_index: int) -> str:
        """捕获当前步骤的截图并返回base64数据"""
        try:
            if hasattr(self.operator, 'device_client') and self.operator.device_client:
                # 获取当前屏幕截图 - 使用step方法获取最新截图
                screenshot_base64 = self.operator.device_client.step({
                    "name": "wait",
                    "arguments": "{}"
                })
                if screenshot_base64:
                    return screenshot_base64
                else:
                    logging.warning(f"无法获取第{step_index + 1}步的截图")
                    return None
            else:
                logging.warning("无法访问设备客户端获取截图")
                return None
                
        except Exception as e:
            logging.error(f"捕获截图失败: {e}")
            return None
    
    def parse_action_to_json(self, action: str) -> str:
        """将动作字符串解析为JSON格式"""
        try:
            action = action.strip()
            
            # 处理不同的动作类型
            if "打开" in action and "应用" in action:
                # "打开小红书应用" -> {"action_type": "open", "app": "小红书"}
                app_name = action.replace("打开", "").replace("应用", "").strip()
                action_json = {
                    "action_type": "open",
                    "app": app_name
                }
            elif "点击" in action and "坐标" in action:
                # "点击屏幕坐标[925,1700]位置" -> {"action_type": "click", "x": "925", "y": "1700"}
                coord_match = re.search(r'\[(\d+),\s*(\d+)\]', action)
                if coord_match:
                    x, y = coord_match.groups()
                    action_json = {
                        "action_type": "click",
                        "x": x,
                        "y": y
                    }
                else:
                    action_json = {"action_type": "click", "x": "0", "y": "0"}
            elif "点击" in action:
                # "点击搜索按钮" -> {"action_type": "click", "element": "搜索按钮"}
                element_name = action.replace("点击", "").strip()
                action_json = {
                    "action_type": "click",
                    "element": element_name
                }
            elif "输入" in action:
                # "输入文本" -> {"action_type": "type", "text": "文本"}
                text = action.replace("输入", "").strip()
                action_json = {
                    "action_type": "type",
                    "text": text
                }
            elif "滑动" in action or "滚动" in action:
                # "滑动屏幕" -> {"action_type": "swipe", "direction": "up"}
                action_json = {
                    "action_type": "swipe",
                    "direction": "up"
                }
            else:
                # 默认情况
                action_json = {
                    "action_type": "unknown",
                    "description": action
                }
            
            return json.dumps(action_json, ensure_ascii=False)
            
        except Exception as e:
            logging.warning(f"解析动作失败: {e}, 使用默认格式")
            return json.dumps({"action_type": "unknown", "description": action}, ensure_ascii=False)
    
    def generate_action_description(self, action: str, step_index: int) -> str:
        """生成动作描述"""
        try:
            action = action.strip()
            
            if "打开" in action and "应用" in action:
                app_name = action.replace("打开", "").replace("应用", "").strip()
                return f"用户要求打开{app_name}应用，当前屏幕显示的是手机主屏幕，因此需要使用open命令来打开指定的应用。"
            elif "点击" in action and "坐标" in action:
                coord_match = re.search(r'\[(\d+),\s*(\d+)\]', action)
                if coord_match:
                    x, y = coord_match.groups()
                    return f"用户要求点击屏幕坐标[{x},{y}]位置，这是一个精确的坐标点击操作。"
                else:
                    return f"用户要求执行点击操作：{action}"
            elif "点击" in action:
                element_name = action.replace("点击", "").strip()
                return f"用户要求点击{element_name}，这是一个UI元素点击操作。"
            elif "输入" in action:
                text = action.replace("输入", "").strip()
                return f"用户要求在输入框中输入\"{text}\"，这是一个文本输入操作。"
            elif "滑动" in action or "滚动" in action:
                return f"用户要求执行滑动操作：{action}，这是一个屏幕滑动操作。"
            else:
                return f"用户要求执行操作：{action}，这是第{step_index + 1}步操作。"
                
        except Exception as e:
            logging.warning(f"生成动作描述失败: {e}")
            return f"执行操作：{action}"
    
    def extract_noun_from_action(self, action: str) -> str:
        """从动作中提取名词作为文件名"""
        try:
            # 移除动作前缀，提取名词
            action = action.strip()
            
            # 处理常见的动作模式
            if "打开" in action:
                # "打开小红书" -> "小红书"
                noun = action.replace("打开", "").strip()
            elif "点击" in action:
                # "点击搜索按钮" -> "搜索按钮"
                # "点击搜索框" -> "搜索框" 
                # "点击设置按钮" -> "设置"
                noun = action.replace("点击", "").strip()
                
                # 特殊处理：如果以"按钮"结尾，根据内容决定是否保留
                if noun.endswith("按钮"):
                    # 对于"设置按钮" -> "设置"，"搜索按钮" -> "搜索"
                    if noun in ["设置按钮", "搜索按钮", "返回按钮", "确认按钮", "取消按钮"]:
                        noun = noun.replace("按钮", "").strip()
                    # 对于其他按钮，保留"按钮"后缀
                    # 例如："登录按钮" -> "登录按钮"
            elif "输入" in action:
                # "输入文本" -> "输入"
                noun = "输入"
            elif "滑动" in action or "滚动" in action:
                # "滑动屏幕" -> "滑动"
                noun = "滑动"
            else:
                # 默认情况，取动作的后半部分
                noun = action
            
            # 清理文件名中的非法字符
            illegal_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            for char in illegal_chars:
                noun = noun.replace(char, '_')
            
            # 如果名称为空或太短，使用默认名称
            if not noun or len(noun) < 2:
                noun = "action"
            
            return noun
            
        except Exception as e:
            logging.warning(f"提取名词失败: {e}, 使用默认名称")
            return "action"

    def run_bfs_exploration(self):
        """运行广度优先搜索探索"""
        logging.info("开始BFS探索...")
        
        # 初始化设备
        if not self.initialize_device():
            logging.error("设备初始化失败，退出探索")
            return
        
        # 测试API连接
        #if not self.test_api_connection():
        #    logging.error("API连接测试失败，退出探索")
        #    return
        
        # 创建初始轨迹节点
        initial_node = TrajectoryNode(
            path=[f"请执行以下操作：打开{self.config.app_name}"],
            current_page=f"{self.config.app_name}首页",
            clickable_elements=[],
            depth=0,
            status="pending",
            page_signature="",
            timestamp=datetime.now().isoformat()
        )
        
        # 将初始节点加入队列
        self.exploration_queue.put(initial_node)
        
        # BFS主循环
        while not self.exploration_queue.empty() and len(self.completed_trajectories) < self.config.max_trajectories:
            try:
                # 从队列取出节点
                current_node = self.exploration_queue.get()
                logging.info(f"探索节点: {current_node.path}, 深度: {current_node.depth}")
                
                # 探索当前页面
                new_nodes = self.explore_page(current_node)
                
                # 保存当前轨迹
                self.save_trajectory(current_node)
                self.completed_trajectories.append(current_node)
                
                # 重新初始化环境，确保下一个任务基于全新环境
                if self.config.reset_environment_per_task:
                    if not self.reinitialize_environment():
                        logging.warning("环境重新初始化失败，但继续执行下一个任务")
                
                # 将新节点加入队列
                for new_node in new_nodes:
                    self.exploration_queue.put(new_node)
                    logging.info(f"添加新轨迹: {new_node.path}")
                
                # 检查深度限制
                if current_node.depth >= self.config.max_depth:
                    logging.info(f"达到最大深度 {self.config.max_depth}，停止探索")
                    break
                    
            except Exception as e:
                logging.error(f"BFS循环出错: {e}")
                continue
        
        # 探索完成
        logging.info(f"BFS探索完成，共探索 {len(self.completed_trajectories)} 个轨迹")

    def reinitialize_environment(self):
        """重新初始化环境，确保每个任务基于全新环境"""
        try:
            # 1. 关闭当前连接
            if self.client:
                self.client.close()
                logging.info("已关闭当前client连接")
            
            # 2. 等待一小段时间确保连接完全关闭
            time.sleep(self.config.reset_delay)
            
            # 3. 重新初始化设备连接
            if not self.initialize_device():
                logging.error("环境重新初始化失败")
                return False
                
            # 4. 测试新连接
            #if not self.test_api_connection():
            #    logging.error("重新初始化后的API连接测试失败")
            #    return False
                
            logging.info("✅ 环境重新初始化成功")
            return True
            
        except Exception as e:
            logging.error(f"环境重新初始化失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        try:
            if self.client:
                self.client.close()
            logging.info("资源清理完成")
        except Exception as e:
            logging.error(f"资源清理失败: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="修复版BFS应用探索器")
    
    parser.add_argument("--server-name", type=str, default="http://localhost:7880/")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--app-name", type=str, default="小红书")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-trajectories", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="trajectories")
    parser.add_argument("--delay", type=float, default=2.0)
    
    return parser.parse_args()


if __name__ == "__main__":
    # 记录开始时间
    start_time = datetime.now()
    
    args = parse_args()
    
    # 创建配置
    config = ExplorationConfig(
        max_depth=args.max_depth,
        max_trajectories=args.max_trajectories,
        app_name=args.app_name,
        output_dir=args.output_dir,
        delay_between_actions=args.delay,
        model_name=args.model_name
    )
    
    # 创建探索器
    explorer = BFSEplorer(config, args.server_name)
    
    try:
        # 运行探索
        explorer.run_bfs_exploration()
    except KeyboardInterrupt:
        logging.info("用户中断探索")
    except Exception as e:
        logging.error(f"探索过程出错: {e}")
    finally:
        # 清理资源
        explorer.cleanup()
    
    # 计算总耗时
    end_time = datetime.now()
    total_duration = end_time - start_time
    logging.info(f"总耗时: {total_duration}")
