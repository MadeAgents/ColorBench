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

import threading
import logging
from src.agent.planner_agent import PlannerAgent
from src.agent.executor_agent import ExecutorAgent
from src.agent.reflector_agent import ReflectorAgent
from src.agent.plan_reflect_agent import PlanReflectAgent
from src.agent.agent import VanillaAgent
from src.agent.memory_agent import MemoryAgent
from src.test.graph_tools_ma import Graph_DataSet

logger = logging.getLogger(__name__)

class ThreadSafeAgentFactory:
    """线程安全的智能体工厂类，为每个线程创建独立的智能体实例"""
    
    def __init__(self, agent_configs):
        """
        初始化工厂
        :param agent_configs: 智能体配置字典
        """
        self.agent_configs = agent_configs
        self._local = threading.local()
    
    def get_agent(self, mode, model_name):
        """
        获取线程安全的智能体实例
        :param mode: 智能体模式 (vanilla, memory, thought, plan-reflect)
        :param model_name: 模型名称
        :return: 智能体实例
        """

        thread_id = threading.current_thread().ident
        
        if not hasattr(self._local, 'agents'):
            self._local.agents = {}
        
        agent_key = f"{mode}_{model_name}_{thread_id}"
        
        if agent_key not in self._local.agents:
            logger.info(f"Creating new agent instance for thread {thread_id}, mode: {mode}")
            
            if mode == 'plan-reflect':
                agent = PlanReflectAgent(self.agent_configs['plan-reflect'])
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
            self._local.agents[agent_key] = agent
        
        return self._local.agents[agent_key]
    
    def clear_thread_agents(self):
        """清理当前线程的智能体实例"""
        if hasattr(self._local, 'agents'):
            self._local.agents.clear()
            logger.info(f"Cleared agents for thread {threading.current_thread().ident}")

class ThreadSafeGraphDataSet:
    """线程安全的图数据集类"""
    
    def __init__(self, graph_config):
        """
        初始化图数据集
        :param graph_config: 图配置
        """
        self.graph_config = graph_config
        self._local = threading.local()
    
    def get_graph_dataset(self):
        """
        获取线程安全的图数据集实例
        :return: 图数据集实例
        """
        
        thread_id = threading.current_thread().ident
        
        if not hasattr(self._local, 'graph_dataset'):
            logger.info(f"Creating new Graph_DataSet instance for thread {thread_id}")
            self._local.graph_dataset = Graph_DataSet(self.graph_config)
        
        return self._local.graph_dataset
    
    def clear_thread_dataset(self):
        """清理当前线程的图数据集实例"""
        if hasattr(self._local, 'graph_dataset'):
            delattr(self._local, 'graph_dataset')
            logger.info(f"Cleared Graph_DataSet for thread {threading.current_thread().ident}")

class ThreadSafeTaskExecutor:
    """线程安全的任务执行器"""
    
    def __init__(self, agent_factory, graph_factory, config):
        """
        初始化任务执行器
        :param agent_factory: 智能体工厂
        :param graph_factory: 图数据集工厂
        :param config: 配置
        """
        self.agent_factory = agent_factory
        self.graph_factory = graph_factory
        self.config = config
        self.lock = threading.Lock()
    
    def execute_task(self, task_item, mode, model_name, output_dir, parent_dir, config_name):
        """
        执行单个任务
        :param task_item: 任务项
        :param mode: 智能体模式
        :param model_name: 模型名称
        :param output_dir: 输出目录
        :param parent_dir: 父目录
        :param config_name: 配置名称
        :return: 执行结果
        """
        thread_id = threading.current_thread().ident
        task_id = task_item.get('task_id', 'unknown')
        task = task_item['query']
        
        try:
            logger.info(f"Thread {thread_id}: Starting task {task_id}: {task}")
            
            agent = self.agent_factory.get_agent(mode, model_name)
            graph_dataset = self.graph_factory.get_graph_dataset()
            
            graph_dataset.set_task(task)
            agent.set_task(task)
            
            complete = False
            image_path = graph_dataset.home_page
            max_step = self.config['tasks']['max_steps']
            current_step = 0
            
            import time
            import os
            start_time = time.time()
            
            while not complete and current_step < max_step:
                image_path = os.path.join(parent_dir, image_path)
                
                if mode == 'plan-reflect':
                    # Plan-Reflect Agent returns (action, action_description, error)
                    action, action_description, error = agent.agent_step(image_path)
                    if error:
                        logger.error(f"Thread {thread_id}: Task {task_id} agent step failed: {error}")
                        complete = True
                        continue
                    
                    # Execute action in graph environment
                    image_path, answer = graph_dataset.step(
                        action, 
                        action_description=action_description,
                        action_step_info=agent.step_history[-1] if agent.step_history else None
                    )
                elif mode == 'thought':
                    # Thought Agent returns (thought, action, action_description)
                    thought, action, action_description = agent.agent_step(image_path)
                    image_path, answer = graph_dataset.step(
                        action, 
                        action_description=action_description,
                        action_plan=thought
                    )
                else:
                    # Other agents return (action, action_description)
                    action, action_description = agent.agent_step(image_path)
                    image_path, answer = graph_dataset.step(
                        action, 
                        action_description=action_description
                    )
                
                if answer:
                    logger.info(f"Thread {thread_id}: Task {task_id} completed! Answer: {answer}")
                    complete = True
                elif image_path is None:
                    logger.warning(f"Thread {thread_id}: Task {task_id} failed at step {current_step}")
                    complete = True
                current_step += 1
            
            use_time = time.time() - start_time
            logger.info(f"Thread {thread_id}: Task {task_id} finished. Steps: {current_step}, Time: {use_time:.2f}s")
            
            # Save trajectory
            graph_dataset.save_trajectory(
                output_dir, 
                use_time, 
                save_image=True, 
                config_name=config_name, 
                parent_dir=parent_dir,
                task_id=task_id
            )
            
            return {
                'task_id': task_id,
                'task': task,
                'success': complete and current_step < max_step,
                'steps': current_step,
                'time': use_time,
                'thread_id': thread_id
            }
            
        except Exception as e:
            logger.error(f"Thread {thread_id}: Task {task_id} failed with error: {str(e)}")
            return {
                'task_id': task_id,
                'task': task,
                'success': False,
                'error': str(e),
                'thread_id': thread_id
            }
        finally:
            # clear thread-specific resources
            self.agent_factory.clear_thread_agents()
            self.graph_factory.clear_thread_dataset()
