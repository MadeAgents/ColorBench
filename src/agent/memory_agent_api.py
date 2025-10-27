from openai import OpenAI
from zai import ZhipuAiClient
import os
import base64
import json
import time
from PIL import Image
import logging
import re

logger = logging.getLogger(__name__)

MAX_RETRIES = 5

def get_response(messages, temperature=0.1, top_k=5, top_p=0.9):
    # top_k越小越确定，top_p越大越多样(一般不会太大）

    client = ZhipuAiClient(api_key="")
    # print(f"Using Zhipu AI Client with API Key: {client.api_key}")
    retries = 0
    retry_delay = 2  # 初始重试延迟时间（秒）
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

class MemoryAgentGLM:
    """Memory Agent responsible for storing and recalling information"""
    
    def __init__(self, agent_config):
        self.agent_config = agent_config
        self.model = agent_config['model']
        self.api_key = agent_config['api_key']
        self.base_url = agent_config['base_url']
        self.temperature = agent_config.get('temperature', 0.1)
        self.task = None
        self.execution_history = []
        
    def set_task(self, task):
        """Set the current task"""
        self.task = task
        self.execution_history = []
        
    def update_history(self, planning_result, action, action_description, memory=None):
        """Update execution history"""
        self.execution_history.append({
            'planning_result': planning_result,
            'action': action,
            'action_description': action_description,
            'memory': memory
        })
        
    def get_memory(self, image_path, cur_planning = None, action = None, action_description = None):
        """Generate a plan for the next action based on current state"""
        try:
            # Read and encode image
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
            # # Build history context - 使用完整历史轨迹
            # history_context = ""
            # history_memory = ""
            # if self.execution_history:
            #     for i, step in enumerate(self.execution_history, 1):  # 完整历史轨迹
            #         history_context += f"Step {i}: Action: {step['action']}; Action description: {step['action_description']}\n"
            #         if step['memory']:
            #             history_memory += f"{i}. {step['memory']}\n"
            # if history_context:
            #     history_context = history_context.rstrip('\n')
            # if history_memory:
            #     history_memory = history_memory.rstrip('\n')

            # Create planning prompt
            # You are a Planning Agent. You task is to Analyze the screen and plan the next action based on the complete execution history and last reflection result. 你是一个GUI智能体系统中的记忆智能体。给定用户任务，上一步任务规划以及当前屏幕截图，你需要是为未来的操作记住的重要信息。记忆内容应仅限于将来操作中会使用的屏幕内容。它应满足以下条件：没有这些记忆，你无法确定一个或多个未来的操作。
            # 2. Screen resolution: {img_width}x{img_height}
            if not cur_planning and action_description:
                cur_planning = action_description
            memory_prompt = f"""
# Role: 
You are a GUI Agent, and your primary task is to respond accurately to user requests or questions. In addition to directly answering the user's queries, you can also use tools or perform GUI operations directly until you fulfill the user's request or provide a correct answer. You should carefully read and understand the images and questions provided by the user, and engage in thinking and reflection when appropriate. 

# Background
1. The user query: {self.task}
2. The current action plan: {cur_planning if cur_planning else '[no planning available]'}
3. The current action: {action if action else '[unknown]'}

# Output Format
Memory: important information you want to remember for the future actions. The memory should be only contents on current screen that will be used in the future actions. It should satisfy that: you cannnot determine one or more future actions without this memory. If no memory is needed in current screen, output **None**.

Your answer should look like:
Memory: ...
"""

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a memory agent in a GUI intelligent system. Given the user's task, the current task planning, and the current screen, you need to extract important information from screen you want to remember for the future actions"}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": memory_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}
                    ]
                }
            ]
            
            logger.info(f"Memorizing agent analyzing: {image_path}")
            logger.info(f"Memorizing agent prompt:\nYou are a memory agent in a GUI intelligent system. Given the user's task, the current task planning, and the current screen, you need to remember important information for future operations.\n{memory_prompt}")
            response = get_response(messages=messages)
            
            if response:
                logger.info(f"Memorizing agent raw response: {response}")
                parsed_memory, error = self._parse_memorizing_response(response)
                logger.info(f"Memorizing agent response: {parsed_memory}")
                return parsed_memory, error
            else:
                return None, "Failed to generate memory"
                
        except Exception as e:
            logger.error(f"Error in memory agent: {str(e)}")
            return None, f"Memorizing error: {str(e)}"

    def _parse_memorizing_response(self, response):
        """Parse the memorizing response into structured components"""
        try:
            # Extract memory
            pattern = r"Memory:(.*?)"
            match = re.search(pattern, response, re.DOTALL)
            if not match:
                return None, None
            memory = match.group(1).strip()

            return memory if "None" not in memory else None, None

        except Exception as e:
            logger.error(f"Error parsing memorizing response: {str(e)}")
            return None, f"Failed to parse memorizing response: {str(e)}"
