from openai import OpenAI
import os
import base64
import json
import time
from PIL import Image
import logging
import re

logger = logging.getLogger(__name__)

MAX_RETRIES = 5

def get_response(model, messages, api_key, base_url, temperature=0.1):
    """Get response from LLM with retry mechanism"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    retries = 0
    retry_delay = 2
    
    while retries <= MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
            ).choices[0].message.content.strip()
            return response
        except Exception as e:
            logger.warning(f"Request failed, retrying... Error: {str(e)}")
            retries += 1
            time.sleep(retry_delay)
    
    logger.error("Request failed after multiple retries.")
    return None

class MemoryAgent:
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
# Background
1. The user query: {self.task}
2. The current action plan: {cur_planning if cur_planning else '[no planning available]'}
3. The current action: {action if action else '[unknown]'}

# Instruction
Memory: important information you want to remember for the future actions. The memory should be only contents on current screen that will be used in the future actions. It should satisfy that: you cannnot determine one or more future actions without this memory. If no memory is needed in current screen, output **None**.

# Response Format
<memory>
[important information you want to remember for the future actions. If no memory is needed in current screen, output "None".]
</memory>"""

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a memory agent in a GUI intelligent system. Given the user's task, the current task planning, and the current screen, you need to remember important information for future operations."}
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
            response = get_response(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature
            )
            
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
            memory_match = re.search(r'<memory>\n(.*?)\n</memory>', response, re.DOTALL)
            memory = memory_match.group(1).strip() if memory_match else "No memory provided"

            return memory if memory.replace('"', '') != "None" else None, None

        except Exception as e:
            logger.error(f"Error parsing memorizing response: {str(e)}")
            return None, f"Failed to parse memorizing response: {str(e)}"
