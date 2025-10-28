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

def position_to_direction(x1, y1, x2, y2):
    """Convert position coordinates to direction"""
    if x1 == x2 and y1 == y2:
        logger.info("Start and end points are the same, cannot determine direction")
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

class ExecutorAgent:
    """Execution Agent responsible for executing planned actions on mobile GUI"""
    
    def __init__(self, agent_config):
        self.agent_config = agent_config
        self.model = agent_config['model']
        self.api_key = agent_config['api_key']
        self.base_url = agent_config['base_url']
        self.temperature = agent_config.get('temperature', 0.1)
        self.system_prompt = agent_config['system_prompt']
        self.task = None
        self.execution_history = []
        
    def set_task(self, task):
        """Set the current task"""
        self.task = task
        self.execution_history = []
        
    def update_history(self, action, action_description, memory=None):
        """Update execution history"""
        self.execution_history.append({
            'action': action,
            'action_description': action_description,
            'memory': memory
        })
        
    def execute_action(self, image_path, action_plan=None, reflection=None):
        """Execute the planned action based on current screen state"""
        try:
            # Read and encode image
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Build execution history context
            history_context = ""
            history_memory = ""
            if self.execution_history:
                for i, step in enumerate(self.execution_history, 1): 
                    history_context += f"Step {i}: Action: {step['action']}; Action description: {step['action_description']}\n"
                    if step['memory']:
                        history_memory += f"    ({i}). {step['memory']}\n"
            if history_context:
                history_context = history_context.rstrip('\n')
            if history_memory:
                history_memory = history_memory.rstrip('\n')

            # Build execution prompt
            execution_prompt = f"""You are an action-executing agent in a GUI intelligent system. You need to output actions that can be executed directly based on the action plan provided by the planning agent to accomplish the task instructions given by the user. Please strictly follow the format requirements and output a brief action description after each action.

### Background Information 
1. The user query: {self.task}
2. Screen resolution: {img_width}x{img_height}
3. Current action planning: {action_plan if action_plan else '[no plan available]'}
4. Task progress: The executor has done the following operation on the current device
{history_context if history_context else '[First step and no prior actions taken.]'}
5. Last execution reflection: {reflection if reflection else '[no reflection available]'}
6. Task history memory:
{history_memory if history_memory else 'Empty, ignore it'}

Based on the above information, please analyze the current screen and output action that strictly adhere to the format and can be executed directly to complete the user task.
### Output Format
<action>
{{"name": "mobile_use", "arguments": <args-json-object>}}
</action>
<description>
[briefly describe your action]
</description>"""

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt.format(width=img_width, height=img_height)}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": execution_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}
                    ]
                }
            ]
            
            logger.info(f"Execution agent executing: {image_path}")
            logger.info(f"Execution agent prompt:\n{self.system_prompt.format(width=img_width, height=img_height)}\n{execution_prompt}")
            response = get_response(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature
            )
            
            if response:
                logger.info(f"Execution agent raw response: {response}")
                action, action_description = self._parse_execution_response(response)
                logger.info(f"Execution agent action: {action}, description: {action_description}")
                parsed_action = self._parse_user_input(action)
                logger.info(f"Parsed agent action: {parsed_action}")
                return parsed_action, action_description
            else:
                return None, "Failed to execute action"
                
        except Exception as e:
            logger.error(f"Error in execution agent: {str(e)}")
            return None, f"Execution error: {str(e)}"
    
    def _parse_execution_response(self, response):
        """Parse the execution response to extract action and description"""
        try:
            # Extract action
            action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
            if action_match:
                action = action_match.group(1).strip()
            else:
                action = re.search(r'\{.*\}', response)
                action = action.group(0).strip() if action else None
            
            # Extract description
            description_match = re.search(r'<description>\n(.*?)\n</description>', response, re.DOTALL)
            action_description = description_match.group(1).strip() if description_match else "No description provided"
            
            return action, action_description
            
        except Exception as e:
            logger.error(f"Error parsing execution response: {str(e)}")
            return None, f"Failed to parse execution response: {str(e)}"
    
    def _parse_user_input(self, input_str):
        """Parse user input format into action dictionary"""
        try:
            if not input_str:
                return None
                
            input_str = input_str.replace("{{", "{")
            match_s = re.search(r'"arguments": (\{.*\})\}', input_str)
            if match_s:
                action_s = match_s.group(1).strip()
            else:
                action_s = input_str

            action_s = action_s.rstrip('}').strip() + '}'
            action_s = action_s.replace("，", ",").replace("：", ":").replace("“", "\"").replace("”", "\"")
            logger.info(f"Parsing action string: {action_s}")
            action = json.loads(action_s)
            action_type = action.get('action', 'None')
            
            if action_type == 'None':
                action_type = action.get('name', 'None')
                action.pop('name', None)
            else:
                action.pop('action', None)
                
            if action_type == 'None':
                action_type = 'wait'
                action = action.get('arguments', action)
            
            params = action
            result = {'action_type': action_type}
            
            if action_type in ['click', 'long_press']:
                coordinate = params.get('coordinate', None)
                if coordinate:
                    result['x'] = coordinate[0]
                    result['y'] = coordinate[1]
                    
            elif action_type == 'swipe':
                direction = params.get('direction', None)
                if direction:
                    result['direction'] = direction.lower()
                else:
                    coordinate1 = params.get('coordinate', None)
                    coordinate2 = params.get('coordinate2', None)
                    if coordinate1 and coordinate2:
                        result['direction'] = position_to_direction(
                            coordinate1[0], coordinate1[1],
                            coordinate2[0], coordinate2[1]
                        )
                    
            elif action_type == 'system_button':
                result['button'] = params.get('button', '').lower()
                
            elif action_type == 'type':
                result['text'] = params.get('text', '')
                
            elif action_type == 'open':
                result['app'] = params.get('text', '') or params.get('app', '')
                
            elif action_type == 'terminate':
                result['action_type'] = 'complete'
                result['status'] = params.get('status', '')
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing user input: {str(e)}")
            return None
