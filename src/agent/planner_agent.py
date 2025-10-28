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

class PlannerAgent:
    """Planning Agent responsible for generating action plans based on current state and task progress"""
    
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
        
    def plan_next_action(self, image_path, reflection_content = None):
        """Generate a plan for the next action based on current state"""
        try:
            # Read and encode image
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Build history context 
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
            
            planning_prompt = f"""You are a task-planning agent in a GUI intelligent system. Your task is to formulate the next action plan based on the given user task by analyzing the historical trajectory, the current screenshot, and possible task history memory, while referring to the reflection suggestions from the previous step. Please ensure that you output only one action plan and strictly adhere to the format requirements.

### Background Information 
1. The user query: {self.task}
2. Last step reflection: {reflection_content if reflection_content else '[no reflection available]'}
3. Task progress: The executor has done the following operation on the current device
{history_context if history_context else '[First step and no prior actions taken.]'}
4. Task history memory:
{history_memory if history_memory else 'Empty, ignore it'}

### Instructions
1. Based on the above information, please analyze the current screen and formulate the next action plan to complete the user task.
2. Available actions only include the following types: click[x,y], long_press[x,y], swipe[direction], type[text], system_button[back/home], open[app], wait, complete. You need to ensure that the planned actions are executable on the current screen.
3. Keep all responses under 50 words each. Be concise and direct. Consider the complete execution history as well as significant historical memory information when planning.

### Response Format
<reasoning>
[Your reasoning about planning]
</reasoning>
<action_plan>
[Use one sentence to describe the next action to be performed, including the key text information required to carry out this step.]
</action_plan>"""

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are an expert mobile GUI automation planner. Analyze screenshots and create strategic action plans."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": planning_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}
                    ]
                }
            ]
            
            logger.info(f"Planning agent analyzing: {image_path}")
            response = get_response(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature
            )
            
            if response:
                logger.info(f"Planning agent raw response: {response}")
                parsed_planning, error = self._parse_planning_response(response)
                logger.info(f"Planning agent response: {parsed_planning}")
                return parsed_planning, error
            else:
                return None, "Failed to generate plan"
                
        except Exception as e:
            logger.error(f"Error in planning agent: {str(e)}")
            return None, f"Planning error: {str(e)}"
    
    def _parse_planning_response(self, response):
        """Parse the planning response into structured components"""
        try:
            # Extract reasoning
            reasoning_match = re.search(r'<reasoning>\n(.*?)\n</reasoning>', response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"

            # Extract action plan
            action_plan_match = re.search(r'<action_plan>\n(.*?)\n</action_plan>', response, re.DOTALL)
            action_plan = action_plan_match.group(1).strip() if action_plan_match else "No action plan provided"
            
            return {                
                'reasoning': reasoning,
                'action_plan': action_plan
            }, None
            
        except Exception as e:
            logger.error(f"Error parsing planning response: {str(e)}")
            return None, f"Failed to parse planning response: {str(e)}"
