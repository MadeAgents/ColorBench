
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

class ReflectorAgent:
    """Reflection Agent responsible for analyzing action results and providing feedback"""
    
    def __init__(self, agent_config):
        self.agent_config = agent_config
        self.model = agent_config['model']
        self.api_key = agent_config['api_key']
        self.base_url = agent_config['base_url']
        self.temperature = agent_config.get('temperature', 0.1)
        self.task = None
        self.reflection_history = []
        
    def set_task(self, task):
        """Set the current task"""
        self.task = task
        self.reflection_history = []
        
    def update_history(self, reflection_result, action, action_description, memory=None):
        """Update reflection history"""
        self.reflection_history.append({
            'reflection_result': reflection_result,
            'action': action,
            'action_description': action_description,
            'memory': memory
        })
        
    def reflect_on_action(self, current_image_path, pre_image_path, action_plan, action, action_description):
        """Reflect on the executed action and its results"""
        try:
            # Read and encode current image
            with open(current_image_path, "rb") as image_file:
                cur_encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            # Read and encode previous image if available
            with Image.open(pre_image_path) as img:
                pre_img_width, pre_img_height = img.size
            with open(pre_image_path, "rb") as image_file:
                pre_encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
            # Build reflection history context 
            # history_context = ""
            # if self.execution_history:
            #     history_context = "Previous execution history:\n"
            #     for i, step in enumerate(self.execution_history, 1):
            #         history_context += f"Step {i}: {step['action_description']}\n"
            #         if step['result']:
            #             history_context += f"Result: {step['result']}\n"
            # if self.reflection_history:
            #     history_context += "Previous reflection history:\n"
            #     for i, reflection in enumerate(self.reflection_history, 1):
            #         history_context += f"Step {i} Reflection: {reflection.get('success_evaluation', 'N/A')}\n"
            #         history_context += f"Progress: {reflection.get('progress_assessment', 'N/A')}\n"
           
            # Build reflection prompt
            reflection_prompt_before = f"""You are a reflective agent in a GUI intelligent system. Given the user query, the previous step's task plan and action, as well as the screenshots before and after the actions, you need to analyze whether any errors occurred in this step from three aspects: task objective, task planning, and task execution. If errors are found, you need to provide improvement suggestions for both the task planning and task execution.  Please ensure that your output strictly adheres to the format requirements.

### Background
The user query: {self.task}
Task plan: {action_plan}
Action execution: {action} - {action_description}
Previous Screen resolution: {pre_img_width}x{pre_img_height}
The changes between the two screenshots should correspond to the given plan and action. Otherwise, it indicates an error and you should analyze the cause and propose solutions in reflection. The previous screen and current screen are as follows:"""

            reflection_prompt_after = f"""
### Response Format
<reasoning>
[Provide a brief analysis of the action result, including any errors found and their possible causes (max 100 words)]
</reasoning>
<planning_reflection>
[If planning errors were found, provide improvement suggestions for the task planning (max 50 words). If no errors, state "No issues found in planning."]
</planning_reflection>
<execution_reflection>
[If execution errors were found, provide improvement suggestions for the task execution (max 50 words). If no errors, state "No issues found in execution."]
</execution_reflection>"""

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are an expert mobile GUI automation analyst. Analyze action results and reflect separately on task planning and task execution."}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": reflection_prompt_before},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pre_encoded_string}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{cur_encoded_string}"}},
                        {"type": "text", "text": reflection_prompt_after}
                    ]
                }
            ]
            
            logger.info(f"Reflection agent analyzing: {pre_image_path} and {current_image_path}")
            logger.info(f"Reflection agent prompt:\nYou are an expert mobile GUI automation analyst. Analyze action results and provide constructive feedback for improvement.\n{reflection_prompt_before}...[images]...{reflection_prompt_after}")
            response = get_response(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=self.temperature
            )
            
            if response:
                logger.info(f"Reflection agent raw response: {response}")
                parsed_reflection, error = self._parse_reflection_response(response)
                logger.info(f"Reflection agent response: {parsed_reflection}")
                return parsed_reflection, error
            else:
                return None, "Failed to generate reflection"
                
        except Exception as e:
            logger.error(f"Error in reflection agent: {str(e)}")
            return None, f"Reflection error: {str(e)}"
    
    def _parse_reflection_response(self, response):
        """Parse the reflection response into structured components"""
        try:
            # Extract action analysis
            analysis_match = re.search(r'<reasoning>\n(.*?)\n</reasoning>', response, re.DOTALL)
            action_analysis = analysis_match.group(1).strip() if analysis_match else "No action_analysis provided"

            # Extract planning reflection
            planning_match = re.search(r'<planning_reflection>\n(.*?)\n</planning_reflection>', response, re.DOTALL)
            planning_reflection = planning_match.group(1).strip() if planning_match else "No planning_reflection described"

            # Extract execution reflection
            execute_match = re.search(r'<execution_reflection>\n(.*?)\n</execution_reflection>', response, re.DOTALL)
            execution_reflection = execute_match.group(1).strip() if execute_match else "No execution_reflection provided"
            
            return {
                'action_analysis': action_analysis,
                'planning_reflection': planning_reflection,
                'execution_reflection': execution_reflection
            }, None
            
        except Exception as e:
            logger.error(f"Error parsing reflection response: {str(e)}")
            return None, f"Failed to parse reflection response: {str(e)}"
    
    def get_reflection_summary(self):
        """Get a summary of all reflections for the current task"""
        if not self.reflection_history:
            return "No reflections available"
        
        summary = "Reflection Summary:\n"
        for i, reflection in enumerate(self.reflection_history, 1):
            summary += f"\nStep {i}:\n"
            summary += f"- Success: {reflection.get('success_evaluation', 'N/A')}\n"
            summary += f"- Progress: {reflection.get('progress_assessment', 'N/A')}\n"
        
        return summary
