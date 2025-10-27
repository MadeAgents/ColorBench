from openai import OpenAI
import os
import base64
import json
import time
import threading
from PIL import Image
import logging
from src.agent.planner_agent import PlannerAgent
from src.agent.executor_agent import ExecutorAgent
from src.agent.reflector_agent import ReflectorAgent
from src.agent.memory_agent import MemoryAgent
from src.agent.memory_agent_api import MemoryAgentGLM

logger = logging.getLogger(__name__)

class PlanReflectAgent:
    """Integrated Agent that combines Planning, Execution, and Reflection in a three-step process"""
    
    def __init__(self, agent_config):
        self.agent_config = agent_config
        self.task = None
        self.use_plan = agent_config.get('plan', True)
        self.use_reflect = agent_config.get('reflect', True)
        self.use_memory = agent_config.get('memory', False)
        self.glm = agent_config.get('glm', False)
        print(f"PlanReflectAgent initialized with plan: {self.use_plan} and {type(self.use_plan)}, reflect: {self.use_reflect} and {type(self.use_reflect)}, memory: {self.use_memory} and {type(self.use_memory)}")
        # Initialize the three specialized agents
        if self.use_plan:
            self.planner = PlannerAgent(agent_config)
        self.executor = ExecutorAgent(agent_config)
        if self.use_reflect:
            self.reflector = ReflectorAgent(agent_config)
        if self.use_memory:
            if self.glm:
                print("Using GLM model for MemoryAgent")
                self.memory = MemoryAgentGLM(agent_config)
            else:
                self.memory = MemoryAgent(agent_config)
            # raise NotImplementedError("Memory functionality is not implemented in PlanReflectAgent yet")
        
        # Step tracking (thread-safe)
        self.current_step = 0
        self.step_history = []
        
        # Thread safety
        self._lock = threading.Lock()
        
    def set_task(self, task):
        """Set the current task for all agents"""
        self.task = task
        if self.use_plan:
            self.planner.set_task(task)
        if self.use_reflect:
            self.reflector.set_task(task)
        if self.use_memory:
            self.memory.set_task(task)
        self.executor.set_task(task)
        self.current_step = 0
        self.step_history = []
        
    def agent_step(self, image_path):
        """Execute one complete step: Plan -> Execute -> Reflect (thread-safe)"""
        with self._lock:
            try:
                self.current_step += 1  # 第一步就是1
                thread_id = threading.current_thread().ident
                logger.info(f"Thread {thread_id}: === Starting Step {self.current_step} ===")
                step_info = {
                    'step_number': self.current_step,
                    'screenshot': image_path}
                
                # Step 0: Reflecting on previous action result (if any)
                if self.use_reflect and self.current_step > 1:
                    logger.info(f"Thread {thread_id}: Reflecting...")
                    pre_image_path = self.step_history[-1]['screenshot']
                    if self.use_plan:
                        inter_planning = self.step_history[-1]['planning']
                    else:
                        inter_planning = 'No action plan'
                    inter_executed_action = self.step_history[-1]['execution']['action']
                    inter_action_description = self.step_history[-1]['execution']['description']
                    reflection_result, reflection_error = self.reflector.reflect_on_action(
                        image_path,
                        pre_image_path,
                        inter_planning,
                        inter_executed_action,
                        inter_action_description,
                    )
                    
                    if reflection_error:
                        logger.warning(f"Thread {thread_id}: Reflection failed: {reflection_error}")
                        reflection_result = None
                        # return None, None, f"Reflection error: {reflection_error}"
                    else:
                        step_info['planning_reflection'] = reflection_result.get('planning_reflection', 'No planning reflection')
                        step_info['execution_reflection'] = reflection_result.get('execution_reflection', 'No execution reflection')


                # Step 1: Planning
                if self.use_plan:
                    print(self.use_plan)
                    logger.info(f"Thread {thread_id}: Step 1: Planning...")
                    if self.use_reflect and self.current_step > 1:
                        planning_reflection = reflection_result.get('planning_reflection', None) if reflection_result else None
                    planning_result, planning_error = self.planner.plan_next_action(image_path, planning_reflection if self.use_reflect and self.current_step > 1 else None)  # 上一步的反思内容，过往的action和action description

                    if planning_error:
                        logger.error(f"Thread {thread_id}: Planning failed: {planning_error}")
                        planning_result = None
                        # return None, None, f"Planning error: {planning_error}"
                    else:
                        step_info['planning'] = planning_result.get('action_plan', 'No action plan')
                

                # Step 2: Execution
                logger.info(f"Thread {thread_id}: Step 2: Executing...")
                if self.use_reflect and self.current_step > 1:
                    execution_reflection = reflection_result.get('execution_reflection', None) if reflection_result else None
                if self.use_plan:
                    planning_context = planning_result.get('action_plan', None) if planning_result else None
                executed_action, action_description = self.executor.execute_action(
                    image_path, 
                    action_plan = planning_context if self.use_plan else None,
                    reflection = execution_reflection if self.use_reflect and self.current_step > 1 else None
                )
                
                if not executed_action:
                    logger.error(f"Thread {thread_id}: Execution failed")
                    return None, None, "Execution failed"
                else:
                    step_info['execution'] = {
                        'action': executed_action,
                        'description': action_description
                    }
                    
                if self.use_memory:
                    planning_context = planning_result.get('action_plan', None) if self.use_plan and planning_result else None
                    memory_content, memory_error = self.memory.get_memory(image_path, planning_context, action=executed_action, action_description=action_description)
                    # 给当前任务和当前截图，输出一些关键的信息，存入当前任务的记忆库
                    # raise NotImplementedError("Memory functionality is not implemented in PlanReflectAgent yet")
                    if memory_error:
                        logger.error(f"Thread {thread_id}: Memory generation failed: {memory_error}")
                        memory_content = None
                    else:
                        step_info['memory'] = memory_content
                
                self.step_history.append(step_info)
                
                # Update agent histories
                memory = memory_content if self.use_memory and memory_content else None
                if self.use_plan:
                    self.planner.update_history(planning_result=planning_result, action=executed_action, action_description=action_description, memory=memory)
                self.executor.update_history(action=executed_action, action_description=action_description, memory=memory)
                if self.use_reflect and self.current_step > 1:
                    self.reflector.update_history(reflection_result,action=executed_action, action_description=action_description, memory=memory)
                # if self.use_memory:
                #     raise NotImplementedError("Memory functionality is not implemented in PlanReflectAgent yet")
                
                # Log step completion
                # logger.info(f"Thread {thread_id}: === Step {self.current_step} Completed ===")
                # logger.info(f"Thread {thread_id}: Planned: {planning_result['action_plan']}")
                # logger.info(f"Thread {thread_id}: Executed: {action_description}")
                # if reflection_result:
                #     logger.info(f"Thread {thread_id}: Reflected: {reflection_result['success_evaluation']}")
                logger.info(f"Thread {thread_id}: Step {self.current_step} Completed.")
                logger.info(f"Thread {thread_id}: Step Information: {json.dumps(step_info, ensure_ascii=False, indent=4)}")
                # logger.info(json.dumps(step_info, ensure_ascii=False, indent=2))
                
                return executed_action, action_description, None
                
            except Exception as e:
                logger.error(f"Thread {thread_id}: Error in PlanReflectAgent step: {str(e)}")
                return None, None, f"Agent step error: {str(e)}"
    
    def _build_planning_context(self, planning_result):
        """Build context string from planning result"""
        if not planning_result:
            return "No planning context available"
        
        context = f"""Planning Context:
Observation: {planning_result.get('observation', 'N/A')}
Action Plan: {planning_result.get('action_plan', 'N/A')}
Reasoning: {planning_result.get('reasoning', 'N/A')}
Expected Outcome: {planning_result.get('expected_outcome', 'N/A')}"""
        
        return context
    
    def get_step_summary(self, step_number=None):
        """Get summary of a specific step or all steps"""
        if step_number is None:
            # Return summary of all steps
            summary = f"Task: {self.task}\n"
            summary += f"Total Steps: {len(self.step_history)}\n\n"
            
            for step in self.step_history:
                summary += f"Step {step['step_number']}:\n"
                summary += f"  Planned: {step['planning']['action_plan']}\n"
                summary += f"  Executed: {step['execution']['description']}\n"
                if step['reflection']:
                    summary += f"  Success: {step['reflection']['success_evaluation']}\n"
                summary += "\n"
            
            return summary
        else:
            # Return summary of specific step
            if step_number <= 0 or step_number > len(self.step_history):
                return f"Step {step_number} not found"
            
            step = self.step_history[step_number - 1]
            summary = f"Step {step_number} Summary:\n"
            summary += f"Planned: {step['planning']['action_plan']}\n"
            summary += f"Executed: {step['execution']['description']}\n"
            if step['reflection']:
                summary += f"Success: {step['reflection']['success_evaluation']}\n"
                summary += f"Progress: {step['reflection']['progress_assessment']}\n"
            
            return summary
    
    def get_task_progress(self):
        """Get overall task progress assessment"""
        if not self.step_history:
            return "No steps completed yet"
        
        # Get the latest reflection for progress assessment
        latest_step = self.step_history[-1]
        if latest_step['reflection']:
            return latest_step['reflection']['progress_assessment']
        else:
            return "Progress assessment not available"
    
    def get_improvement_suggestions(self):
        """Get improvement suggestions from all reflections"""
        suggestions = []
        for step in self.step_history:
            if step['reflection'] and step['reflection'].get('improvement_suggestions'):
                suggestions.append(f"Step {step['step_number']}: {step['reflection']['improvement_suggestions']}")
        
        if suggestions:
            return "\n".join(suggestions)
        else:
            return "No improvement suggestions available"
    
    def get_next_steps_recommendation(self):
        """Get next steps recommendation from latest reflection"""
        if not self.step_history:
            return "No steps completed yet"
        
        latest_step = self.step_history[-1]
        if latest_step['reflection'] and latest_step['reflection'].get('next_steps_recommendation'):
            return latest_step['reflection']['next_steps_recommendation']
        else:
            return "No next steps recommendation available"
