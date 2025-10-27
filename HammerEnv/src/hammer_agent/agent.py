import time
from absl import logging
from agents import Agent, Runner

import json
import re

from server.client import HammerEnvClient


PROMPT_PREFIX = """
You are an agent who can operate an Android phone on behalf of a user. Based on user's goal/request, you may
- Answer back if the request/goal is a question (or a chat message), like user asks "What is my schedule for today?".
- Complete some tasks described in the requests/goals by performing actions (step by step) on the phone.

When given a user request, you will try to complete it step by step. At each step, you will be given the current screenshot (the original screenshot) and a history of what you have done (in text). Based on these pieces of information and the goal, you must choose to perform one of the action in the following list (action description followed by the JSON format) by outputing the action in the correct JSON format.
- If you think the task has been completed, finish the task by using the status action with complete as goal_status: `{{"action_type": "status", "goal_status": "complete"}}`
- If you think the task is not feasible (including cases like you don't have enough information or can not perform some necessary actions), finish by using the `status` action with infeasible as goal_status: `{{"action_type": "status", "goal_status": "infeasible"}}`
- Answer user's question: `{{"action_type": "answer", "text": "<answer_text>"}}`
- Click/tap on an element on the screen. Use the coordinates to indicate which element you want to click: `{{"action_type": "click", "x": <target_x>, "y": <target_y>}}`.
- Long press on an element on the screen, similar with the click action above,use the coordinates to indicate which element you want to long press: `{{"action_type": "long_press", "x": <target_x>, "y": <target_y>}}`.
- Type text into a text field (this action contains clicking the text field, typing in the text and pressing the enter, so no need to click on the target field to start), use the coordinates to indicate the target text field: `{{"action_type": "input_text", "text": <text_input>, "x": <target_x>, "y": <target_y>}}`
- Press the Enter key: `{{"action_type": "keyboard_enter"}}`
- Navigate to the home screen: `{{"action_type": "navigate_home"}}`
- Navigate back: `{{"action_type": "navigate_back"}}`
- Scroll the screen or a scrollable UI element in one of the four directions, use the same coordinates as above if you want to scroll a specific UI element, leave it empty when scrolling the whole screen: `{{"action_type": "scroll", "direction": <up, down, left, right>, "x": <optional_target_x>, "y": <optional_target_y>}}`
- Open an app (nothing will happen if the app is not installed): `{{"action_type": "open_app", "app_name": <name>}}`
- Wait for the screen to update: `{{"action_type": "wait"}}`
"""

GUIDANCE = """
Here are some useful guidelines you need to follow:
General:
- Usually there will be multiple ways to complete a task, pick the easiest one. Also when something does not work as expected (due to various reasons), sometimes a simple retry can solve the problem, but if it doesn't (you can see that from the history), SWITCH to other solutions.
- Sometimes you may need to navigate the phone to gather information needed to complete the task, for example if user asks "what is my schedule tomorrow", then you may want to open the calendar app (using the `open_app` action), look up information there, answer user's question (using the `answer` action) and finish (using the `status` action with complete as goal_status).
- For requests that are questions (or chat messages), remember to use the `answer` action to reply to user explicitly before finish! Merely displaying the answer on the screen is NOT sufficient (unless the goal is something like "show me ...").
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), you can just complete the task.
Action Related:
- Use the `open_app` action whenever you want to open an app (nothing will happen if the app is not installed), do not use the app drawer to open an app unless all other ways have failed.
- Use the `input_text` action whenever you want to type something (including password) instead of clicking characters on the keyboard one by one. Sometimes there is some default text in the text field you want to type in, remember to delete them before typing.
- For `click`, `long_press` and `input_text`, the index parameter you pick must be VISIBLE in the screenshot and also in the UI element list given to you (some elements in the list may NOT be visible on the screen so you can not interact with them).
- Consider exploring the screen by using the `scroll` action with different directions to reveal additional content.
- The direction parameter for the `scroll` action can be confusing sometimes as it's opposite to swipe, for example, to view content at the bottom, the `scroll` direction should be set to "down". It has been observed that you have difficulties in choosing the correct direction, so if one does not work, try the opposite as well.
Text Related Operations:
- Normally to select certain text on the screen: <i> Enter text selection mode by long pressing the area where the text is, then some of the words near the long press point will be selected (highlighted with two pointers indicating the range) and usually a text selection bar will also appear with options like `copy`, `paste`, `select all`, etc. <ii> Select the exact text you need. Usually the text selected from the previous step is NOT the one you want, you need to adjust the range by dragging the two pointers. If you want to select all text in the text field, simply click the `select all` button in the bar.
- At this point, you don't have the ability to drag something around the screen, so in general you can not select arbitrary text.
- To delete some text: the most traditional way is to place the cursor at the right place and use the backspace button in the keyboard to delete the characters one by one (can long press the backspace to accelerate if there are many to delete). Another approach is to first select the text you want to delete, then click the backspace button in the keyboard.
- To copy some text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the `copy` button in bar.
- To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a `paste` button in it.
- When typing into a text field, sometimes an auto-complete dropdown list will appear. This usually indicating this is a enum field and you should try to select the best match by clicking the corresponding one in the list.
"""


ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + """
The current user goal/request is: {goal}

Here is a history of what you have done so far:
{history}

The current screenshot is also given to you.
"""
    + GUIDANCE
    + """
Now output an action from the above list in the correct JSON format, following the reason why you do that. Your answer should look like:
Reason: ...
Action: {{"action_type":...}}

Your Answer:
"""
)


SUMMARY_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + """
The (overall) user goal/request is: {goal}
Now I want you to summerize the latest step.
You will be given the screenshot before you performed the action (which has a text label "before" on the bottom right), the action you chose (together with the reason) and the screenshot after the action was performed (which has a text label "after" on the bottom right).
This is the action you picked: {action}
Based on the reason: {reason}

By comparing the two screenshots (plus the UI element lists) and the action performed, give a brief summary of this step. This summary will be added to action history and used in future action selection, so try to include essential information you think that will be most useful for future action selections like what you intended to do, why, if it worked as expected, if not what might be the reason (be critical, the action/reason might be wrong), what should/should not be done next and so on. Some more rules/tips you should follow:
- Keep it short (better less than 50 words) and in a single line
- Some actions (like `answer`, `wait`) don't involve screen change, you can just assume they work as expected.
- Given this summary will be added into action history, it can be used as memory to include information that needs to be remembered, or shared between different apps.

Summary of this step: 
"""
)


class Operator:
    def __init__(self, device_client: HammerEnvClient, max_steps=20):
        self.agent = Agent(name="Assistant", instructions="You are a helpful assistant")
        self.device_client = device_client
        self.max_steps = max_steps

    def run(self, task, run_config):
        tic = time.perf_counter()
        screenshot = self.device_client.init_task(task)
        toc = time.perf_counter()
        logging.debug(f"client init_task execution time: {toc - tic:.2f}s")

        step = 0
        history = []
        while step < self.max_steps:
            _history = [step["response"] for step in history]
            result = Runner.run_sync(
                starting_agent=self.agent,
                input=_input_msg(task, screenshot, _history),
                run_config=run_config,
            )
            response = result.final_output
            logging.info(response)

            action = None
            history.append(
                {
                    "observation": screenshot,
                    "response": response,
                    "action": "{}",
                }
            )
            try:
                action = _parse_resp(response)
                logging.debug(f"action: {action}")
                history[-1]["action"] = json.dumps(action, ensure_ascii=False)
                # next step
                if action["action_type"] == "status":
                    break
                action_type = action.pop("action_type")
                action_args = action
                action = {"name": action_type, "arguments": json.dumps(action_args)}
                tic = time.perf_counter()
                screenshot = self.device_client.step(action=action)
                toc = time.perf_counter()
                logging.debug(f"client step execution time: {toc - tic:.2f}s")
            except Exception as e:
                logging.error(e)
                break
            step += 1

        return {
            "task": task,
            "trajectory": history,
        }


def _input_msg(goal, screenshot, history):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": screenshot,
                }
            ],
        },
        {
            "role": "user",
            "content": ACTION_SELECTION_PROMPT_TEMPLATE.format(
                goal=goal, history=history
            ),
        },
    ]


def _parse_resp(response):
    action = None
    try:
        pattern = r"Action: (.*)"
        match = re.search(pattern, response)
        if match:
            action = match.group(1).strip()
            action = json.loads(action)
    except Exception as e:
        logging.warning(f"parsing action field, {response}, {e}")
    return action
