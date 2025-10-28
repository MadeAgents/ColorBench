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

from absl import logging
from datetime import datetime
from dotenv import load_dotenv

import argparse
import json
import os
import uuid

from hammer_agent.qwen_agent import Operator
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
device_manager = DeviceManager()

print("Python模块搜索路径：")
for path in sys.path:
    print(path)
logging.set_verbosity("debug")
load_dotenv(os.path.join(os.getcwd(), ".env"))
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = "http://your-api-endpoint/v1"

APP_NAME = "general"
MODEL_NAME = "Qwen2.5-VL-72B-Instruct"


def run_operator(task, src, model_name=MODEL_NAME, max_steps=20):
    client = HammerEnvClient(src=src)
    device_info = client.request_device()
    logging.info(f"device info: {device_info}")
    try:
        timestamp = datetime.now().strftime("%Y%m%d")
        operator = Operator(device_client=client, model_name=model_name, max_steps=max_steps)
        result = operator.run(task=task)
        filepath = f"./records/{APP_NAME}_{timestamp}_{uuid.uuid4().hex}.json"
        with open(filepath, mode="w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        logging.info(f"filepath: {filepath}")
    except Exception as e:
        logging.error(e)
    client.close()


def parse_args():
    parser = argparse.ArgumentParser(description="GUI 智能体")

    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--server-name", type=str, default="http://localhost:7880/")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--max-steps", type=int, default=20)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    task = args.task
    src = args.server_name
    model_name = args.model_name
    max_steps = args.max_steps
    run_operator(task=task, src=src, model_name=model_name, max_steps=max_steps)
