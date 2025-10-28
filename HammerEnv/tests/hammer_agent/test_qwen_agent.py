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

import json
from absl import logging
from dotenv import load_dotenv

import os

from hammer_agent.qwen_agent import Operator
from server.client import HammerEnvClient

logging.set_verbosity("debug")
load_dotenv(os.path.join(os.getcwd(), ".env"))
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = "http://your-api-endpoint/v1"


def test_operator():
    client = HammerEnvClient("http://localhost:7860/")
    device_info = client.request_device()
    logging.info(f"device info: {device_info}")
    try:
        operator = Operator(device_client=client, max_steps=10)
        result = operator.run(task="定个明早8点的闹钟")
        with open("./records/tmp.json", mode="w") as f:
            json.dump(result, f, ensure_ascii=False)
    except Exception as e:
        logging.error(e)
    client.close()


if __name__ == "__main__":
    test_operator()
