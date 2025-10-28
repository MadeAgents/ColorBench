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
from agents import (
    ModelSettings,
    RunConfig,
    set_default_openai_api,
    set_tracing_disabled,
)
from dotenv import load_dotenv

import os

from hammer_agent.agent import Operator
from server.client import HammerEnvClient

logging.set_verbosity("debug")
load_dotenv(os.path.join(os.getcwd(), ".env"))
set_default_openai_api("chat_completions")
set_tracing_disabled(True)


def test_operator():
    client = HammerEnvClient("http://localhost:7860/")
    device_info = client.request_device()
    logging.info(f"device info: {device_info}")
    run_config = RunConfig(
        model="gpt-4o-mini",
        model_settings=ModelSettings(
            **{
                "temperature": 0.8,
                "top_p": 1,
                "max_tokens": 1204,
            }
        ),
    )
    try:
        operator = Operator(device_client=client, max_steps=5)
        result = operator.run(task="定个明早8点的闹钟", run_config=run_config)
        with open("./records/tmp.json", mode="w") as f:
            json.dump(result, f, ensure_ascii=False)
    except Exception as e:
        logging.error(e)
    client.close()


if __name__ == "__main__":
    test_operator()
