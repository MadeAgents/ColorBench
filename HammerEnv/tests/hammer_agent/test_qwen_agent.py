import json
from absl import logging
from dotenv import load_dotenv

import os

from hammer_agent.qwen_agent import Operator
from server.client import HammerEnvClient

logging.set_verbosity("debug")
load_dotenv(os.path.join(os.getcwd(), ".env"))
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = "https://ctd-browser-tool.wanyol.com/v1"


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
