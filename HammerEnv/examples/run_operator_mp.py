from pathlib import Path
import random
import uuid
from absl import logging
from agents import (
    ModelSettings,
    RunConfig,
    set_default_openai_api,
    set_tracing_disabled,
)
from dotenv import load_dotenv

import json
import multiprocessing
import os

from hammer_agent.agent import Operator
from server.client import HammerEnvClient

logging.set_verbosity("debug")

WORK_DIR = Path(__file__).parent
logging.debug(f"work dir: {WORK_DIR}")
load_dotenv((WORK_DIR / ".env").as_posix())

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

OUTPUT_DIR = "./records"


def worker(queue):
    client = HammerEnvClient("http://localhost:7860/")
    device_info = client.request_device()
    logging.info(f"device info: {device_info}")
    run_config = RunConfig(
        model="gpt-4o",
        model_settings=ModelSettings(
            **{
                "temperature": 0.8,
                "top_p": 1,
                "max_tokens": 1204,
            }
        ),
    )
    operator = Operator(device_client=client, max_steps=5)
    while True:
        task = queue.get()
        if task is None:
            break
        task_id = uuid.uuid4().hex
        logging.info(f"process: {multiprocessing.current_process().name}\ntask: {task}")
        try:
            result = operator.run(task=task, run_config=run_config)
            with open(f"./records/{task}_{task_id}.json", mode="w") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logging.error(e)
        queue.task_done()
    client.close()


def main():
    task_queue = multiprocessing.JoinableQueue()
    tasks = ["打开相机", "打开 Kimi", "现在几点了", "定个明早8点的闹钟"]
    logging.info(f"tasks: {tasks}")
    for t in tasks:
        task_queue.put(t)
    processes = []

    client = HammerEnvClient("http://localhost:7860/")
    num_workers = len(client.avaliable_devices)
    logging.info(f"# of workers: {num_workers}")

    for _ in range(num_workers):
        p = multiprocessing.Process(target=worker, args=(task_queue,))
        p.start()
        processes.append(p)

    for _ in range(len(processes)):
        task_queue.put(None)

    for p in processes:
        p.join()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
