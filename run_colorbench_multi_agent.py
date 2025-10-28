import os
import time
import json
import logging
import colorlog
import argparse
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import yaml
from src.agent.thread_safe_agent_factory import ThreadSafeAgentFactory, ThreadSafeGraphDataSet, ThreadSafeTaskExecutor

load_dotenv()

def setup_console_encoding():
    """设置控制台编码为UTF-8"""
    try:
        import sys
        import codecs

        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')

        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
    except Exception as e:
        print(f"Warning: Could not set console encoding: {e}")

def setup_logging(log_file_path):
    """Configure logging system with thread-safe handlers"""
    import io
    import codecs

    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - [Thread-%(thread)d] - %(name)s - %(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'
        }
    )

    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_path, mode="w", encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - [Thread-%(thread)d] - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_yaml(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading YAML file {file_path}: {e}")
        return None

def main():
    setup_console_encoding()
    
    # Load configuration
    parser = argparse.ArgumentParser(
        description="Run specific tasks (160-170) with multithreading support"
    )
    parser.add_argument(
        "--config",
        default='./config/mlas.yaml',
        help="Path to the config YAML file.",
    )
    parser.add_argument(
        "--model",
        default='gui-owl-32b',
        help="Model configuration to use.",
    )  
    parser.add_argument(
        "--mode",
        default='plan-reflect',
        help="Agent mode to use (vanilla, memory, thought, plan-reflect).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=3,
        help="Maximum number of worker threads (default: 3).",
    )
    parser.add_argument(
        "--task_start",
        type=int,
        default=160,
        help="Start task ID (default: 160).",
    )
    parser.add_argument(
        "--task_end",
        type=int,
        default=170,
        help="End task ID (default: 170).",
    )
    parser.add_argument(
        "--no_use_plan",
        action='store_true'
    )
    parser.add_argument(
        "--no_use_reflect",
        action='store_true'
    )
    parser.add_argument(
        "--no_use_memory",
        action='store_true'
    )
    parser.add_argument(
        "--use_glm",
        action='store_true'
    )
    
    
    args = parser.parse_args()
    tmp_time = datetime.datetime.now().strftime("%m%d_%H%M")
    config_name = f'tasks_glm{args.use_glm}_{args.task_start}_{args.task_end}_{args.mode}_{args.model}_noplan{args.no_use_plan}_noreflect{args.no_use_reflect}_nomemory{args.no_use_memory}_multithread_{tmp_time}'

    config = load_yaml(args.config)
    parent_dir = config['path']['image_folder']
    output_dir = config['path']['output_folder']
    os.makedirs(output_dir, exist_ok=True)

    log_file_path = f'./log/{config_name}.log'
    os.makedirs('./log', exist_ok=True)
    setup_logging(log_file_path)
    logger = logging.getLogger(__name__)
    logger.info("Starting Multithreaded Tasks Execution!")

    # renew config
    if args.no_use_memory:
        config['agent'][args.mode]['memory'] = False
    if args.no_use_reflect:
        config['agent'][args.mode]['reflect'] = False
    if args.no_use_plan:
        config['agent'][args.mode]['plan'] = False
    config['agent'][args.mode]['glm'] = args.use_glm
    print(args.no_use_plan, args.no_use_reflect, args.no_use_memory)
    print(config['agent'][args.mode]['plan'] , config['agent'][args.mode]['reflect'], config['agent'][args.mode]['memory'])
    print(type(args.no_use_plan), type(args.no_use_reflect), type(args.no_use_memory))

    agent_factory = ThreadSafeAgentFactory(config['agent'])
    graph_factory = ThreadSafeGraphDataSet(config['graph'])
    task_executor = ThreadSafeTaskExecutor(agent_factory, graph_factory, config)

    task_json = config['tasks']['tasks_file']
    with open(task_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    task_range = [item for item in data if args.task_start <= item.get('task_id', 0) <= args.task_end]
    logger.info(f"Processing {len(task_range)} tasks ({args.task_start}-{args.task_end}) with {args.max_workers} threads")

    results = []
    completed_tasks = 0
    failed_tasks = 0
    total_start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {}
        for task_item in task_range:
            future = executor.submit(
                task_executor.execute_task,
                task_item,
                args.mode,
                args.model,
                output_dir,
                parent_dir,
                config_name
            )
            future_to_task[future] = task_item

        for future in as_completed(future_to_task):
            task_item = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    completed_tasks += 1
                    logger.info(f"[SUCCESS] Task {result['task_id']} completed successfully in {result['time']:.2f}s")
                elif not result['success']:
                    logger.info(f"[FAILED] Task {result['task_id']} completed with failure.")
                else:
                    print(result)
                    failed_tasks += 1
                    logger.error(f"[FAILED] Task {result['task_id']} failed: {result.get('error', 'Unknown error')}")
                
            except Exception as e:
                failed_tasks += 1
                logger.error(f"[FAILED] Task {task_item.get('task_id', 'unknown')} failed with exception: {str(e)}")
                results.append({
                    'task_id': task_item.get('task_id', 'unknown'),
                    'task': task_item.get('query', 'unknown'),
                    'success': False,
                    'error': str(e),
                    'thread_id': 'unknown'
                })

    total_time = time.time() - total_start_time
    logger.info("=" * 80)
    logger.info("MULTITHREADED EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tasks: {len(task_range)}")
    logger.info(f"Completed successfully: {completed_tasks}")
    logger.info(f"Failed: {failed_tasks}")
    logger.info(f"Success rate: {completed_tasks/len(task_range)*100:.1f}%")
    logger.info(f"Total execution time: {total_time:.2f}s")
    logger.info(f"Average time per task: {total_time/len(task_range):.2f}s")
    logger.info(f"Threads used: {args.max_workers}")
    logger.info("=" * 80)

    summary_file = os.path.join(output_dir, config_name, 'execution_summary.json')
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    summary_data = {
        'config_name': config_name,
        'mode': args.mode,
        'model': args.model,
        'max_workers': args.max_workers,
        'task_range': f"{args.task_start}-{args.task_end}",
        'total_tasks': len(task_range),
        'completed_tasks': completed_tasks,
        'failed_tasks': failed_tasks,
        'success_rate': completed_tasks/len(task_range)*100,
        'total_execution_time': total_time,
        'average_time_per_task': total_time/len(task_range),
        'results': results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Execution summary saved to: {summary_file}")
    logger.info("Multithreaded execution completed!")

if __name__ == "__main__":
    main()
