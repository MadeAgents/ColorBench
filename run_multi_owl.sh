cd path/to/ColorBench


python3 run_colorbench_multi_agent.py --model gui-owl-32b --mode plan-reflect --max_workers 10 --task_start 0 --task_end 176
python3 run_colorbench_multi_agent.py --model gui-owl-32b --mode plan-reflect --max_workers 10 --task_start 0 --task_end 176 --no_use_memory
python3 run_colorbench_multi_agent.py --model gui-owl-32b --mode plan-reflect --max_workers 10 --task_start 0 --task_end 176 --no_use_reflect --no_use_memory
python3 run_colorbench_multi_agent.py --model gui-owl-32b --mode plan-reflect --max_workers 10 --task_start 0 --task_end 176 --no_use_plan --no_use_memory
python3 run_colorbench_multi_agent.py --model gui-owl-32b --mode plan-reflect --max_workers 10 --task_start 0 --task_end 176 --no_use_plan --no_use_reflect