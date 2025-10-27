import json
import os
from collections import defaultdict


checkpoint_path = 'path/to/checkpoints'
task_json = './data/tasks.json'


with open(task_json, 'r', encoding='utf-8') as f:
    all_tasks = json.load(f)

tasks_list = [item['query'] for item in all_tasks]
tasks_optimal_steps = {item['query']: item['optimal_steps'] for item in all_tasks}
tasks_milestone = {item['query']: item['milestone'] for item in all_tasks}
tasks_app_nums = {item['query']: item['app_num'] for item in all_tasks}



def get_successful_tasks_by_rule(checkpoint_path, tasks_milestone):
    """获取已经成功完成的任务列表，按照规则判断任务是否完成"""
    completed_tasks = set()
    ability_dicts = defaultdict(int)  # 记录每一个能力在任务中的总数
    ability_success_dicts = defaultdict(int)
    task_total_parts = list()
    task_success_part = list()

    error_task_lists = []
    for task, milestones in tasks_milestone.items():
        task_checkpoint_path = os.path.join(checkpoint_path, task.replace('/','_'))
        task_trajectory_json = os.path.join(task_checkpoint_path, 'trajectory.json')
        try:
            with open(task_trajectory_json, 'r', encoding = 'utf-8') as ft:
                trajectory = json.load(ft)
        except FileNotFoundError:
            print(f"Connot open file {task_trajectory_json}")
            error_task_lists.append(task)
            continue

        screenshot_lists = set([item['screenshot'] for item in trajectory])

        successful_part_count = 0

        fail_ability = list()
        for milestone in milestones:
            # 顺序遍历这个任务的每一个milestone
            ability = milestone['ability']  
            fail_ability.append(ability)
            # 更新正确率
            try:
                pagenodes = milestone['page_node']
            except:
                print(f"Task {task} need answer {milestone['answer']} with ability {ability}")
                continue
            for node in pagenodes:
                if node in screenshot_lists:
                    ability_success_dicts[ability] += 1  # 表示成功
                    successful_part_count += 1
                    ability_dicts[ability] += 1  # 能力总数
                    fail_ability.pop() # 成功,把他删除掉
                    break
        if fail_ability:              
            ability_dicts[fail_ability[0]] += 1  # 能力总数

        task_total_parts.append(len(milestones))
        task_success_part.append(successful_part_count)
        if len(milestones) == successful_part_count:
            completed_tasks.add(task)
    
    # 计算每个任务的成功率
    task_success_rate = [
        success / total if total > 0 else 0
        for success, total in zip(task_success_part, task_total_parts)
    ]

    # 计算每个能力的成功率
    ability_success_rate = {
        ability: int(ability_success_dicts[ability]) / int(total) for ability, total in ability_dicts.items()
    }

    print(f"处理任务时发生错误的任务: {error_task_lists}")
    print(f"处理任务时发生错误的任务数量: {len(error_task_lists)}")
    return completed_tasks, task_success_rate, ability_success_rate


def get_successful_certain_app_tasks_by_rule(checkpoint_path, tasks_milestone, app='single'):
    """获取已经成功完成的任务列表，按照规则判断任务是否完成"""
    completed_tasks = set()
    ability_dicts = defaultdict(int)  # 记录每一个能力在任务中的总数
    ability_success_dicts = defaultdict(int)
    task_total_parts = list()
    task_success_part = list()

    error_task_lists = []
    for task, milestones in tasks_milestone.items():
        if (app == 'single' and tasks_app_nums[task] == 1) or (app == 'multi' and tasks_app_nums[task] > 1):
            
            task_checkpoint_path = os.path.join(checkpoint_path, task.replace('/','_'))
            task_trajectory_json = os.path.join(task_checkpoint_path, 'trajectory.json')
            try:
                with open(task_trajectory_json, 'r', encoding = 'utf-8') as ft:
                    trajectory = json.load(ft)
            except FileNotFoundError:
                print(f"Connot open file {task_trajectory_json}")
                error_task_lists.append(task)
                continue

            screenshot_lists = set([item['screenshot'] for item in trajectory])

            successful_part_count = 0

            fail_ability = list()
            for milestone in milestones:
                # 顺序遍历这个任务的每一个milestone
                ability = milestone['ability']  
                fail_ability.append(ability)
                # 更新正确率
                try:
                    pagenodes = milestone['page_node']
                except:
                    print(f"Task {task} need answer {milestone['answer']} with ability {ability}")
                    continue
                for node in pagenodes:
                    if node in screenshot_lists:
                        ability_success_dicts[ability] += 1  # 表示成功了
                        successful_part_count += 1
                        ability_dicts[ability] += 1  # 能力总数
                        fail_ability.pop() # 成功了，把他删除掉
                        break
            if fail_ability:              
                ability_dicts[fail_ability[0]] += 1  # 能力总数

            task_total_parts.append(len(milestones))
            task_success_part.append(successful_part_count)
            if len(milestones) == successful_part_count:
                completed_tasks.add(task)
    
    # 计算每个任务的成功率
    task_success_rate = [
        success / total if total > 0 else 0
        for success, total in zip(task_success_part, task_total_parts)
    ]


    print(f"处理任务时发生错误的任务: {error_task_lists}")
    print(f"处理任务时发生错误的任务数量: {len(error_task_lists)}")
    print(f'总计的{app}任务数量: {len(task_total_parts)}')
    return completed_tasks, task_success_rate  #, ability_success_rate


# never use
def get_step_efficiency(successful_tasks_list, tasks_optimal_steps):
    step_efficiency = dict()
    for task in successful_tasks_list:
        task_checkpoint_path = os.path.join(checkpoint_path, task.replace('/','_'))
        task_trajectory_json = os.path.join(task_checkpoint_path, 'trajectory.json')
        try:
            with open(task_trajectory_json, 'r', encoding = 'utf-8') as ft:
                trajectory = json.load(ft)
            practical_steps = len(trajectory)
            optimal_steps = tasks_optimal_steps[task]
            step_efficiency[task] = (practical_steps, optimal_steps)
            
        except Exception as e:
            print(f"处理任务'{task}'时发生错误: {e}")

    step_efficiency_rate = {
        task: int(item[1]) / int(item[0]) for task, item in step_efficiency.items()
    }
    return step_efficiency, step_efficiency_rate


def get_wrong_tasks(successful_tasks_list, tasks_list):
    return [task for task in tasks_list if task not in successful_tasks_list]


if __name__ == "__main__":
    completed_tasks, task_success_rate, ability_success_rate = get_successful_tasks_by_rule(checkpoint_path, tasks_milestone)
    print(f"\n\n成功完成的任务数: {len(completed_tasks)} / {len(tasks_list)}，成功率: {len(completed_tasks)/len(tasks_list):.2%}")
    print(f"任务平均成功率: {sum(task_success_rate)/len(task_success_rate):.2%}")
    single_completed_tasks, single_task_success_rate = get_successful_certain_app_tasks_by_rule(checkpoint_path, tasks_milestone, app='single')
    print(f"\n\n单应用成功完成的任务数: {len(single_completed_tasks)} / {len(single_task_success_rate)}，成功率: {len(single_completed_tasks)/len(single_task_success_rate):.2%}")
    print(f"单应用任务平均成功率: {sum(single_task_success_rate)/len(single_task_success_rate):.2%}")
    multi_completed_tasks, multi_task_success_rate = get_successful_certain_app_tasks_by_rule(checkpoint_path, tasks_milestone, app='multi')
    print(f"\n\n多应用成功完成的任务数: {len(multi_completed_tasks)} / {len(multi_task_success_rate)}，成功率: {len(multi_completed_tasks)/len(multi_task_success_rate):.2%}")
    print(f"多应用任务平均成功率: {sum(multi_task_success_rate)/len(multi_task_success_rate):.2%}")
    print("\n\n各能力成功率:")
    ability_mapping = {
        "分享": "share",
        "下载": "save",
        "保存": "save",
        "复制": "copy",
        "购买": "pay",
        "搜索": "search",
        "记忆": "memory",
        "发布": "send",
        "发送": "send",
        "地址": "location",
        "关注": "follow",
        "筛选": "filter",
        "喜欢": "like",
        "点赞": "like",
        "收藏": "like",
        "导航": "navigation",
        "定位": "find",
        "查看": "find",
        "设置": "set"
    }
    ability_lists = list(ability_mapping.values())
    for ability, rate in ability_success_rate.items():
        if ability in ability_lists or ability == 'others':
            print(f"  {ability}: {rate:.2%}")   

    # wrong_task = get_wrong_tasks(completed_tasks, tasks_list)
    # print(f"没有完成的任务数: {len(wrong_task)}")
    # for task in wrong_task:
    #     print(f"  {task}")

    # print(set(ability_mapping.values()))
    # print(completed_tasks)