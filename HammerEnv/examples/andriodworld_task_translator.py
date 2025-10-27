import re
import csv
import json
from googletrans import Translator

def extract_tasks(file_path):
    """从文本文件中提取任务信息"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 使用正则表达式匹配每个任务块
    task_pattern = r'任务名称: (.*?)\n实例数量: (\d+)\n\n--- 实例 (\d+) ---\n目标: (.*?)\n参数: (.*?)\n复杂度: (.*?)\n\n=================================================='
    matches = re.findall(task_pattern, content, re.DOTALL)
    
    tasks = []
    for match in matches:
        task_name, instance_count, instance_num, target, params, complexity = match
        
        # 处理参数
        try:
            params_dict = eval(params)  # 将字符串转换为字典
        except:
            params_dict = {}
            
        tasks.append({
            '任务名称': task_name.strip(),
            '实例数量': instance_count.strip(),
            '实例编号': instance_num.strip(),
            '目标(英文)': target.strip(),
            '参数': params_dict,
            '复杂度': complexity.strip()
        })
    
    return tasks

def translate_text(text, translator):
    """将英文文本翻译为中文"""
    try:
        translation = translator.translate(text, dest='zh-CN')
        return translation.text
    except Exception as e:
        print(f"翻译失败: {text}, 错误: {e}")
        return text  # 翻译失败时返回原文

def save_to_csv(tasks, output_file):
    """将任务信息保存为CSV文件"""
    # 初始化翻译器
    translator = Translator()
    
    # 定义CSV列名
    fieldnames = ['任务名称', '实例数量', '实例编号', '目标(英文)', '目标(中文)', '参数', '复杂度']
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for task in tasks:
            # 翻译目标文本
            task['目标(中文)'] = translate_text(task['目标(英文)'], translator)
            # 将参数字典转换为字符串
            task['参数'] = json.dumps(task['参数'], ensure_ascii=False)
            writer.writerow(task)

if __name__ == "__main__":
    # 输入文件路径
    input_file = 'all_task_seed30.txt'
    # 输出CSV文件路径
    output_file = 'tasks_translated.csv'
    
    # 提取任务信息
    print(f"正在从 {input_file} 提取任务信息...")
    tasks = extract_tasks(input_file)
    print(f"成功提取 {len(tasks)} 个任务")
    
    # 保存为CSV文件（包含翻译）
    print(f"正在翻译并保存到 {output_file}...")
    save_to_csv(tasks, output_file)
    print("操作完成！")
