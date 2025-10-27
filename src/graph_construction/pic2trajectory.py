# -*- coding: utf-8 -*-
"""
功能：全流程支持PNG/JPG/JPEG，从图片处理到邻接矩阵生成
1. 处理指定目录下所有子文件夹中的图片（含PNG），生成轨迹文件trajectory_v0.txt（优化顺序识别）
2. 基于轨迹文件生成包含JSON格式跳转条件的邻接矩阵CSV（不忽视PNG）
3. CSV文件名为query内容（移除特殊字符后）
@author: 80398388
"""

import os
import cv2
import re
import base64
import json
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class AtomicCounter:
    """线程安全的原子计数器"""
    def __init__(self, initial=0):
        self.value = initial
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self.value += 1
            return self.value


thread_local = threading.local()
def get_openai_client():
    """获取线程本地的OpenAI客户端"""
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(
            base_url="your_base_url",
            api_key="empty",
        )
    return thread_local.client


def extract_order(text):
    """从模型输出中提取order字段"""
    pattern = r'###order[：:](.+)'
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def sort_files_by_step(files):
    """根据文件名中的step编号排序（支持PNG/JPG，确保顺序正确）"""
    def extract_step(file):
        # 匹配所有格式图片的step编号（Screenshot_step_xxxx_raw.xxx）
        match = re.search(r'screenshot_step_(\d+)_raw\.(jpg|jpeg|png)', file, re.IGNORECASE)
        return int(match.group(1)) if match else 0  # 无step编号默认最小
    sorted_files = sorted(files, key=extract_step)
    # 打印排序结果，标注格式，便于确认PNG未被忽视
    sorted_info = [f"{os.path.basename(f)}（{os.path.splitext(f)[1][1:].upper()}）" for f in sorted_files]
    print(f"✅ 图片排序结果（含格式）：{sorted_info}")
    return sorted_files


def generate_pairs(files):
    """生成相邻文件对（支持PNG/JPG，严格按前→后顺序）"""
    pairs = [(files[i], files[i+1]) for i in range(len(files)-1)]
    # 打印图片对信息，标注格式，确认PNG参与配对
    pair_info = [
        f"({os.path.basename(p[0])}({os.path.splitext(p[0])[1][1:].upper()}) → {os.path.basename(p[1])}({os.path.splitext(p[1])[1][1:].upper()}))" 
        for p in pairs
    ]
    print(f"✅ 生成图片对（含格式）：{pair_info}")
    return pairs


def extract_step_number(file_name):
    """从文件名提取step编号（支持PNG/JPG）"""
    match = re.search(r'screenshot_step_(\d+)_raw\.(jpg|jpeg|png)', file_name, re.IGNORECASE)
    return int(match.group(1)) if match else "未知"


def save_list_to_txt(data_list, file_path):
    """将列表保存为文本文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for item in data_list:
                file.write(f"{str(item)}\n")
        print(f"✅ 已保存文件到：{file_path}")
    except Exception as e:
        print(f"❌ 保存文件失败：{e}")


def clean_filename(filename):
    """清理文件名，移除特殊字符"""
    invalid_chars = '/\\:*?"<>|'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    return filename[:50].strip()  # 限制长度，避免系统不支持过长文件名


def extract_frames(video_path, output_dir, mode='average'):
    """从视频中提取帧（支持生成PNG/JPG，备用功能）"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"❌ 视频没有可读取的帧: {video_path}")
        cap.release()
        return 0
    
    target_frames = max(1, int(total_frames * 0.01))
    save_count = 0

    if mode == 'average':
        frame_indices = [int(i*(total_frames-1)/(target_frames-1)) for i in range(target_frames)]
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # 支持生成PNG（可根据需求切换为.jpg）
                frame_filename = f"screenshot_step_{frame_idx:06d}_raw.png"
                cv2.imwrite(os.path.join(output_dir, frame_filename), frame)
                save_count += 1
    
    elif mode == 'skip':
        frame_skip = max(1, total_frames // target_frames)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0 and save_count < target_frames:
                frame_filename = f"screenshot_step_{frame_count:06d}_raw.png"
                cv2.imwrite(os.path.join(output_dir, frame_filename), frame)
                save_count += 1
            frame_count += 1
    
    cap.release()
    print(f"📹 视频 {video_path} 处理完成，提取 {save_count}/{target_frames} 帧（PNG格式）")
    return save_count


def process_video(video_path, output_root_dir, mode='average'):
    """处理单个视频文件（支持生成PNG帧）"""
    video_name = Path(video_path).stem
    output_dir = os.path.join(output_root_dir, video_name)
    frame_count = extract_frames(video_path, output_dir, mode)
    return video_name if frame_count >= 2 else None


def process_videos(input_dir, output_root_dir, mode='average', max_workers=None):
    """多线程处理目录中的视频文件（支持生成PNG帧）"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
    video_files = [entry.path for entry in os.scandir(input_dir) 
                  if entry.is_file() and entry.name.lower().endswith(tuple(video_extensions))]
    
    os.makedirs(output_root_dir, exist_ok=True)
    print(f"🔍 找到 {len(video_files)} 个视频文件")
    
    valid_video_folders = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {executor.submit(process_video, vp, output_root_dir, mode): vp for vp in video_files}
        for future in as_completed(future_to_video):
            video_path = future_to_video[future]
            try:
                result = future.result()
                if result:
                    valid_video_folders.append(result)
            except Exception as e:
                print(f"❌ 处理视频 {video_path} 出错: {e}")
    
    return valid_video_folders


def get_image_mime_type(image_path):
    """根据文件扩展名获取图片MIME类型（支持PNG/JPG/JPEG）"""
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.png':
        return "image/png"
    elif ext in ('.jpg', '.jpeg'):
        return "image/jpeg"
    else:
        raise ValueError(f"❌ 不支持的图片格式：{ext}（仅支持PNG/JPG/JPEG）")


def analyze_images(image_files, input_dir, counter):
    """分析图片对（完整支持PNG/JPG，强化顺序），生成轨迹数据，返回query内容"""
    print(f'\n📊 开始分析图片组 ({counter.increment()}/1)')
    
    if len(image_files) < 2:
        print(f"⚠️ 需要至少2张图片（PNG/JPG），当前找到 {len(image_files)} 张")
        return "direct_images", [], ""
        
    # 读取query.txt（用户任务）
    query_file_path = os.path.join(input_dir, "query.txt")
    if not (os.path.exists(query_file_path) and os.path.isfile(query_file_path)):
        print(f"❌ 未找到 query.txt 文件: {query_file_path}")
        return "direct_images", [], ""
    
    try:
        with open(query_file_path, 'r', encoding='utf-8') as file:
            query = file.read().strip()
            print(f"📝 读取到用户任务：{query}")
    except Exception as e:
        print(f"❌ 读取 query.txt 出错: {e}")
        return "direct_images", [], ""
    
    # 确保图片对顺序正确（支持PNG/JPG，按step编号排序）
    sorted_files = sort_files_by_step(image_files)
    pairs = generate_pairs(sorted_files)
    client = get_openai_client()
    
    # 1. 识别用户任务涉及的APP（用于提示增强）
    intent_prompt = f"用户query：{query}，提取最可能涉及的app名（如‘淘宝’‘小红书’），仅输出app名，不额外说明"
    intent_msg = [{"role": "user", "content": [{"type": "text", "text": intent_prompt}]}]
    
    try:
        intent_response = client.chat.completions.create(
            model="gui-owl-32b",
            messages=intent_msg,
            temperature=0,
            max_tokens=1024,
        )
        app = intent_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️  识别APP出错: {e}")
        app = ""
            
    # APP提示增强数据库
    enhanced_db = {
        "小红书": "1.红书的关注、粉丝、获赞与收藏等页面需要先进入‘我’；",
        "淘宝": "1.产品详情页右上角小推车图标可进入购物车",
        "中国移动":"1.流量、生活缴费等功能可通过充话费按钮进入；",
        "中国电信":"1.增值业务可从已定业务进入；",
        "高德地图":"1.打车页面下拉显示更多车型；",
        "懂车帝":"1.更多服务通过“我的”页面进入；",
        "12306":"1.酒店住宿可通过搜索进入；",
        "铁路12306":"1.酒店住宿可通过搜索进入；",
        "bilibili":"暗色模型通过右上角月亮图切换"
    }
    
    enhanced_info = enhanced_db.get(app, "")
    enhanced_prompt = f"辅助信息（{app}）: {enhanced_info}" if enhanced_info else "无额外辅助信息"
    
    # 2. 处理每对图片（核心：支持PNG/JPG，强化顺序提示）
    folder_results = []
    summary_list = []  # 历史动作记录
    result_str_list = []
    
    for k, pair in enumerate(pairs, 1):
        img1_path, img2_path = pair  # img1：前序（操作前），img2：后序（操作后）
        img1_name = os.path.basename(img1_path)
        img2_name = os.path.basename(img2_path)
        # 明确获取图片格式，确保PNG/JPG都被正确处理
        img1_format = os.path.splitext(img1_path)[1][1:].upper()
        img2_format = os.path.splitext(img2_path)[1][1:].upper()
        step1 = extract_step_number(img1_name)
        step2 = extract_step_number(img2_name)
        
        # 构建第一阶段提示（强调顺序+明确图片格式）
        reward_prompt = f"""
            ### 核心约束1：顺序不可颠倒 ###
            - 截图1（{img1_name}，{img1_format}格式）是【操作前的初始状态】
            - 截图2（{img2_name}，{img2_format}格式）是【操作后的结果状态】
            - 你的任务是分析：在截图1上执行什么操作，能得到截图2的状态
            - 绝对不允许分析“从截图2到截图1”的反向操作！

            ### 核心约束2：操作范围 ###
            - 所有操作必须在截图1上执行，不能涉及截图2的元素
            - 若截图1有弹窗（覆盖>1/3屏幕或居中），必须优先处理弹窗（点×/取消/等待）

            ### 背景信息 ###
            用户最终任务：{query}
            当前步骤：第{k}步（分析截图1→截图2的动作）
            截图分辨率：1080x2374像素（x：从左到右，y：从上到下）
            历史已执行操作：{summary_list if summary_list else "无"}
            辅助信息：{enhanced_prompt}

            ### 可选操作类型 ###
            1. "AWAKE[app_name]": 打开指定APP（仅当截图1是桌面/非目标APP时可用）
            2. "CLICK[x,y]": 点击截图1上的(x,y)坐标（需精准到具体元素）
            3. "TYPE[text]": 在截图1中最近点击的输入框内输入文本
            4. "WAIT": 在截图1停留（如广告倒计时、页面加载）
            5. "system_button[Home]": 点击手机Home键返回桌面

            ### 输出格式（严格遵守，少任何字段都无效） ###
            "ACTION[parameters]###reasoning: 用中文说明：在截图1的哪个元素执行什么操作，为何能得到截图2的状态###order: 一句话描述“在截图1上做什么”（无参数，不提截图名）###confidence: 0.0-1.0的置信度"
        """
        
        # 准备带图片的模型请求（支持PNG/JPG，按顺序添加，明确MIME类型）
        msg = [{"role": "user", "content": [{"type": "text", "text": reward_prompt}]}]
        # 1. 添加截图1（含格式标识，正确MIME类型）
        msg[0]['content'].append({"type": "text", "text": f"【操作前：截图1（{img1_format}）】{img1_name}"})
        try:
            img1_mime = get_image_mime_type(img1_path)
        except ValueError as e:
            print(f"❌ 第{k}步截图1格式错误：{e}，跳过该对图片")
            continue
        with open(img1_path, "rb") as f:
            img1_base64 = base64.b64encode(f.read()).decode("utf-8")
        msg[0]['content'].append({
            "type": "image_url", 
            "image_url": {"url": f"data:{img1_mime};base64,{img1_base64}", "order": 1}  # order=1强化顺序
        })
        # 2. 添加截图2（含格式标识，正确MIME类型）
        msg[0]['content'].append({"type": "text", "text": f"【操作后：截图2（{img2_format}）】{img2_name}"})
        try:
            img2_mime = get_image_mime_type(img2_path)
        except ValueError as e:
            print(f"❌ 第{k}步截图2格式错误：{e}，跳过该对图片")
            continue
        with open(img2_path, "rb") as f:
            img2_base64 = base64.b64encode(f.read()).decode("utf-8")
        msg[0]['content'].append({
            "type": "image_url", 
            "image_url": {"url": f"data:{img2_mime};base64,{img2_base64}", "order": 2}  # order=2强化顺序
        })
        
        # 调用模型分析动作
        try:
            response = client.chat.completions.create(
                model="gui-owl-32b",
                messages=msg,
                temperature=0,
                max_tokens=1500,
            )
            response_content = response.choices[0].message.content.strip()
            print(f"\n第{k}步（{img1_format}→{img2_format}）模型输出：{response_content[:100]}...")  # 打印前100字符
        except Exception as e:
            print(f"❌ 第{k}步分析图片对（{img1_name}→{img2_name}）出错: {str(e)}")
            folder_results.append({
                "step1": step1, "step2": step2, "image1": img1_name, "image2": img2_name, "error": str(e)
            })
            continue
        
        # 提取order（用于第二阶段精准动作生成）
        order = extract_order(response_content)
        if not order:
            print(f"⚠️  第{k}步未提取到有效order，跳过该步骤")
            continue

        # 第二阶段提示（仅传入截图1，支持PNG/JPG）
        reward_prompt2 = f"""
            ### 核心要求 ###
            - 当前仅需分析【截图1：{img1_name}（{img1_format}）】（操作前状态）
            - 用户需要执行的操作：{order}
            - 输出必须是能让截图1转变为截图2的动作，不能反向

            ### 动作规范 ###
            选择以下1种动作，参数需精准：
            - "AWAKE[app_name]": 打开APP（如"AWAKE[小红书]"）
            - "CLICK[x,y]": 点击坐标（如"CLICK[1009,185]"，x/y为整数）
            - "TYPE[text]": 输入文本（如"TYPE[ai agent]"）
            - "WAIT": 等待（直接输出"WAIT"）
            - "system_button[Home]": 返回桌面（直接输出"system_button[Home]"）

            ### 输出格式（严格遵守，不要任何额外文字） ###
            "ACTION[PARAMETERS]###reasoning: 简洁说明为何选这个动作（不超过50字）###confidence: 0.0-1.0"
        """
        
        # 准备第二阶段请求（仅传入截图1，正确MIME类型）
        msg2 = [{"role": "user", "content": [{"type": "text", "text": reward_prompt2}]}]
        msg2[0]['content'].append({
            "type": "image_url", 
            "image_url": {"url": f"data:{img1_mime};base64,{img1_base64}"}
        })
        
        try:
            response2 = client.chat.completions.create(
                model="gui-owl-32b",
                messages=msg2,
                temperature=0,
                max_tokens=500,
            )
            action = response2.choices[0].message.content.strip()
            print(f"第{k}步（{img1_format}）最终动作：{action}")
        except Exception as e:
            print(f"❌ 第{k}步获取最终动作出错: {str(e)}")
            folder_results.append({
                "step1": step1, "step2": step2, "image1": img1_name, "image2": img2_name, "error": str(e)
            })
            continue
        
        # 记录结果（包含图片格式信息）
        summary_list.append(action)
        folder_results.append({
            "step1": step1, "step2": step2, 
            "image1": img1_name, "image2": img2_name, 
            "image1_format": img1_format, "image2_format": img2_format,  # 记录格式
            "analysis": action
        })
        # 轨迹文件内容（包含完整图片路径，支持PNG）
        result_str = f"query:{query} Step{k}: {action} images:{img1_path}"
        result_str_list.append(result_str)
        
        time.sleep(1)
    
    # 3. 添加任务完成动作
    if sorted_files:
        last_img_path = sorted_files[-1]
        last_img_name = os.path.basename(last_img_path)
        last_img_format = os.path.splitext(last_img_path)[1][1:].upper()
        last_step = extract_step_number(last_img_name)
        k += 1
        
        complete_action = f"Complete###reasoning: 已完成用户任务「{query}」的所有操作###confidence: 1.0"
        complete_result = {
            "step1": last_step, "step2": last_step,
            "image1": last_img_name, "image2": last_img_name,
            "image1_format": last_img_format, "image2_format": last_img_format,
            "analysis": complete_action
        }
        folder_results.append(complete_result)
        
        complete_str = f"query:{query} Step{k}: {complete_action} images:{last_img_path}"
        print(f"\n第{k}步（完成，{last_img_format}）：{complete_str}")
        result_str_list.append(complete_str)
    
    # 4. 保存轨迹文件（含PNG图片路径）
    trajectory_path = os.path.join(input_dir, "trajectory_v0.txt")
    save_list_to_txt(result_str_list, trajectory_path)
    
    return "direct_images", folder_results, query


def analyze_image_pairs(image_files, input_dir, max_workers=None):
    """分析图片对（支持PNG/JPG），识别轨迹，返回query内容"""
    if not image_files:
        print("❌ 没有找到图片文件（PNG/JPG/JPEG）")
        return {}, ""
    
    # 统计各格式图片数量，确认PNG未被忽视
    format_count = {}
    for img_path in image_files:
        ext = os.path.splitext(img_path)[1].lower()
        format_count[ext] = format_count.get(ext, 0) + 1
    format_info = [f"{k[1:].upper()}: {v}张" for k, v in format_count.items()]
    print(f"\n🔍 开始分析图片（格式分布：{', '.join(format_info)}），共 {len(image_files)} 张")
    
    results = {}
    counter = AtomicCounter()
    query_content = ""
    # 单线程处理（确保顺序正确）
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(analyze_images, image_files, input_dir, counter)
        try:
            folder_name, folder_results, query_content = future.result()
            results[folder_name] = folder_results
        except Exception as e:
            print(f"❌ 图片分析总出错: {e}")
    
    return results, query_content


def generate_adjacency_matrix(trajectory_path, output_dir, query_content):
    """基于轨迹文件生成邻接矩阵CSV（完整支持PNG/JPG，不忽视PNG）"""
    print(f"\n📈 开始生成邻接矩阵（支持PNG/JPG图片名）")
    
    # 1. 处理CSV文件名
    if not query_content:
        csv_filename = "adjacency_matrix.csv"
        print(f"⚠️ query内容为空，使用默认文件名：{csv_filename}")
    else:
        csv_filename = f"{clean_filename(query_content)}.csv"
    output_csv = os.path.join(output_dir, csv_filename)
    print(f"🎯 邻接矩阵保存路径：{output_csv}")
    
    # 2. 读取轨迹文件
    try:
        with open(trajectory_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"✅ 成功读取轨迹文件，共 {len(lines)} 行内容")
    except Exception as e:
        print(f"❌ 读取轨迹文件出错: {e}")
        return output_csv
    
    # 3. 提取图片名（含PNG/JPG，带前缀）和动作（格式化为JSON）
    image_names = []  # 原始图片名（含扩展名，如Screenshot_xxxx.png）
    actions = []      # 格式化后的动作
    
    # 正则表达式：匹配PNG/JPG/JPEG图片名（不忽视PNG）
    img_pattern = re.compile(r'Screenshot_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_[a-f0-9]+\.(jpg|jpeg|png)', re.IGNORECASE)
    awake_pattern = re.compile(r'AWAKE\[([^\]]+)\]')
    click_pattern = re.compile(r'CLICK\[(\d+),(\d+)\]')
    type_pattern = re.compile(r'TYPE\[([^\]]+)\]')
    
    for line in lines:
        # 跳过Complete行，避免重复统计
        if "Complete" in line:
            continue
        
        # 提取图片名（支持PNG/JPG，保留完整扩展名）
        img_match = img_pattern.search(line)
        if img_match:
            img_full_name = img_match.group()  # 含扩展名，如Screenshot_xxxx.png
            image_names.append(img_full_name)
            # 打印提取的图片名，确认PNG被包含
            print(f"🔍 提取图片名：{img_full_name}（{os.path.splitext(img_full_name)[1][1:].upper()}）")
        
        # 提取动作并格式化
        action = None
        awake_match = awake_pattern.search(line)
        if awake_match:
            action = [{"action_type": "open", "details": awake_match.group(1)}]
        click_match = click_pattern.search(line)
        if click_match:
            action = [{"action_type": "click", "x": int(click_match.group(1)), "y": int(click_match.group(2))}]
        type_match = type_pattern.search(line)
        if type_match:
            action = [{"action_type": "input_text", "text": type_match.group(1)}]
        elif "WAIT###" in line:
            action = [{"action_type": "wait", "reason": "页面加载/广告倒计时"}]
        elif "system_button[Home]" in line:
            action = [{"action_type": "system_button", "button_type": "Home"}]
        
        if action:
            actions.append(json.dumps(action, ensure_ascii=False))
    
    # 4. 校验数据（确保PNG图片被正确提取）
    if not image_names:
        print("❌ 未从轨迹文件中提取到图片名（PNG/JPG）")
        return output_csv
    # 统计邻接矩阵中的图片格式分布
    matrix_format_count = {}
    for img_name in image_names:
        ext = os.path.splitext(img_name)[1].lower()
        matrix_format_count[ext] = matrix_format_count.get(ext, 0) + 1
    matrix_format_info = [f"{k[1:].upper()}: {v}张" for k, v in matrix_format_count.items()]
    print(f"✅ 邻接矩阵图片统计（格式分布：{', '.join(matrix_format_info)}），共 {len(image_names)} 张")
    
    # 动作数应为图片数-1
    if len(actions) != len(image_names) - 1:
        print(f"⚠️  动作数与图片数不匹配：图片{len(image_names)}张，动作{len(actions)}个（正常应为图片数-1）")
        # 补全空动作，避免矩阵构建失败
        while len(actions) < len(image_names) - 1:
            actions.append(json.dumps([{"action_type": "unknown", "reason": "未识别到动作"}], ensure_ascii=False))
    
    # 5. 为图片名添加前缀（records\aiagent\，保留完整扩展名）
    prefixed_images = [f'records\\aiagent\\{img}' for img in image_names]
    print(f"✅ 带前缀的图片名示例：{prefixed_images[0]}")
    
    # 6. 构建邻接矩阵（行/列含PNG图片名，不忽视PNG）
    n = len(prefixed_images)
    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
    # 填充相邻图片的跳转动作
    for i in range(n - 1):
        if i < len(actions):
            adj_matrix[i+1][i] = actions[i]
    
    # 7. 保存为CSV（支持含PNG的图片名）
    try:
        df = pd.DataFrame(adj_matrix, index=prefixed_images, columns=prefixed_images)
        df.to_csv(output_csv, encoding='utf-8-sig', index=True)
        print(f"✅ 邻接矩阵已成功保存（含PNG图片名）：{output_csv}")
        print(f"📊 矩阵维度：{n}×{n}，有效跳转动作数：{len(actions)}")
    except Exception as e:
        print(f"❌ 保存邻接矩阵CSV出错: {e}")
    
    return output_csv


def process_subfolder(subfolder_path, output_base_dir):
    """处理单个子文件夹中的图片"""
    print("\n" + "="*80)
    print(f"📂 开始处理子文件夹: {subfolder_path}")
    print("="*80)
    
    # 创建对应的输出目录
    subfolder_name = os.path.basename(subfolder_path)
    output_subfolder = os.path.join(output_base_dir, subfolder_name)
    os.makedirs(output_subfolder, exist_ok=True)
    
    trajectory_path = os.path.join(output_subfolder, "trajectory_v0.txt")
    
    # 1. 获取子文件夹中的图片文件（仅PNG/JPG/JPEG，不遗漏PNG）
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [
        os.path.join(subfolder_path, f) 
        for f in os.listdir(subfolder_path) 
        if os.path.isfile(os.path.join(subfolder_path, f)) 
        and f.lower().endswith(image_extensions)
    ]
    
    if not image_files:
        print(f"❌ 子文件夹中未找到图片文件（仅支持PNG/JPG/JPEG）")
        return
    
    # 统计初始图片格式分布
    init_format_count = {}
    for img_path in image_files:
        ext = os.path.splitext(img_path)[1].lower()
        init_format_count[ext] = init_format_count.get(ext, 0) + 1
    init_format_info = [f"{k[1:].upper()}: {v}张" for k, v in init_format_count.items()]
    print(f"✅ 找到图片（格式分布：{', '.join(init_format_info)}），共 {len(image_files)} 张：")
    for img_path in image_files[:5]:  # 只显示前5张，避免输出过多
        print(f"   - {os.path.basename(img_path)}（{os.path.splitext(img_path)[1][1:].upper()}）")
    if len(image_files) > 5:
        print(f"   - ... 还有 {len(image_files) - 5} 张图片未显示")
    
    # 2. 分析图片生成轨迹文件（支持PNG/JPG）
    print("\n" + "="*40)
    print("🚀 阶段1：分析图片（PNG/JPG）生成轨迹文件")
    print("="*40)
    results, query_content = analyze_image_pairs(image_files, subfolder_path, max_workers=1)
    
    # 3. 检查轨迹文件
    if not os.path.exists(trajectory_path):
        print(f"\n❌ 轨迹文件生成失败，无法继续生成邻接矩阵")
        return
    print(f"\n✅ 轨迹文件生成成功（含PNG图片路径）：{trajectory_path}")
    
    # 4. 生成邻接矩阵CSV（含PNG图片名）
    print("\n" + "="*40)
    print("🚀 阶段2：生成邻接矩阵CSV（支持PNG图片名）")
    print("="*40)
    generate_adjacency_matrix(trajectory_path, output_subfolder, query_content)
    
    # 5. 子文件夹处理结束
    print("\n" + "="*80)
    print(f"✅ 子文件夹处理完成: {subfolder_path}")
    print(f"📁 输出目录: {output_subfolder}")
    print("="*80)


def main():
    """主函数：处理指定目录下所有子文件夹中的图片，生成轨迹和邻接矩阵"""
    # 设置输入和输出路径
    input_dir = 'dfs\\pic'  # 输入目录：dfs\pic
    output_dir = 'dfs\\trajectory'  # 输出目录：dfs\trajectory
    
    print("="*80)
    print("🎯 图片（PNG/JPG）→轨迹→邻接矩阵全流程启动")
    print(f"📂 输入目录：{input_dir}")
    print(f"📁 输出目录：{output_dir}")
    print("="*80)
    
    # 1. 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在：{input_dir}")
        print("💡 请确保在项目根目录下创建 dfs\\pic 目录并放入图片子文件夹")
        return
    
    # 2. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ 输出目录已创建：{output_dir}")
    
    # 3. 获取所有子文件夹
    subfolders = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    
    if not subfolders:
        print(f"❌ 输入目录下未找到任何子文件夹：{input_dir}")
        print("💡 请在 dfs\\pic 目录下创建子文件夹，每个子文件夹包含图片和query.txt")
        return
    
    print(f"✅ 找到 {len(subfolders)} 个子文件夹，将依次处理：")
    for i, subfolder in enumerate(subfolders, 1):
        subfolder_name = os.path.basename(subfolder)
        print(f"   {i}. {subfolder_name}")
    
    # 4. 逐个处理子文件夹
    for i, subfolder in enumerate(subfolders, 1):
        subfolder_name = os.path.basename(subfolder)
        print(f"\n\n===== 处理子文件夹 {i}/{len(subfolders)}: {subfolder_name} =====")
        process_subfolder(subfolder, output_dir)
    
    # 5. 全流程结束
    print("\n" + "="*80)
    print("🎉 所有子文件夹处理完成！")
    print(f"📊 总计处理了 {len(subfolders)} 个子文件夹")
    print(f"📁 输出文件保存在：{output_dir}")
    print("💡 每个子文件夹中已生成对应的轨迹文件和邻接矩阵")
    print("="*80)


if __name__ == "__main__":
    main()
