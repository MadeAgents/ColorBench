import json
import os
import re
import base64
from pathlib import Path
from PIL import Image, ImageDraw
from openai import OpenAI
from io import BytesIO

class ImageJumpParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.main_image = None  
        self.jump_relations = {}  
    
    def load_and_parse(self):
        """读取文件并解析JSON数据"""
        try:
            # 读取文件内容
            content = Path(self.file_path).read_text(encoding='utf-8')
            
            # 处理可能的JSON格式问题（如最后可能有多余逗号）
            content = content.rstrip(',\n') + '}' if content.endswith(',') else content
            
            # 解析JSON
            self.data = json.loads(content)
            
            # 获取主图片（第一个图片）
            if self.data:
                self.main_image = next(iter(self.data.keys()))
            
            # 构建跳转关系字典
            self._build_jump_relations()
            return True
            
        except FileNotFoundError:
            print(f"错误：文件 {self.file_path} 未找到")
            return False
        except json.JSONDecodeError as e:
            print(f"错误：JSON格式解析失败 - {str(e)}")
            return False
        except Exception as e:
            print(f"错误：处理文件时发生异常 - {str(e)}")
            return False
    
    def _build_jump_relations(self):
        """构建跳转关系：当前图片 -> {目标图片: [跳转条件]}"""
        for current_img, targets in self.data.items():
            self.jump_relations[current_img] = {}
            for target_img, conditions in targets.items():
                self.jump_relations[current_img][target_img] = conditions
    
    def get_main_image(self):
        return self.main_image
    
    def get_jumps_from_image(self, image_path):
        if image_path in self.jump_relations:
            return self.jump_relations[image_path]
        return None
    
    def check_jump_condition(self, current_img, action_type, x, y):
        if current_img not in self.jump_relations:
            return None
            
        for target_img, conditions in self.jump_relations[current_img].items():
            for cond in conditions:
                if ('action_type' in cond and 'x' in cond and 'y' in cond and
                    cond['action_type'] == action_type and 
                    cond['x'] == x and 
                    cond['y'] == y):
                    return target_img
        return None
    
    def print_relations(self):
        if not self.jump_relations:
            print("没有解析到跳转关系数据")
            return
            
        print(f"主图片: {self.main_image}\n")
        print("图片跳转关系列表：")
        for current_img, targets in self.jump_relations.items():
            print(f"\n当前图片: {current_img}")
            if not targets:
                print("  无跳转目标")
                continue
            for target_img, conditions in targets.items():
                print(f"  可跳转到: {target_img}")
                valid_condition_count = 0
                for cond in conditions:
                    if 'action_type' in cond and 'x' in cond and 'y' in cond:
                        valid_condition_count += 1
                        print(f"    条件{valid_condition_count}: 动作类型={cond['action_type']}, 坐标=({cond['x']},{cond['y']})")
                if valid_condition_count == 0:
                    print("    无有效跳转条件（存在缺失值）")


class ImageAnalyzer:
    def __init__(self, image_dir=None, output_dir="aiagent3", 
                 openai_base_url="http://your-api-endpoint/v1", api_key="empty",
                 min_box_size=10, max_box_ratio=0.2,
                 annotator_model1="gui-owl-32b",  # 倾向于较大边界框的模型
                 annotator_model2="gui-owl-32b",  # 倾向于较小边界框的模型
                 referee_model="gui-owl-32b"):
        self.image_dir = image_dir if image_dir else "."
        self.output_dir = output_dir
        self.min_box_size = min_box_size  # 最小边界框尺寸（像素）
        self.max_box_ratio = max_box_ratio  # 最大边界框占图片比例
        self.annotator_model1 = annotator_model1  # 倾向于较大边界框的模型
        self.annotator_model2 = annotator_model2  # 倾向于较小边界框的模型
        self.referee_model = referee_model      # 裁判模型
        self._create_output_dir()
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=openai_base_url,
            api_key=api_key,
        )
        
        # 存储所有图片的边界框结果
        self.all_bounding_boxes = {}
    
    def _create_output_dir(self):
        """创建输出文件夹（如果不存在）"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"已创建输出文件夹: {self.output_dir}")
        else:
            print(f"输出文件夹已存在: {self.output_dir}")
    
    def calculate_center(self, box):
        """计算边界框的中心点坐标"""
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (round(center_x), round(center_y))
    
    def adjust_box_to_center(self, box, target_center):
        """调整边界框，使目标点成为边界框的中心"""
        x1, y1, x2, y2 = box
        target_x, target_y = target_center
        
        # 计算当前边界框的宽度和高度
        width = x2 - x1
        height = y2 - y1
        
        # 计算新的边界框坐标，使目标点成为中心
        new_x1 = round(target_x - width / 2)
        new_y1 = round(target_y - height / 2)
        new_x2 = round(target_x + width / 2)
        new_y2 = round(target_y + height / 2)
        
        return (new_x1, new_y1, new_x2, new_y2)
    
    def constrain_box_size(self, box, image_width, image_height):
        """限制边界框大小，防止过大或过小"""
        x1, y1, x2, y2 = box
        
        # 计算当前宽高
        current_width = x2 - x1
        current_height = y2 - y1
        
        # 确保最小尺寸
        if current_width < self.min_box_size:
            current_width = self.min_box_size
        if current_height < self.min_box_size:
            current_height = self.min_box_size
            
        # 计算最大允许尺寸（基于图片比例）
        max_width = image_width * self.max_box_ratio
        max_height = image_height * self.max_box_ratio
        
        # 确保不超过最大尺寸
        if current_width > max_width:
            current_width = max_width
        if current_height > max_height:
            current_height = max_height
        
        # 计算中心
        center_x, center_y = self.calculate_center(box)
        
        # 基于中心和新的宽高重新计算边界框
        new_x1 = round(center_x - current_width / 2)
        new_y1 = round(center_y - current_height / 2)
        new_x2 = round(center_x + current_width / 2)
        new_y2 = round(center_y + current_height / 2)
        
        return (new_x1, new_y1, new_x2, new_y2)
        
    def parse_model_output(self, model_response):
        """解析模型输出，提取边界框坐标和点击意图"""
        if not model_response:
            print("模型输出为空")
            return None
            
        # 打印原始模型输出用于调试
        print("原始模型输出:", model_response[:200] + "..." if len(model_response) > 200 else model_response)
        
        # 边界框匹配模式
        box_pattern = r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]"
        box_matches = re.findall(box_pattern, model_response)
        
        # 如果找到多个边界框，取第一个
        box_match = box_matches[0] if box_matches else None
        
        # 提取点击意图和功能描述
        purpose_text = "未明确点击意图"
        patterns = [
            r"点击目的是：(.*?)(。|,|;|\n|$)",
            r"用户意图是：(.*?)(。|,|;|\n|$)",
            r"功能是：(.*?)(。|,|;|\n|$)",
            r"作用是：(.*?)(。|,|;|\n|$)",
            r"用于：(.*?)(。|,|;|\n|$)",
            r"会触发：(.*?)(。|,|;|\n|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_response, re.DOTALL)
            if match:
                purpose_text = match.group(1).strip()
                break
        
        # 如果没找到，尝试从整体文本中提取
        if purpose_text == "未明确点击意图":
            cleaned_response = re.sub(box_pattern, "", model_response)
            purpose_text = cleaned_response.strip()[:100]
        
        if box_match:
            try:
                coords = tuple(map(lambda x: round(float(x)), box_match))
                return {
                    "intent": purpose_text,
                    "coords": coords,
                    "raw_response": model_response
                }
            except ValueError as e:
                print(f"转换坐标时出错: {e}")
                return None
        
        # 尝试从文本中提取坐标信息
        print("未找到标准格式的边界框，尝试提取坐标...")
        num_pattern = r"(\d+\.?\d*)"
        nums = re.findall(num_pattern, model_response)
        if len(nums) >= 4:
            try:
                coords = tuple(map(lambda x: round(float(x)), nums[:4]))
                return {
                    "intent": purpose_text,
                    "coords": coords,
                    "raw_response": model_response
                }
            except ValueError as e:
                print(f"转换坐标时出错: {e}")
        
        print("无法从模型输出中提取有效的边界框信息")
        return None

    def ensure_point_in_box(self, box, point):
        """确保点击坐标点在边界框内，如果不在则调整边界框"""
        x1, y1, x2, y2 = box
        px, py = point
        
        # 检查点是否在边界框内
        if x1 <= px <= x2 and y1 <= py <= y2:
            return box  # 点已在框内，无需调整
            
        # 调整边界框以包含点（最小化调整）
        new_x1 = min(x1, px)
        new_y1 = min(y1, py)
        new_x2 = max(x2, px)
        new_y2 = max(y2, py)
        
        return (round(new_x1), round(new_y1), round(new_x2), round(new_y2))

    def analyze_with_model(self, image_path, action_type, x, y, annotation_id=1, total_points=None, current_index=None):
        """分析单个交互点，根据标注ID设置不同的尺寸偏好"""
        try:
            # 获取图片尺寸用于提示
            
            with Image.open(image_path) as original_img:
                # 确保图片模式支持绘制
                if original_img.mode != 'RGB':
                    original_img = original_img.convert('RGB')
                
                # 复制图片避免修改原始文件
                annotated_img = original_img.copy()
                draw = ImageDraw.Draw(annotated_img)
                
                # 获取图片尺寸
                img_width, img_height = annotated_img.size
                
                # 确保坐标在图片范围内
                px, py = int(x), int(y)
                px = max(0, min(px, img_width - 1))
                py = max(0, min(py, img_height - 1))
                
                # 在图片上标注动作位置（大红点）
                point_radius = 8
                draw.ellipse([px - point_radius, py - point_radius, 
                            px + point_radius, py + point_radius], 
                            fill="#FF0000", outline="#FFFFFF", width=2)
                
                # 添加点击坐标文字标注
                draw.text((px + 10, py - 10), f"({px},{py})", fill="#FF0000", stroke_width=1, stroke_fill="#FFFFFF")
                
                print(f"已在坐标 ({px}, {py}) 标注红点")
        
            # 第二步：将标注后的图片转换为base64
            buffer = BytesIO()
            annotated_img.save(buffer, format='PNG')
            buffer.seek(0)
            encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # with open(image_path, "rb") as image_file:
            #     encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

            # 构建提示信息，根据标注ID设置不同的尺寸偏好

            point_context = ""
            if total_points and current_index:
                point_context = f"这是第{current_index}/{total_points}个交互点的第{annotation_id}次标注，每个点都是独立的操作。"
            else:
                point_context = f"这是该交互点的第{annotation_id}次标注。"
            
            # 计算合理的尺寸范围提示
            min_size_px = self.min_box_size
            max_width_px = int(img_width * self.max_box_ratio)
            max_height_px = int(img_height * self.max_box_ratio)
            
            # 根据标注ID设置不同的尺寸偏好提示
            size_preference = ""
            if annotation_id == 1:
                # 第一个标注：倾向于较大边界框
                size_preference = "特别说明：请生成相对较大的边界框，确保完整包含所有相关的UI元素，宁可稍大也不要遗漏任何能够完成该功能的可能相关的部分，宽度和高度建议在允许范围内取较大值。"
                # size_preference = """
                # 特别说明：请倾向于生成稍大的边界框，确保完整包含该UI元素区域，不要遗漏任何可能相关的部分。
                # """
            else:
                # 第二个标注：倾向于较小边界框
                size_preference = "特别说明：请生成相对较小的边界框，仅包含最核心的UI元素，尽可能紧凑，避免包含任何无关区域，宽度和高度建议在允许范围内取较小值。"
                # size_preference = """
                # 特别说明：请倾向于生成稍小的边界框，仅包含最核心的UI元素区域，尽可能紧凑，避免包含任何无关区域。
                # """
            system_prompt = "你是一个专业的UI元素边界框标注智能体。你的任务是对所给的图片和精确的点击动作，通过分析用户意图，划出能够完成用户意图的精准的点击范围，生成边界框。"
            prompt = f"""
            {point_context}动作类型为{action_type}，用户在图片上的精确点击坐标为({x}, {y})，即图中半径为8的红点位置。图片尺寸为{img_width}x{img_height}像素。

            请精准推测用户的点击目的，并生成最合适的边界框：

            关键要求：
            1. 独立分析此点击，不考虑图片上可能存在的其他点击点；
            2. 确定点击坐标的内容，分析用户点击该位置的精确意图；
            3. 根据用户意图，识别屏幕上实现该功能的具体UI元素（按钮/图标/区域/文字），并生成恰好包围该元素的边界框；
            4. 确保点击坐标点({x}, {y})位于生成的边界框范围内，靠近边界框中心；
            
            边界框尺寸限制：
            - 宽度和高度都不能小于{min_size_px}像素（避免过小）
            - 宽度不能超过{max_width_px}像素（避免过大）
            - 高度不能超过{max_height_px}像素（避免过大）
            {size_preference}

            注意事项：
            1. 如果用户点击的坐标位置周围有精确的颜色划分或者边框存在，可直接将其作为边界框，无需考虑大小倾向；
            2. 如果点击的是内容输入框，标注宽度可以超过限制要求；
            3. 如果点击的是带文字的图标且文字和图标是同一功能，标注区域应该同时包含二者；
            4. 你标注的边框必须只包含用户想点击的UI元素或内容框，不可以包含会造成其他跳转的区域。
            
            输出格式：
            [x1, y1, x2, y2]
            点击意图：[仅针对此点的推测目的]
            功能说明：[该UI元素的具体功能]
            边界框说明：[详细说明为什么这个大小和位置是合适的，如何精准包围目标元素]
            """
           
            
            # 构建消息
            msg = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}
                    ],
                }
            ]
            
            # 根据标注ID选择不同的模型
            model = self.annotator_model1 if annotation_id == 1 else self.annotator_model2
            
            # 调用AI模型（降低温度，提高准确性）
            response = self.client.chat.completions.create(
                model=model,
                messages=msg,
                temperature=0.1,
                max_tokens=400,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"分析点击动作时出错: {str(e)}"
    
    def referee_between_annotations(self, image_path, action_type, x, y, annotation1, annotation2):
        """裁判模型，比较两个标注结果并选择更优的一个"""
        try:
            # 获取图片尺寸
            # with Image.open(image_path) as img:
            #     img_width, img_height = img.size
            with Image.open(image_path) as original_img:
                # 确保图片模式支持绘制
                if original_img.mode != 'RGB':
                    original_img = original_img.convert('RGB')
                
                # 复制图片避免修改原始文件
                annotated_img = original_img.copy()
                draw = ImageDraw.Draw(annotated_img)
                
                # 获取图片尺寸
                img_width, img_height = annotated_img.size
                
                # 确保坐标在图片范围内
                px, py = int(x), int(y)
                px = max(0, min(px, img_width - 1))
                py = max(0, min(py, img_height - 1))
                
                # 在图片上标注动作位置（大红点）
                point_radius = 8
                draw.ellipse([px - point_radius, py - point_radius, 
                            px + point_radius, py + point_radius], 
                            fill="#FF0000", outline="#FFFFFF", width=2)
                
                # 添加点击坐标文字标注
                draw.text((px + 10, py - 10), f"({px},{py})", fill="#FF0000", stroke_width=1, stroke_fill="#FFFFFF")
                
                print(f"已在坐标 ({px}, {py}) 标注红点")
        
            # 第二步：将标注后的图片转换为base64
            buffer = BytesIO()
            annotated_img.save(buffer, format='PNG')
            buffer.seek(0)
            encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

            
            # 预编译正则表达式，避免在f-string中使用反斜杠
            pattern = r"边界框说明：(.*?)(\n|$)"
            
            # 提取两个标注的模型说明
            annot1_desc = "无"
            match1 = re.search(pattern, annotation1['raw_response'], re.DOTALL)
            if match1:
                annot1_desc = match1.group(1)
                
            annot2_desc = "无"
            match2 = re.search(pattern, annotation2['raw_response'], re.DOTALL)
            if match2:
                annot2_desc = match2.group(1)
            
            # 构建裁判提示
            system_prompt = "你是一个专业的UI元素边界框标注裁判智能体。你的任务是对所给的图片和精确的点击动作，及其两个UI边界标注结果，选择出最能够完整的覆盖用户点意图的一个。"
            
            prompt = f"""
            请比较以下两个针对同一交互点的标注结果，选择更优的一个。
            
            交互点信息：
            - 动作类型：{action_type}
            - 点击坐标：({x}, {y})（图中半径为8的红色点位置）
            - 图片尺寸：{img_width}x{img_height}像素
            
            优秀边界框的标准：
            1. 准确包围用户实际点击的UI元素（按钮/图标/可交互区域/文字）；
            2. 不遗漏周围点击后能达到相同跳转结果的UI区域（按钮/图标/可交互区域）；
            3. 点击坐标({x}, {y})靠近边界框中心；
            4. 只要边界框中的任意位置被点击，都能实现相同的跳转结果，则倾向于选择更大的边界框。否则选择更小的边界框。
            
            标注1（较大边界框倾向）：
            边界框：{annotation1['coords']}
            点击意图：{annotation1['intent']}
            模型说明：{annot1_desc}
            
            标注2（较小边界框倾向）：
            边界框：{annotation2['coords']}
            点击意图：{annotation2['intent']}
            模型说明：{annot2_desc}
            
            请基于上述标准，判断哪个标注更优。
            输出格式：
            更优标注：1 或 2
            判断理由：[详细说明为什么选择该标注，比较两个标注的优缺点]
            """
            # 4. 符合尺寸限制（不小于{self.min_box_size}像素，不超过图片的{self.max_box_ratio*100}%）
            # # 读取并编码图片
            # with open(image_path, "rb") as image_file:
            #     encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            
            # 构建消息
            msg = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}
                    ],
                }
            ]
            
            # 调用裁判模型
            response = self.client.chat.completions.create(
                model=self.referee_model,
                messages=msg,
                temperature=0.1,  # 低温度确保判断更稳定
                max_tokens=500,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"裁判模型比较时出错: {str(e)}"
    
    def draw_bounding_boxes(self, image_path, boxes_with_points, output_suffix="_with_optimized_boxes"):
        """绘制优化后的边界框，清晰区分每个点"""
        # 检查文件是否存在
        if not os.path.exists(image_path):
            print(f"错误：文件 {image_path} 不存在")
            return None
        
        try:
            with Image.open(image_path) as img:
                draw = ImageDraw.Draw(img)
                width, height = img.size
                
                # 颜色配置（使用明显区分的颜色）
                box_colors = [
                    "#FF5733", "#33FF57", "#3357FF", 
                    "#F3FF33", "#FF33F3", "#33FFF3", 
                    "#F333FF", "#FF9933", "#33FF99", "#9933FF"
                ]  # 更多颜色选项，确保多个点能区分
                click_color = "#FF9900"  # 点击点颜色
                center_color = "#00FFFF"  # 中心点标记颜色
                text_color = "#FFFFFF"   # 文本颜色
                text_bg_color = "#000000CC"  # 半透明黑色背景
                
                # 绘制每个独立的边界框和点击坐标
                for i, item in enumerate(boxes_with_points):
                    # 1. 绘制功能边界框
                    box_coords = item["coords"]
                    intent = item.get("intent", "未明确意图")
                    function = item.get("function", "未知功能")
                    original_click = item.get("original_point")
                    point_index = item.get("point_index", i+1)
                    annotation_info = item.get("annotation_info", "标注信息未提供")
                    
                    # 确保边界框坐标是整数
                    x1, y1, x2, y2 = map(int, box_coords)
                    
                    # 确保边界框包含原始点击点并使其位于中心
                    if original_click:
                        # 先确保点在框内
                        x1, y1, x2, y2 = self.ensure_point_in_box((x1, y1, x2, y2), original_click)
                        # 再调整框使点位于中心
                        # x1, y1, x2, y2 = self.adjust_box_to_center((x1, y1, x2, y2), original_click)
                    
                    # 限制边界框大小
                    # x1, y1, x2, y2 = self.constrain_box_size((x1, y1, x2, y2), width, height)
                    
                    # 确保边界框在图片范围内且有效
                    x1 = max(0, min(x1, width))
                    y1 = max(0, min(y1, height))
                    x2 = max(x1 + self.min_box_size, min(x2, width))  # 确保最小宽度
                    y2 = max(y1 + self.min_box_size, min(y2, height))  # 确保最小高度
                    x2 = min(x2, width)  # 确保最小宽度
                    y2 = min(y2, height)  # 确保最小高度
                    
                    # 选择边界框颜色（循环使用颜色列表）
                    box_color = box_colors[i % len(box_colors)]
                    
                    # 绘制边界框，添加点编号便于识别
                    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
                    draw.text((x1, y1), f"点 {point_index}", fill=box_color)
                    
                    # 绘制尺寸信息
                    box_width = x2 - x1
                    box_height = y2 - y1
                    draw.text((x1, y2 + 5), f"{box_width}x{box_height}", fill=box_color)
                    
                    # 绘制功能说明标签
                    label_lines = [
                        f"意图: {intent[:20]}",
                        f"功能: {function[:20]}",
                        f"标注: {annotation_info[:15]}"
                    ]
                    
                    # 绘制带背景的文本标签，位置错开避免重叠
                    label_y = max(0, y1 - 70 - (i % 3) * 20)  # 轻微错开多个标签
                    for j, line in enumerate(label_lines):
                        # 绘制文本背景
                        text_bbox = draw.textbbox((x1, label_y + j*20), line)
                        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, 
                                      text_bbox[2]+2, text_bbox[3]+2], 
                                     fill=text_bg_color)
                        # 绘制文本
                        draw.text((x1, label_y + j*20), line, fill=text_color)
                    
                    # 2. 绘制原始点击点（中心点），标记编号
                    if original_click:
                        px, py = original_click
                        px, py = int(px), int(py)
                        
                        # 确保点击点在图片范围内
                        px = max(0, min(px, width))
                        py = max(0, min(py, height))
                        
                        # 绘制中心点标记（十字线+中心点+编号）
                        # draw.line([px-5, py, px+5, py], fill=center_color, width=2)  # 水平线
                        # draw.line([px, py-5, px, py+5], fill=center_color, width=2)  # 垂直线
                        # draw.ellipse([px-3, py-3, px+3, py+3], fill=center_color)
                        draw.ellipse([px-8, py-8, px+8, py+8], fill="#FF0000", width=20)  # 大红点，白色边框
    
                        draw.text((px+10, py-5), f"{point_index}", fill=click_color)
                
                # 构建输出路径
                file_name = os.path.basename(image_path)
                base_name, ext = os.path.splitext(file_name)
                file_names = image_path.split('/')
                base_name = f'{file_names[-2]}_{file_names[-1]}' 
                output_path = os.path.abspath(os.path.join(self.output_dir, f"{base_name}{output_suffix}{ext}"))
                
                img.save(output_path)
                print(f"已生成带优化边界框的图片：{output_path}")
                return output_path
                
        except Exception as e:
            print(f"绘制时出错: {str(e)}")
            return None
    
    def process_image_jumps(self, parser, image_path=None):
        """处理图片跳转关系：使用双标注+裁判机制优化边界框"""
        # 如果未指定图片路径，使用主图片
        if not image_path:
            image_path = parser.get_main_image()
            if not image_path:
                print("没有找到主图片")
                return False
        
        # 检查图片是否存在
        full_image_path = os.path.join(self.image_dir, image_path) if self.image_dir else image_path
        full_image_path = os.path.abspath(full_image_path)
        if not os.path.exists(full_image_path):
            print(f"错误：图片文件 {full_image_path} 不存在")
            return False
        
        # 初始化当前图片在结果字典中的条目
        if image_path not in self.all_bounding_boxes:
            self.all_bounding_boxes[image_path] = {}
        
        # 获取图片尺寸用于边界框限制
        with Image.open(full_image_path) as img:
            img_width, img_height = img.size
        
        # 获取该图片的所有跳转关系
        jumps = parser.get_jumps_from_image(image_path)
        if not jumps:
            print(f"图片 {image_path} 没有跳转关系")
            return False
        
        # 收集所有点击点（先汇总所有点再处理，便于编号）
        all_conditions = []
        for target_img, conditions in jumps.items():
            for cond in conditions:
                if 'action_type' in cond and 'x' in cond and 'y' in cond:
                    all_conditions.append((target_img, cond))
        
        total_points = len(all_conditions)
        if total_points == 0:
            print(f"图片 {image_path} 没有有效的点击条件")
            return False
        
        print(f"发现 {total_points} 个独立交互点，开始双标注+裁判机制处理...")
        
        # 逐一处理每个交互点，使用双标注+裁判机制
        boxes_with_points = []
        for index, (target_img, cond) in enumerate(all_conditions, 1):
            x, y = cond['x'], cond['y']
            print(f"\n===== 处理第 {index}/{total_points} 个交互点：{cond['action_type']} ({x},{y}) =====")
            
            # 第一次标注 - 倾向于较大边界框
            print("执行第一次标注（倾向较大边界框）...")
            annotation1_response = self.analyze_with_model(
                full_image_path, 
                cond['action_type'], 
                x, y,
                annotation_id=1,
                total_points=total_points,
                current_index=index
            )
            annotation1 = self.parse_model_output(annotation1_response)
            
            # 第二次标注 - 倾向于较小边界框
            print("执行第二次标注（倾向较小边界框）...")
            annotation2_response = self.analyze_with_model(
                full_image_path, 
                cond['action_type'], 
                x, y,
                annotation_id=2,
                total_points=total_points,
                current_index=index
            )
            annotation2 = self.parse_model_output(annotation2_response)
            
            # 检查标注是否有效
            valid_annotations = []
            if annotation1:
                valid_annotations.append(annotation1)
            if annotation2:
                valid_annotations.append(annotation2)
                
            if len(valid_annotations) == 0:
                print(f"第 {index} 个交互点的两次标注都失败")
                continue
            elif len(valid_annotations) == 1:
                # 只有一个有效标注，直接使用
                selected_annotation = valid_annotations[0]
                annotation_source = f"仅标注{1 if annotation1 else 2}有效"
                print(f"只有一个有效标注，选择标注{1 if annotation1 else 2}")
            else:
                # 两个标注都有效，调用裁判模型
                print("调用裁判模型选择更优标注...")
                referee_response = self.referee_between_annotations(
                    full_image_path, cond['action_type'], x, y,
                    annotation1, annotation2
                )
                print("裁判模型输出:", referee_response[:300] + "..." if len(referee_response) > 300 else referee_response)
                
                # 解析裁判结果
                match = re.search(r"更优标注：(\d)", referee_response)
                if match and match.group(1) == "2":
                    selected_annotation = annotation2
                    annotation_source = "裁判选择较小标注"
                    print("裁判选择了较小的标注")
                else:
                    selected_annotation = annotation1
                    annotation_source = "裁判选择较大标注"
                    print("裁判选择了较大的标注")
            
            # 提取点击意图和功能说明
            intent = selected_annotation["intent"]
            function = "未知功能"
            
            # 从模型响应中提取功能说明
            if selected_annotation['raw_response']:
                func_match = re.search(r"功能说明：(.*?)(\n|$)", selected_annotation['raw_response'])
                if func_match:
                    function = func_match.group(1).strip()
            
            bbox = selected_annotation["coords"]
            original_click = (x, y)
            
            # 确保点击点位于边界框中心
            current_center = self.calculate_center(bbox)
            if current_center != original_click:
                # print(f"调整边界框使点击点位于中心 (原中心: {current_center}, 目标: {original_click})")
                # bbox = self.adjust_box_to_center(bbox, original_click)
                # 再次确保点在框内
                bbox = self.ensure_point_in_box(bbox, original_click)
            
            # 限制边界框大小，防止过大或过小
            # bbox = self.constrain_box_size(bbox, img_width, img_height)
            
            # 添加到列表，包含点编号和标注来源
            boxes_with_points.append({
                "coords": bbox,
                "intent": intent,
                "function": function,
                "original_point": original_click,
                "point_index": index,
                "annotation_info": annotation_source
            })
            
            # 保存边界框到结果字典
            # 确保目标图片在当前图片的条目中存在
            if target_img not in self.all_bounding_boxes[image_path]:
                self.all_bounding_boxes[image_path][target_img] = []
            
            # 添加动作和边界框信息
            self.all_bounding_boxes[image_path][target_img].append({
                "action_type": cond['action_type'],
                "x": x,
                "y": y,
                "bounding_box": list(bbox)  # 转换为列表以便JSON序列化
            })
        
        if not boxes_with_points:
            print(f"图片 {image_path} 没有成功处理的交互点")
            return False
        
        # 绘制所有优化后的边界框和点击点
        self.draw_bounding_boxes(full_image_path, boxes_with_points)
        return True
    
    def save_bounding_boxes_to_file(self, filename="aiagent3.json"):
        """将所有边界框信息保存到JSON格式的文件"""
        try:
            # 构建完整文件路径
            file_path = os.path.join(self.output_dir, filename)
            
            # 写入JSON数据，确保中文正常显示且格式美观
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.all_bounding_boxes, f, ensure_ascii=False, indent=2)
            
            print(f"已将边界框信息保存到JSON文件: {file_path}")
            return True
        except Exception as e:
            print(f"保存边界框信息时出错: {str(e)}")
            return False


def main():
    # 配置
    json_file = "/home/notebook/code/personal/S9060045/demonstration_based_learning/completely_graph_0909.json"  # JSON文件路径
    image_directory = "/home/notebook/code/personal/S9060045/demonstration_based_learning/data"  # 图片所在目录
    output_directory = "/home/notebook/code/personal/S9060045/demonstration_based_learning/aiagent09092"  # 输出文件夹
    openai_base_url = "http://your-api-endpoint/v1"  # 大模型服务地址
    api_key = "empty"  # API密钥
    min_box_size = 10  # 最小边界框尺寸（像素）
    max_box_ratio = 0.15  # 最大边界框占图片比例（15%）
    max_images_to_process = None  # 只处理前5张图片，设置为None处理所有图片
    # 可以指定不同的模型用于两种标注和裁判
    annotator_model1 = "gui-owl-32b"  # 用于较大边界框标注
    annotator_model2 = "gui-owl-32b"  # 用于较小边界框标注
    referee_model = "gui-owl-32b"     # 裁判模型
    
    # 1. 解析JSON文件获取跳转关系
    parser = ImageJumpParser(json_file)
    if not parser.load_and_parse():
        print("解析JSON文件失败")
        return
    
    # 打印跳转关系
    parser.print_relations()
    
    # 2. 处理图片并使用双标注+裁判机制优化边界框
    analyzer = ImageAnalyzer(
        image_directory,
        output_dir=output_directory,
        openai_base_url=openai_base_url,
        api_key=api_key,
        min_box_size=min_box_size,
        max_box_ratio=max_box_ratio,
        annotator_model1=annotator_model1,
        annotator_model2=annotator_model2,
        referee_model=referee_model
    )
    
    # 获取所有图片路径列表
    all_image_paths = list(parser.jump_relations.keys())
    if max_images_to_process:
        print(f"\n共发现 {len(all_image_paths)} 张图片，将只处理前 {max_images_to_process} 张...")
        images_to_process = all_image_paths[:max_images_to_process]
    else:
        print(f"\n共发现 {len(all_image_paths)} 张图片，将处理所有图片...")
        images_to_process = all_image_paths  # 处理所有图片
    
    # 处理选定的图片
    for i, image_path in enumerate(images_to_process, 1):
        print(f"\n===== 处理第 {i}/{len(images_to_process)} 张图片: {image_path} =====")
        analyzer.process_image_jumps(parser, image_path)
    
    # 保存边界框信息到JSON文件
    analyzer.save_bounding_boxes_to_file()


if __name__ == "__main__":
    main()
    print("\n所有任务处理完成，结果已保存到aiagent3文件夹")
    