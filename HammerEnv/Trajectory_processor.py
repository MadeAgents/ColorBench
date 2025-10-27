import json
import base64
import os
from pathlib import Path

def decode_base64_to_image(base64_str, output_path):
    try:
        if base64_str.startswith('data:image/'):
            base64_str = base64_str.split(',', 1)[1]
        
        image_data = base64.b64decode(base64_str)
        
        with open(output_path, 'wb') as f:
            f.write(image_data)
            
        print(f"图片已保存至: {output_path}")
        return True
    except Exception as e:
        print(f"转换失败: {e}")
        return False

def process_json_file(json_file_path, update_json=False):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'history' not in data or not isinstance(data['history'], list):
            print("JSON结构不符合预期，未找到history数组")
            return False
            
        history_items = data['history']
        if not history_items:
            print("history数组为空")
            return False
            
        json_filename = os.path.splitext(os.path.basename(json_file_path))[0]
        json_dir = os.path.dirname(json_file_path)
        
        success_count = 0
        for i, item in enumerate(history_items):
            if not isinstance(item, dict) or 'screenshot' not in item:
                print(f"第 {i} 项不包含screenshot字段，跳过")
                continue
                
            screenshot = item['screenshot']
            if not isinstance(screenshot, str):
                print(f"第 {i} 项的screenshot不是字符串类型，跳过")
                continue
                
            output_file = os.path.join(json_dir, f"image_{json_filename}_{i}.png")
            if decode_base64_to_image(screenshot, output_file):
                success_count += 1    
                if update_json:
                    item['screenshot'] = os.path.basename(output_file)
                    print(f"已更新JSON中的screenshot字段为: {os.path.basename(output_file)}")
        print(f"处理完成: 共 {len(history_items)} 个历史记录，成功转换 {success_count} 张图片")
        
        if update_json and success_count > 0:
            base_name, ext = os.path.splitext(json_file_path)
            updated_json_path = f"{base_name}_1{ext}"
            with open(updated_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"已将更新后的JSON保存至: {updated_json_path}")
        return True
        
    except FileNotFoundError:
        print(f"错误: 文件 '{json_file_path}' 不存在")
        return False
    except json.JSONDecodeError:
        print(f"错误: 文件 '{json_file_path}' 不是有效的JSON格式")
        return False
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False

def process_all_json_files(root_dir, update_json=False):
    total_json_files = 0
    processed_successfully = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                json_file_path = os.path.join(dirpath, filename)
                total_json_files += 1
                
                print(f"\n处理 JSON 文件: {os.path.relpath(json_file_path, root_dir)}")
                success = process_json_file(json_file_path, update_json)
                
                if success:
                    processed_successfully += 1
    
    print(f"\n全部处理完成！共发现 {total_json_files} 个 JSON 文件，成功处理 {processed_successfully} 个。")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory = os.path.join(script_dir, "records")
    update_json_files = True
    process_all_json_files(root_directory, update_json_files)