import os
import numpy as np
import base64
import requests
from openai import OpenAI

def calculate_cos_similarity_A_and_Batch_B(A, B):
    dot_product = np.dot(A, B.T)
    # 计算模长
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B, axis=1)  # (4,)
    # 计算余弦相似度
    cosine_sim = dot_product / (norm_B * norm_A)
    return cosine_sim


def calculate_cos_similarity_A_and_B(A, B):
    dot_product = np.dot(B, A)
    # 计算模长
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    # 计算余弦相似度
    cosine_sim = dot_product / (norm_B * norm_A)
    return cosine_sim

def calculate_iou(box1, box2):
    """
    计算两个边界框的交并比（IoU）
    box1, box2: {x_min:,x_max:,y_min:,y_max:}
    """
    x1 = max(box1['x_min'], box2['x_min'])
    y1 = max(box1['y_min'], box2['y_min'])
    x2 = min(box1['x_max'], box2['x_max'])
    y2 = min(box1['y_max'], box2['y_max'])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    area_box2 = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    union = area_box1 + area_box2 - intersection

    if union == 0:
        return 0.0
    else:
        return intersection / union

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv('VLM_API_KEY', 'empty')
        self.base_url = os.getenv('VLM_BASE_URL', 'http://demonstration.oppo.test/v1')
        if not self.api_key:
            raise ValueError("VLM_API_KEY environment variable is not set")
        if not self.base_url:
            raise ValueError("VLM_BASE_URL environment variable is not set")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        # self.model = os.environ.get('model_name')
        self.model = "qwen2.5-vl-72b-instruct"

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_response_vlm(self, messages, max_retries=20, retry_delay=5, stream=False, temperature=0.1, **kwargs):
        """
        Get response from VLM model with single image input
        """
       
        retries = 0
        while retries <= max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=stream,
                    **kwargs
                ).choices[0].message.content

                return response
            
            except requests.exceptions.RequestException as e:
                print(f"Request failed with exception: {e}")
            
            retries += 1
            if retries <= max_retries:
                print(f"Retrying... Attempt {retries}/{max_retries}")
                time.sleep(retry_delay)
        
        print("Max retries reached. Request failed.")
        return None

