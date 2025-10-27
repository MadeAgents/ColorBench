import os
import numpy as np
import base64
import requests
import logging
import json
import copy
import random
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

from PIL import Image
from FlagEmbedding import BGEM3FlagModel, FlagModel
logger = logging.getLogger(__name__)
from pathlib import Path

import gzip
from datetime import datetime


from src.utils import LLMClient
from dotenv import load_dotenv
load_dotenv()

model = FlagModel( "path/to/your/bge-m3-model", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：", use_fp16=True)   
logger.info(f"Model loaded Successfully!")
agent = LLMClient()
use_threading = False

nodes_save_path = './data/results/nodes/'  # 节点保存路径(包含图片)

class ScreenNode:
    def __init__(self, data=None, node_id=-1, app=None):
        self.node_id = node_id  # -1 means not set yet
        self.screenlists : list[ScreenShot] = []  
        self.ui_element_edge_list : list[UIElementEdge] = list()  
        self.next_node_id_list : set[str] = set()  

        if data is not None:
            screenshot = ScreenShot(data['screenshot'], app=app)
            self.screenlists.append(screenshot)
            uielement = UIElementEdge(data)
            self.ui_element_edge_list.append(uielement)
        self.app=app 

    def get_node_info(self):
        # return information in dictionary
        node_info = {
            'node_id': self.node_id,
            'app': self.app,
            # 'node_description': self.node_description,
            'screenlists': [screenshot.get_screenshot_info() for screenshot in self.screenlists],
            'ui_element_edge_list': [edge.get_edge_info() for edge in self.ui_element_edge_list],
            'next_node_id_list': list(self.next_node_id_list)
        }
        return node_info

    def calculate_similarity(self, new_node, threshold=0.8):

        text_similarity = self._calculate_node_similarity_by_text(new_node)
        logger.info(f"Average text similarity with node{self.node_id}: {text_similarity}")
        if text_similarity > 0.9:
            return 1.0
        if text_similarity > threshold:
            vlm_similarity = self._calculate_node_similarity_by_vlm(new_node)
            logger.info(f"Average VLM similarity with node{self.node_id}: {vlm_similarity}")
            return 0.5*text_similarity + 0.5 * vlm_similarity
        else:
            return 0.0  # low text similarity, no need to calculate vlm

    def _calculate_node_similarity_by_text(self, new_node):
        if len(self.screenlists) == 0:
            return 0.0

        if not isinstance(new_node, ScreenNode):
            raise TypeError("new_node must be an instance of ScreenNode")
        if self.app != new_node.app:
            return 0.0
        
        new_embedding = new_node.screenlists[0].description_embedding
        similarity = 0.0
        # randomly select five screenshots to compare
        if len(self.screenlists) > 5:
            selected_screens = random.sample(self.screenlists, 5)
        else:
            selected_screens = self.screenlists
        for screen in selected_screens:
            similarity += self._calculate_description_similarity(new_embedding, screen.description_embedding)  
        similarity /= len(selected_screens)  
        return similarity

    def _calculate_node_similarity_by_vlm(self, new_node):
        if len(self.screenlists) == 0:
            return 0.0

        system_prompt = {
            "role": "system",
            "content": [
                {"type": "text","text": "你是一个GUI AGENT。请忽略截图中因时间导致的变化，从页面格式、页面作用、动作关系等方面，判断所给的两张图片是否属于同一个页面状态。\n注意事项：\1 手机主屏幕页面全部视作同一页面；\n2. 忽略截图中因时间和推荐内容更新导致的页面变化，只关注动作导致的变化。\n3. 特别关注导航栏标签，在应用中，不同的导航栏标签意味着不同的页面；\n4. 动作导致的页面状态变化应该算作不同页面，如果需要在一张图中执行某动作才能变成另一张图，则这两张图是不同的页面状态；\n"},
            ]
        }
    
        b_img = new_node.screenlists[0].base64_image
        b_path = new_node.screenlists[0].screenshot_path
        similarity = 0.0
        if len(self.screenlists) > 3:
            selected_screenshots = random.sample(self.screenlists, 3)
        else:
            selected_screenshots = self.screenlists
        for screenshot in selected_screenshots:
            a_img = screenshot.base64_image
            messages=[system_prompt, {
                "role": "user",
                "content": [
                    {"type": "image_url","image_url": {"url":f"data:image/png;base64,{a_img}"}},
                    {"type": "image_url","image_url": {"url":f"data:image/png;base64,{b_img}"}},
                    {"type": "text","text": "\n\n回答：是/否（“是”表示同一页面状态，“否”表示不同页面或者同一页面的不同状态）\n原因："},
                ]
            }]
            
            description = agent.get_response_vlm(messages, temperature=0.0).replace("\n\n","\n")
            logger.info(f'比较{b_path}和{screenshot.screenshot_path}的相似度解释\n {description}')
            # parse response
            try:
                if "原因" in description:
                    answer = description.split('原因')[0].strip()
                    if '不是' not in answer and '否' not in answer and "不属于" not in answer and "不同" not in answer:
                        similarity += 1.0
                elif '不是' not in answer and '否' not in answer and "不属于" not in answer and "不同" not in answer:
                    similarity += 1.0
            except Exception as e:
                logger.info(f"Error parsing VLM response\n: {e}")
        
        similarity /= len(selected_screenshots)
        return similarity

    def _calculate_description_similarity(self, a_embedding, b_embedding):
        sim = np.dot(a_embedding, b_embedding.T) 
        sim = sim / (np.linalg.norm(a_embedding) * np.linalg.norm(b_embedding))
        return sim


    def set_nodeid(self, node_id, merge=False):
        if self.node_id == -1:
            self.node_id = node_id
        elif merge:
            logger.info(f"Merging node {self.node_id} with new ID {node_id}")
            self.node_id = node_id
        else:
            logger.warning(f"Node ID already set to {self.node_id}, cannot change to {node_id} without merge flag.")
        # reset source_node id for all edges
        for edge in self.ui_element_edge_list:
            edge.set_source_node(self.node_id)


class ScreenShot:
    def __init__(self, image_path, description=None, app=None):
        self.screenshot_path = image_path
        self.app = app
        with open(image_path, "rb") as img_file:
            self.base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        if description:
            self.description =  description
        else:
            self.description =  self._generate_description_zh(self.base64_image)
        self.description_embedding = model.encode(self.description)
    
    def save_screenshot(self, save_path):
        new_name_list = self.screenshot_path.split('/')
        new_name = f'{new_name_list[-2]}_{new_name_list[-1]}'
        if save_path:
            shutil.copy2(self.screenshot_path, os.path.join(save_path, new_name))  

    def _generate_description_zh(self, img):
        # 使用llm对当前页面做出一个描述
        # UI布局和文本内容
        # 布局，包含不同的 UI 组件、文本按钮概念和图标类
        messages = [{
            "role": "system",
            "content": [
                {"type": "text","text": "你是一个GUI AGENT，请用一句话定义所给的手机截图是什么页面。并概括性的描述页面格式、页面作用等关键信息。你需要忽略截图中因时间和更新导致的变化，最终形成简短的中文文字描述。\n"},
            ]
        }]

        messages.append({
            "role": "user",
            "content": [
                {"type": "text","text": f"\n所在应用: {self.app}\n"},
                {"type": "image_url","image_url": {"url": f"data:image/png;base64,{img}"}},
                {"type": "text","text": "\n\n回答："},
            ]
        })

        agent = LLMClient()
        response = agent.get_response_vlm(messages)
        logger.info(f"生成中文的页面描述: {response}")
        return response
    
    def get_screenshot_info(self):
        return {
            'screenshot_path': self.screenshot_path,
            # 'base64_image': self.base64_image,
            'node_description': self.description
        }

    def __repr__(self):
        return f"ScreenShot({self.screenshot_id})"

class UIElementEdge:
    def __init__(self, data=None):
        self.source_node = -1  
        self.target_node = -1  
        if data:
            raw_action = json.loads(data['action'])  
            self.action_type = raw_action['action_type']
            if self.action_type.lower() in ['click', 'long_press']:
                self.action_parameter = {"x": raw_action['x'], "y": raw_action['y']}
            elif self.action_type.lower() == 'swipe':
                self.action_parameter = {"start": raw_action['touch_xy'], "lift": raw_action['lift_xy']}  
            elif self.action_type.lower() in ['type', 'answer',"input_text"]:
                self.action_parameter = {"text": raw_action['text']}
            elif self.action_type.lower() == 'open':
                self.action_parameter = {"text": raw_action['app']}
            elif self.action_type.lower() == 'system_button':
                self.action_parameter = {"text": raw_action['button']}  # home & back
            elif self.action_type.lower() in ['wait','complete']:
                self.action_parameter = {}
            elif self.action_type.lower() == 'status':
                self.action_parameter = {"status": raw_action['goal_status']}
            else:
                self.action_parameter = {}
        else:
            self.action_type = None
            self.action_parameter = {}

        self.action_box = []  


    def get_edge_info(self):
        edge_info = {
            'source_node': self.source_node,
            'target_node': self.target_node,
            'action_type': self.action_type,
            'action_parameter': self.action_parameter,
            # 'action_box': self.action_box,
            # 'action_description': self.action_description,
        }
        return edge_info

    def load_edge_info(self, edge_info):
        self.source_node = edge_info['source_node']
        self.target_node = edge_info['target_node']
        self.action_type = edge_info['action_type']
        self.action_parameter = edge_info['action_parameter']
        self.action_box = []
        # self.action_description = edge_info['action_description']
    
    def __eq__(self, other):
        if not isinstance(other, UIElementEdge):
            return False
        return (
            self.source_node == other.source_node and
            self.target_node == other.target_node and
            self.action_type == other.action_type and
            self.action_parameter == other.action_parameter and
            self.action_box == other.action_box
        )

    def set_source_node(self, node_id):
        self.source_node = node_id

    def set_target_node(self, node_id):
        self.target_node = node_id

    def __repr__(self):
        return f"UINode({self.ui_id}, {self.content})"


class Graph:
    def __init__(self, max_nodes=1000, app=None):
        self.nodes = {}  
        # self.max_nodes = max_nodes
        self.next_id = 0  
        self.home_id = 0  # default home node id 0
        self.app = app

    def add_node(self, tmpnode, new_trajectory, last_node, last_edge):
        newnode = copy.deepcopy(tmpnode)  
        newnode.set_nodeid(self.next_id)  
        save_dir = os.path.join(nodes_save_path, self.app, f'node{newnode.node_id}')  
        os.makedirs(save_dir, exist_ok=True)
        newnode.screenlists[0].save_screenshot(save_path = save_dir)

        self.nodes[self.next_id] = newnode 
        self.add_home_edge(newnode.node_id)  
        self.next_id += 1

        if not new_trajectory:
            if last_node != -1:
                self.nodes[last_node].ui_element_edge_list[last_edge].set_target_node(newnode.node_id)  
                self.nodes[last_node].next_node_id_list.add(newnode.node_id)  
                if self.nodes[last_node].ui_element_edge_list.count(self.nodes[last_node].ui_element_edge_list[last_edge]) > 1:
                    logger.info(f"Edge from node {last_node} already exists, not adding duplicate.")
                    del self.nodes[last_node].ui_element_edge_list[last_edge]
                if newnode.node_id!=last_node:
                    self.add_back_edge(source_id=newnode.node_id, target_id=last_node)  # 增加回边
                else:
                    pass

        last_edge = 0  
        last_node = newnode.node_id  
        return last_edge, last_node

    def merge_to_node(self, node, merge_target_node_id, new_trajectory, last_node, last_edge):
        save_dir = os.path.join(nodes_save_path, self.app, f'node{merge_target_node_id}')  
        node.screenlists[0].save_screenshot(save_path = save_dir)

        merge_target_node = self.nodes[merge_target_node_id]
        if not new_trajectory:
            if last_node != -1:
                self.nodes[last_node].ui_element_edge_list[last_edge].set_target_node(merge_target_node_id) 
                self.nodes[last_node].next_node_id_list.add(merge_target_node_id) 
                if self.nodes[last_node].ui_element_edge_list.count(self.nodes[last_node].ui_element_edge_list[last_edge]) > 1:
                    logger.info(f"Edge from node {last_node} to {merge_target_node_id} already exists, not adding duplicate.")
                    del self.nodes[last_node].ui_element_edge_list[last_edge]
                if merge_target_node_id != last_node:
                    flag = self.add_back_edge(source_id=merge_target_node_id, target_id=last_node)
                else:
                    pass

        for screenshot in node.screenlists:
            self.nodes[merge_target_node_id].screenlists.append(screenshot) 
        for uielement in node.ui_element_edge_list:
            uielement.set_source_node(merge_target_node_id)
            if uielement not in self.nodes[merge_target_node_id].ui_element_edge_list:
                self.nodes[merge_target_node_id].ui_element_edge_list.append(uielement) 
            
        last_edge = len(self.nodes[merge_target_node_id].ui_element_edge_list) - 1 
        last_node = merge_target_node.node_id 
        return last_edge, last_node
        

    def find_similar_node(self, new_node: ScreenNode, threshold=0.8):
        all_similarity = {}
        for node_id, node in self.nodes.items():
            # if node_id == 0 or node_id == 1: 
            #     continue
            similarity = node.calculate_similarity(new_node, threshold)
            logger.info(f"Average Total similarity with node{node.node_id}: {similarity}")
            if similarity > threshold:
                all_similarity[node_id] = similarity
        if all_similarity:
            max_similarity = max(all_similarity.values())
            similar_node_ids = [node_id for node_id, sim in all_similarity.items() if sim == max_similarity]
            if len(similar_node_ids) == 1:
                return similar_node_ids[0]
            else:
                # randomly return one of the similar nodes
                result = random.choice(similar_node_ids)
                logger.warning(f"Multiple similar nodes found with ID {similar_node_ids} with similarity {max_similarity}, randomly returning {result}.")
                return result
        else:
            return -1

                
    def update(self, data:dict, new_trajectory:bool, last_node:int, last_edge:int, threshold=0.8, step=None):
        """
        data:轨迹history的一个元素        
        """
        try:
            # construct new object
            new_tmp_node = ScreenNode(data, app=self.app)

            # 如果你的轨迹起点相同，直接合并到同一节点或者选择跳过
            # if new_trajectory and len(self.nodes)!=0:
            #     last_edge, last_node = self.merge_to_node(new_tmp_node, 0, new_trajectory, last_node, last_edge)
            #     return last_edge, last_node
            
            # 如果你的轨迹第二个点都是同一个APP，也可以选择直接合并或者选择跳过
            # if step==1:
            #     last_edge, last_node = self.merge_to_node(new_tmp_node, 1, new_trajectory, last_node, last_edge)
            #     return last_edge, last_node

  
            similar_node_id = self.find_similar_node(new_tmp_node, threshold) 
            if similar_node_id != -1:
                # 合并到相似的节点下面
                last_edge, last_node = self.merge_to_node(new_tmp_node, similar_node_id, new_trajectory, last_node, last_edge)
                logger.info(f"Merge to node with ID {similar_node_id}")
            else:
                last_edge, last_node = self.add_node(new_tmp_node, new_trajectory, last_node, last_edge)
                logger.info(f"Added new node with ID {self.next_id-1}")

            # 额外还需要的东西有：
            return last_edge, last_node
        except Exception as e:
            logger.error(f"Graph update失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise

    def add_back_edge(self, source_id, target_id):
        back_edge = UIElementEdge({
            'action': json.dumps({'action_type': 'system_button', 'button': 'back'}),
        })
        back_edge.set_source_node(source_id)  
        back_edge.set_target_node(target_id)  
        if back_edge not in self.nodes[source_id].ui_element_edge_list:
            self.nodes[source_id].ui_element_edge_list.append(back_edge)  
            self.nodes[source_id].next_node_id_list.add(target_id)  
            return True
        return False

    def add_home_edge(self, source_id):
        home_edge = UIElementEdge({
            'action': json.dumps({'action_type': 'system_button', 'button': 'home'}),
        })
        home_edge.set_source_node(source_id)
        home_edge.set_target_node(self.home_id)
        if home_edge not in self.nodes[source_id].ui_element_edge_list:
            self.nodes[source_id].ui_element_edge_list.append(home_edge) 
            self.nodes[source_id].next_node_id_list.add(self.home_id)  
            return True 
        return False

    def get_node(self, page_id):
        return self.nodes.get(page_id)

    def save_graph(self, save_path):
        """图保存功能"""
        graph_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'num_nodes': len(self.nodes),
                'app': self.app,
                # 'max_nodes': self.max_nodes,  
                'next_id': self.next_id
            },
            'nodes': {node_id: node.get_node_info() for node_id, node in self.nodes.items()}
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=4, ensure_ascii=False, default=str)
        
    def load_graph(self, load_path):
        """从文件加载图结构"""
        try:
            if load_path.endswith('.gz'):
                with gzip.open(load_path, 'rt', encoding='utf-8') as f:
                    graph_data = json.load(f)
            else:
                with open(load_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
            
            self.nodes = {}

            # rebuild nodes
            metadata = graph_data.get('metadata', {})
            nodes = graph_data.get('nodes', {})

            for node_id, node_info in nodes.items():
                # rebuild ScreenNode
                node = ScreenNode(node_id=int(node_id), app=node_info.get('app', None))

                # rebuild screenshots list
                for screenshot_info in node_info.get('screenlists', []):
                    screenshot = ScreenShot(image_path = screenshot_info['screenshot_path'], description=screenshot_info.get('node_description', None))
                    node.screenlists.append(screenshot)
                
                # rebuild UI element edges
                for edge_info in node_info.get('ui_element_edge_list', []):
                    edge = UIElementEdge()
                    edge.load_edge_info(edge_info)
                    node.ui_element_edge_list.append(edge)

                # transform list to set
                node.next_node_id_list = set(node_info.get('next_node_id_list', []))

                self.nodes[int(node_id)] = node

                logger.info(f"Load node {node_id}!")
            
            if metadata:
                self.next_id = metadata.get('next_id', len(self.nodes))
                self.app = metadata.get('app', len(self.nodes))
            else:
                self.next_id = len(self.nodes)
            
            logger.info(f"Graph loaded from {load_path} with {len(self.nodes)} nodes.")
            
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise

    def __repr__(self):
        return f"Graph({len(self.nodes)} nodes, {len(self.edges)} edges)"
        

if __name__ == "__main__":
    import time
    agent = LLMClient()
    with open('/home/notebook/code/personal/S9060045/demonstration_based_learning/data/examples/myphoto/1aa2399eb9296d74f3f0ba34f704a900.jpg', "rb") as img_file:
        a_img = base64.b64encode(img_file.read()).decode('utf-8')
    with open('/home/notebook/code/personal/S9060045/demonstration_based_learning/data/examples/myphoto/398b474e729282fbb6fd5097e03a0a81.jpg', "rb") as img_file:
        b_img = base64.b64encode(img_file.read()).decode('utf-8')
    logger.info("图片已加载，开始生成描述...")

    start_time = time.time()
    messages = [{
        "role": "system",
        "content": [
            {"type": "text","text": "你是一个GUI AGENT。请忽略截图中因时间和更新导致的变化，从页面格式、页面作用等方面，判断所给的两张图片是否属于同一个页面。\n"},
        ]
    }]
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url","image_url": {"url":f"data:image/png;base64,{a_img}"}},
            {"type": "image_url","image_url": {"url":f"data:image/png;base64,{b_img}"}},
            {"type": "text","text": "\n\n回答：是/否\n原因："},
        ]
    })
    description = agent.get_response_vlm(messages, temperature=0.0)
    logger.info(description)
    end_time = time.time()
    logger.info(f"VLM判断是否同页面时间: {end_time - start_time} seconds")