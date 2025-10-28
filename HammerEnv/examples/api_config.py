# Copyright 2025 OPPO

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""
API配置管理
"""

import os
from typing import Dict, Optional

class APIConfig:
    """API配置管理类"""
    
    # 可用的API配置
    CONFIGS = {
        "oppo_demo": {
            "api_key": "EMPTY",
            "base_url": "http://your-api-endpoint/v1",
            "model_name": "Qwen2.5-VL-72B-Instruct"
        }
    }
    
    @classmethod
    def setup_config(cls, config_name: str = "oppo_demo"):
        """设置API配置"""
        if config_name not in cls.CONFIGS:
            raise ValueError(f"未知的配置名称: {config_name}")
        
        config = cls.CONFIGS[config_name]
        
        # 设置环境变量
        os.environ["OPENAI_API_KEY"] = config["api_key"]
        os.environ["OPENAI_BASE_URL"] = config["base_url"]
        
        print(f"API配置已设置: {config_name}")
        print(f"API Key: {config['api_key']}")
        print(f"Base URL: {config['base_url']}")
        print(f"Model: {config['model_name']}")
        
        return config
    
    @classmethod
    def test_connection(cls, config_name: str = "oppo_demo") -> bool:
        """测试API连接"""
        try:
            config = cls.setup_config(config_name)
            
            from hammer_agent.qwen_agent import get_chat_completion
            
            # 创建测试消息
            messages = [{
                "role": "user",
                "content": "Hello, this is a connection test."
            }]
            
            print("测试API连接...")
            response = get_chat_completion(messages=messages, model_id=config["model_name"])
            print(f"API连接成功: {response[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"API连接失败: {e}")
            return False
    
    @classmethod
    def list_configs(cls):
        """列出所有可用配置"""
        print("📋 可用的API配置:")
        for name, config in cls.CONFIGS.items():
            print(f"  {name}: {config['base_url']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        if config_name == "list":
            APIConfig.list_configs()
        else:
            APIConfig.test_connection(config_name)
    else:
        print("用法:")
        print("  python api_config.py list          # 列出所有配置")
        print("  python api_config.py oppo_demo     # 测试oppo_demo配置")
        print("  python api_config.py local         # 测试local配置")
