"""openai 消息格式
```json
{
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "文本内容"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,image_base64"
                    }
                }
            ]
        }
    ]
}
```
"""

ASSISTANT = "assistant"
CONTENT = "content"
DEVICE = "device"
IMAGE_URL = "image_url"
NAME = "name"
ROLE = "role"
USER = "user"
SCREEN = "screen"
TYPE = "type"
TEXT = "text"
URL = "url"
