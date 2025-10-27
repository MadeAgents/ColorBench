# BFS应用探索器使用指南

## 🎯 功能简介

BFS应用探索器是一个基于广度优先搜索的Android应用自动化探索工具，能够系统性地探索应用的各个页面，并生成完整的轨迹记录。

## 🚀 快速启动

### 方法1: 使用启动脚本（推荐）

```bash
# 基本使用
python examples/start_bfs_explorer.py

# 自定义参数
python examples/start_bfs_explorer.py \
    --app-name "小红书" \
    --max-depth 3 \
    --max-trajectories 50 \
    --port 7880
```

### 方法2: 手动启动

#### 步骤1: 启动HammerEnv服务器
```bash
cd D:\Users\80398388\files\mu\HammerEnv4\HammerEnv
python src/server/gradio_web_server_physical_device.py --server-port 7880
```

#### 步骤2: 启动BFS探索器
```bash
python examples/bfs_app_explorer.py \
    --server-name "http://localhost:7880/" \
    --app-name "小红书" \
    --max-depth 3 \
    --max-trajectories 50
```

## 📋 参数说明

### 启动脚本参数
- `--port`: 服务器端口（默认7880）
- `--skip-server`: 跳过服务器启动
- `--app-name`: 目标应用名称（默认"小红书"）
- `--max-depth`: 最大探索深度（默认3）
- `--max-trajectories`: 最大轨迹数量（默认50）
- `--output-dir`: 输出目录（默认"trajectories"）
- `--delay`: 操作间延迟秒数（默认2.0）

### BFS探索器参数
- `--server-name`: HammerEnv服务器地址
- `--model-name`: AI模型名称
- `--app-name`: 目标应用名称
- `--max-depth`: 最大探索深度
- `--max-trajectories`: 最大轨迹数量
- `--output-dir`: 输出目录
- `--delay`: 操作间延迟

## 📊 输出结果

### 轨迹文件
```
trajectories/
├── 首页.json
├── 我.json
├── 设置.json
├── 账号与安全.json
├── 通用设置.json
├── 通知设置.json
├── 隐私设置.json
└── exploration_report.json
```

### 轨迹文件内容示例
```json
{
  "trajectory": ["打开小红书", "点击'我'", "点击设置图标", "账号与安全"],
  "current_page": "小红书首页 -> 我 -> 设置图标 -> 账号与安全",
  "clickable_elements": [
    {
      "element_name": "密码管理",
      "element_type": "button",
      "coordinates": [100, 200],
      "description": "密码管理按钮",
      "confidence": 0.9
    }
  ],
  "depth": 4,
  "status": "completed",
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🔧 工作原理

1. **初始化**: 连接Android设备，创建初始轨迹节点
2. **BFS循环**: 从队列取出节点，探索页面，生成新节点
3. **AI分析**: 使用AI识别页面可点击元素
4. **轨迹执行**: 执行操作到达目标页面
5. **文件保存**: 保存轨迹和页面信息
6. **报告生成**: 生成完整的探索报告

## ⚠️ 注意事项

1. **设备连接**: 确保Android设备已连接并开启USB调试
2. **应用安装**: 确保目标应用已安装在设备上
3. **网络连接**: 确保AI模型服务可访问
4. **存储空间**: 确保有足够的存储空间保存轨迹文件
5. **执行时间**: 深度探索可能需要较长时间

## 🐛 故障排除

### 常见问题

1. **设备连接失败**
   - 检查USB连接
   - 确认USB调试已开启
   - 检查ADB驱动

2. **服务器启动失败**
   - 检查端口是否被占用
   - 确认Python环境正确
   - 检查依赖包是否安装

3. **AI分析失败**
   - 检查网络连接
   - 确认AI服务可访问
   - 检查API密钥配置

4. **轨迹执行失败**
   - 检查应用是否正常运行
   - 确认页面元素识别正确
   - 调整操作延迟时间

## 📈 性能优化

1. **调整深度**: 根据需求调整最大探索深度
2. **限制轨迹数**: 设置合理的最大轨迹数量
3. **优化延迟**: 根据设备性能调整操作延迟
4. **并行处理**: 使用多线程提高效率

## 🔄 扩展功能

1. **自定义应用**: 修改目标应用名称
2. **添加过滤**: 过滤不需要的页面
3. **增强分析**: 改进AI分析算法
4. **批量处理**: 支持多设备并行探索
