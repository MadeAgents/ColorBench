<!-- Language Switcher -->
<p align="center">
  <a href="#english-version">English</a> | <a href="#简体中文">简体中文</a>
</p>

---

# <a id="english-version"></a>🎨 ColorBench: Benchmarking Mobile Agents with Graph-Structured Framework for Complex Long-Horizon Tasks

👋 Welcome to the **ColorBench** repository — a **graph-structured benchmark** designed to evaluate mobile GUI agents on complex, long-horizon tasks composed of multiple atomic operations. This project provides:
- A **graph-based benchmark construction methodology** to expand or reconstruct environments.
- A **plug-and-play evaluation framework** for safe, reproducible testing.

![ColorBench](assets/colorbench.jpg)

---

## 📢 News
- **[xx Oct '25]** Released the core code and dataset (including evaluation environment and benchmark graphs).
- **[16 Oct '25]** Our paper [*ColorBench: Benchmarking Mobile Agents with Graph Structured Framework for Complex Long-Horizon Task*](https://arxiv.org/abs/2510.14621) is now available on arXiv!

---

## 🧭 Overview

![ColorBench](assets/graph.png)

### 📦 175 Complex Long-Horizon Tasks
- 🌐 Covering **21 major apps** – WeChat, Meituan, JD, Xiaohongshu, etc.
- 🔄 **101 cross-app** and **74 single-app** tasks
- 🧭 Average optimal path length >13 steps

### 🎨 Graph-Based Design & Multi-Path Evaluation
- 🔀 Multiple correct and error paths supported
- 🔁 Enables **reflection**, **replanning**, and **backtracking** behaviors

### 📊 Comprehensive Evaluation Metrics
- ✅ 3 Core Indicators: **Success Rate (SR)**, **Completion Rate (CR)**, **Atomic Capability (AC)**
- 🧩 15 Atomic Capabilities – e.g., Search, Filter, Save, Share, Memory
- 🎯 Fine-grained diagnostics for weak atomic capabilities

### 🤖 Plug-and-Play Evaluation Framework
- 📱 Static but interactive graph environment
- 📐 Safe and repeatable testing without real devices or accounts
- 🧰 Fully automated evaluation – no human verification required

![ColorBench](assets/benchmark_comparison.png)
![ColorBench](assets/main_result.png)

---

## 📂 Repository Structure

```plaintext
ColorBench/
├── config/
│   ├── default.yaml                # Config for evaluating agents
│   └── customized_config...
├── data/
│   ├── graph.json                  # Graph structure
│   ├── task.json                   # Task details
│   ├── graph_image/                # Screenshots
│   │   ├── Screenshot0.png
│   │   ├── Screenshot1.jpg
│   └── ...
├── HammerEnv/                      # BFS-based trajectory collection
├── src/
│   ├── agent/                      # Evaluation agents
│   ├── graph_construction/         # Graph construction utilities
│   ├── test/                       # Evaluation scripts
│   └── utils.py
├── construct_graph.py
├── run_colorbench_multi_agent.py
├── run_colorbench.py
└── README.md
```

---

## 🚀 Quick Start

### 🛠️ Installation
```bash
git clone https://github.com/MadeAgents/ColorBench
cd ColorBench
pip install -r requirements.txt
```

### 🧪 Evaluation
```bash
python3 run_colorbench.py --config configs/default.yaml --model your_model_name
```
Alternatively, use the provided script:
```bash
bash run_colorbench.sh
```

#### Customize Your Agent
Define your agent in `src/agent/agent_base.py` by inheriting from **AgentBase** and implementing the `agent_step` function (responsible for executing actions and logging). Then, add your agent to `run_colorbench.py` and create a new config file under `./config/`.

Evaluation results are saved under `./checkpoints/`.

### 🧩 Graph-Structured Benchmark Construction

#### 🔍 Breadth-First Search (BFS) Application Exploration

We use our self-developed Android device interaction environment **HammerEnv** for breadth-first application exploration. HammerEnv is a comprehensive Android device interaction environment that enables dynamic exploration and automated operations of mobile applications.

#### 🛠️ Installation Steps

1. **Download and install android_env and android_world open-source projects**:

https://github.com/google-deepmind/android_env
https://github.com/google-research/android_world

Note: When installing via pip, you need to use the editable mode with the command: pip install -e .

2. **Configure ADB connection**:
Refer to https://developer.android.com/tools

3. **Set environment variables**:
```bash
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://xxx.xxx.xxx.xxx/v1"
```

4. **Start interaction environment server**:
```bash
python HammerEnv/src/server/gradio_web_server_physical_device.py
```

5. **Run BFS application explorer**:
```bash
python HammerEnv/examples/bfs_app_explorer_fixed.py
```

#### ⚙️ Configuration

##### Exploration Configuration Parameters
| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `max_depth` | Maximum exploration depth | 3 |
| `max_trajectories` | Maximum number of trajectories to generate | 50 |
| `app_name` | Target application name | "小红书" |
| `output_dir` | Trajectory output directory | "trajectories" |
| `delay_between_actions` | Delay between actions (seconds) | 2.0 |
| `model_name` | AI model name for analysis | "Qwen2.5-VL-72B-Instruct" |
| `reset_environment_per_task` | Reset environment after each task | True |
| `reset_delay` | Environment reset delay (seconds) | 1.0 |

##### Command Line Parameters
```bash
python examples/bfs_app_explorer_fixed.py \
    --server-name "http://localhost:7880/" \
    --model-name "xxx" \
    --app-name "小红书" \
    --max-depth 3 \
    --max-trajectories 20 \
    --output-dir "trajectories" \
    --delay 2.0
```

#### 🔍 Depth-First Search (DFS) Application Exploration

To capture user long-horizon tasks, we manually capture sequences of mobile operation screenshots using a depth-first approach, then generate structured trajectory data through AI model analysis.

##### Workflow
1. **Screenshot Collection**: Manually capture application operation screenshots in order
2. **Trajectory Analysis**: Use large models to analyze adjacent screenshot pairs
3. **Action Recognition**: Extract precise click coordinates, input text, and other operations
4. **Trajectory Generation**: Build trajectory files based on trajectory data

##### Usage
```bash
# Run depth-first trajectory generation
python src/graph_construction/pic2trajectory.py
```

##### Input Requirements
- **Directory Structure**: `dfs/pic/trajectory1/`
- **Required Files**: `query.txt` (task description) + `Screenshot_step_*_raw.{png|jpg}`
- **Naming Convention**: Screenshot files numbered in operation order (trajectory1 represents the first trajectory)

##### Output Results
- **Trajectory File**: `dfs/trajectory/trajectory1/trajectory_v0.txt`
- **Adjacency Matrix**: `dfs/trajectory/trajectory1/{query}.csv`

#### 📁 Output Structure

The system generates well-organized trajectory data with the following structure:

```plaintext
trajectories/
├── 小红书/
│   ├── 小红书.json
│   ├── Screenshot_2025-01-10-20-10-21_0.jpg
│   ├── Screenshot_2025-01-10-20-10-21_1.jpg
│   └── Screenshot_2025-01-10-20-10-21_2.jpg
└── 搜索/
    ├── 搜索.json
    ├── Screenshot_2025-01-10-20-15-30_0.jpg
    └── Screenshot_2025-01-10-20-15-30_1.jpg
```

---

## 📄 File Format Descriptions

### Trajectory File (JSON)
```json
{
  "task": "Task description",
  "trajectory": [
    {
      "action": {"type": "click", "x": 100, "y": 200},
      "screenshot": "base64-encoded screenshot",
      "ui_elements": [
        {"name": "button", "type": "button", "bounds": [0, 0, 100, 50]}
      ]
    }
  ]
}
```

### Adjacency Matrix (CSV)
```csv
,node1,node2,node3
node1,0,{action1,action2},0
node2,0,0,{action1,action2}
node3,{action1,action2},0,0
```

---

## 🤝 Contributing & Citation

Contributions via **Issues** or **Pull Requests** are welcome!
If you use this project, please consider citing our paper:

> **ColorBench: Benchmarking Mobile Agents with Graph Structured Framework for Complex Long-Horizon Task**  
> [arXiv:2510.14621](https://arxiv.org/abs/2510.14621)

📚 Dataset available at: [HuggingFace Dataset (Placeholder)](https://huggingface.co/datasets/ColorBench)

---

# <a id="简体中文"></a>🎨 ColorBench：基于图结构的复杂长程任务移动智能体评测基准

👋 欢迎访问 **ColorBench** 仓库 —— 一个面向移动端 GUI 智能体的 **图结构评测基准**，用于评估智能体在复杂长程任务及其原子子任务上的性能。项目主要特点包括：
- 提供一套 **图结构评测构建方法学**，便于扩展或重建任务图；
- 提供一个 **可插拔式评测框架**，实现安全、可复现的智能体测试。

![ColorBench](assets/colorbench.jpg)

---

## 📢 最新动态
- **[xx Oct '25]** 已开放核心代码与数据集（含评测环境及任务图）。
- **[16 Oct '25]** 我们的论文 [*ColorBench: Benchmarking Mobile Agents with Graph Structured Framework for Complex Long-Horizon Task*](https://arxiv.org/abs/2510.14621) 已发布在 arXiv！

---

## 🧭 概览

![ColorBench](assets/graph.png)

### 📦 175 个复杂长程任务
- 🌐 覆盖 **21 个主流应用** —— 微信、美团、京东、小红书等；
- 🔄 包含 **101 个跨应用任务** 与 **74 个单应用任务**；
- 🧭 平均最优路径长度超过 13 步。

### 🎨 图结构设计与多路径评测
- 🔀 包含多条正确路径与典型错误路径；
- 🔁 支持智能体的 **反思**、**重规划** 与 **回溯** 行为。

### 📊 全面评测指标
- ✅ 三大核心指标：**成功率（SR）**、**完成率（CR）**、**原子能力（AC）**；
- 🧩 15 类原子能力 —— 如搜索、筛选、保存、分享、记忆等；
- 🎯 可精细诊断智能体在原子层级的薄弱环节。

### 🤖 可插拔式评测框架
- 📱 静态但可交互的图环境；
- 📐 无需真实设备或账户即可安全复现；
- 🧰 全自动化评测，无需人工验证。

![ColorBench](assets/benchmark_comparison.png)
![ColorBench](assets/main_result.png)
---

## 📂 项目结构

```plaintext
ColorBench/
├── config/
│   ├── default.yaml                # 智能体评测配置
│   └── customized_config...
├── data/
│   ├── graph.json                  # 图结构信息
│   ├── task.json                   # 任务信息
│   ├── graph_image/                # 截图数据
│   │   ├── Screenshot0.png
│   │   ├── Screenshot1.jpg
│   └── ...
├── HammerEnv/                      # 基于广度优先的轨迹采集环境
├── src/
│   ├── agent/                      # 智能体定义与评测逻辑
│   ├── graph_construction/         # 图构建模块
│   ├── test/                       # 评测脚本
│   └── utils.py
├── construct_graph.py
├── run_colorbench_multi_agent.py
├── run_colorbench.py
└── README.md
```

---

## 🚀 快速开始

### 🛠️ 安装依赖
```bash
git clone https://github.com/MadeAgents/ColorBench
cd ColorBench
pip install -r requirements.txt
```

### 🧪 运行评测
```bash
python3 run_colorbench.py --config configs/default.yaml --model your_model_name
```
或使用脚本：
```bash
bash run_colorbench.sh
```

#### 自定义智能体
在 `src/agent/agent_base.py` 中继承 **AgentBase** 类并实现 `agent_step` 函数，用于执行动作与记录日志。
随后在 `run_colorbench.py` 中注册该智能体，并可在 `./config/` 下添加新的评测配置文件。

评测结果将保存在 `./checkpoints/` 目录下。

### 🧩 图结构评测构建

#### 🔍 基于广度优先的应用探索

我们使用自开发的安卓设备交互环境 **HammerEnv** 进行基于广度优先的应用探索。HammerEnv 是一个全面的安卓设备交互环境，可实现移动应用的动态探索和自动化操作。

#### 📋 安装步骤


1) **下载并安装 android_env、android_world 两个开源项目**:

https://github.com/google-deepmind/android_env
https://github.com/google-research/android_world
注：pip 安装时需要使用编辑模式 pip install -e . 

2) **配置ADB连接**:
参见https://developer.android.com/tools

3) **设置环境变量**：
```bash
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://xxx.xxx.xxx.xxx/v1""
```

4. **启动交互环境服务器**：
```bash
python HammerEnv/src/server/gradio_web_server_physical_device.py
```

5. **运行 BFS 应用探索器**：
```bash
python HammerEnv/examples/bfs_app_explorer_fixed.py
```

#### ⚙️ 配置

##### 探索配置参数
| 参数 | 描述 | 默认值 |
|------|------|--------|
| `max_depth` | 最大探索深度 | 3 |
| `max_trajectories` | 要生成的最大轨迹数 | 50 |
| `app_name` | 目标应用名称 | "小红书" |
| `output_dir` | 轨迹输出目录 | "trajectories" |
| `delay_between_actions` | 动作之间的延迟（秒） | 2.0 |
| `model_name` | 用于分析的 AI 模型名称 | "Qwen2.5-VL-72B-Instruct" |
| `reset_environment_per_task` | 每个任务后重置环境 | True |
| `reset_delay` | 环境重置延迟（秒） | 1.0 |

##### 命令行参数
```bash
python examples/bfs_app_explorer_fixed.py \
    --server-name "http://localhost:7880/" \
    --model-name "xxx" \
    --app-name "小红书" \
    --max-depth 3 \
    --max-trajectories 20 \
    --output-dir "trajectories" \
    --delay 2.0
```
#### 🔍 基于深度优先的应用探索

为了捕捉用户长程任务，我们采用深度优先的方式手动截取操作手机的截图序列，然后通过 AI 模型分析生成结构化轨迹数据。

##### 工作流程
1. **截图收集**：按操作顺序手动截取应用操作截图
2. **轨迹分析**：使用大模型分析相邻截图对
3. **动作识别**：提取精确的点击坐标、输入文本等操作
4. **轨迹生成**：基于轨迹数据构建轨迹文件

##### 使用方法
```bash
# 运行深度优先轨迹生成
python src/graph_construction/pic2trajectory.py
```

##### 输入要求示例如下
- **目录结构**：`dfs/pic/trajectory1/`
- **必需文件**：`query.txt`（任务描述）+ `Screenshot_step_*_raw.{png|jpg}`
- **命名规范**：按操作顺序编号的截图文件trajectory1代表第一条轨迹

##### 输出结果
- **轨迹文件**：`dfs/trajectory/trajectory1/trajectory_v0.txt`
- **邻接矩阵**：`dfs/trajectory/trajectory1/{query}.csv`



#### 📁 输出结构

系统生成组织有序的轨迹数据，结构如下：

```plaintext
trajectories/
├── 小红书/
│   ├── 小红书.json
│   ├── Screenshot_2025-01-10-20-10-21_0.jpg
│   ├── Screenshot_2025-01-10-20-10-21_1.jpg
│   └── Screenshot_2025-01-10-20-10-21_2.jpg
└── 搜索/
    ├── 搜索.json
    ├── Screenshot_2025-01-10-20-15-30_0.jpg
    └── Screenshot_2025-01-10-20-15-30_1.jpg
```

---

## 📄 文件格式说明

### 轨迹文件（JSON）
```json
{
  "task": "任务描述",
  "trajectory": [
    {
      "action": {"type": "click", "x": 100, "y": 200},
      "screenshot": "base64编码截图",
      "ui_elements": [
        {"name": "按钮", "type": "button", "bounds": [0, 0, 100, 50]}
      ]
    }
  ]
}
```

### 邻接矩阵（CSV）
```csv
,node1,node2,node3
node1,0,{action1,action2},0
node2,0,0,{action1,action2}
node3,{action1,action2},0,0
```

---

## 🤝 贡献与引用

欢迎通过 **Issue** 或 **Pull Request** 改进本项目。
如使用本项目，请引用以下论文：

> **ColorBench: Benchmarking Mobile Agents with Graph Structured Framework for Complex Long-Horizon Task**  
> [arXiv:2510.14621](https://arxiv.org/abs/2510.14621)

📚 数据集链接：[HuggingFace 数据集（占位符）](https://huggingface.co/datasets/ColorBench)

---

<p align="center">⭐ If you find this project helpful, please give us a star on GitHub!</p>

