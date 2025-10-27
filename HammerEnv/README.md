# HammerEnv - Android Device Dynamic Interaction Environment

## Overview

HammerEnv is a comprehensive Android device interaction environment that enables dynamic exploration and automation of mobile applications. It provides a robust framework for BFS (Breadth-First Search) based app exploration, AI-powered UI element analysis, and trajectory generation.

## Features

- **Dynamic Device Interaction**: Support for both physical devices and cloud-based Android emulators
- **BFS App Exploration**: Automated breadth-first search exploration of mobile applications
- **AI-Powered Analysis**: Intelligent UI element detection and interaction using advanced AI models
- **Trajectory Generation**: Complete action trajectory recording with screenshots and metadata
- **Environment Reset**: Fresh environment initialization for each task to ensure clean state
- **Multi-Device Support**: Concurrent support for multiple Android devices

## Prerequisites

### Required Dependencies

Install the following open-source projects in editable mode:

```bash
# Android Environment
pip install -e https://github.com/google-deepmind/android_env.git

# Android World
pip install -e https://github.com/google-research/android_world.git
```

### Device Setup

HammerEnv supports both physical devices and cloud-based Android emulators:

- **Physical Devices**: Connect via ADB (Android Debug Bridge)
- **Cloud Emulators**: Configure cloud-based Android instances

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd HammerEnv
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://your-api-endpoint/v1"
```

## Quick Start

### 1. Start the Interaction Environment Server

```bash
python src/server/gradio_web_server_physical_device.py
```

### 2. Run BFS App Explorer

```bash
python examples/bfs_app_explorer_fixed.py
```

## Configuration

### ExplorationConfig Parameters

- `max_depth`: Maximum exploration depth (default: 3)
- `max_trajectories`: Maximum number of trajectories to generate (default: 50)
- `app_name`: Target application name (default: "小红书")
- `output_dir`: Output directory for trajectories (default: "trajectories")
- `delay_between_actions`: Delay between actions in seconds (default: 2.0)
- `model_name`: AI model name for analysis (default: "Qwen2.5-VL-72B-Instruct")
- `reset_environment_per_task`: Reset environment after each task (default: True)
- `reset_delay`: Delay for environment reset in seconds (default: 1.0)

### Command Line Arguments

```bash
python examples/bfs_app_explorer_fixed.py \
    --server-name "http://localhost:7880/" \
    --model-name "Qwen2.5-VL-72B-Instruct" \
    --app-name "小红书" \
    --max-depth 3 \
    --max-trajectories 20 \
    --output-dir "trajectories" \
    --delay 2.0
```

## Output Structure

The system generates organized trajectory data with the following structure:

```
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

### JSON Trajectory Format

Each trajectory JSON file contains an array of steps:

```json
[
  {
    "action": "{\"action_type\": \"open\", \"app\": \"小红书\"}",
    "action_description_v1": "用户要求打开小红书应用，当前屏幕显示的是手机主屏幕，因此需要使用open命令来打开指定的应用。",
    "screenshot": "Screenshot_2025-01-10-20-10-21_0.jpg",
    "ui_elements": []
  },
  {
    "action": "{\"action_type\": \"complete\", \"status\": \"success\"}",
    "action_description_v1": "任务完成，所有操作已成功执行。",
    "screenshot": "Screenshot_2025-01-10-20-10-21_1.jpg",
    "ui_elements": []
  }
]
```

## Architecture

### Core Components

1. **BFSEplorer**: Main exploration engine implementing breadth-first search
2. **HammerEnvClient**: Device interaction client for Android devices
3. **QwenOperator**: AI-powered operator for UI analysis and action generation
4. **TrajectoryNode**: Data structure representing exploration states
5. **ClickableElement**: UI element representation with coordinates and metadata

### Key Features

- **Environment Reset**: Automatic environment reinitialization between tasks
- **AI Analysis**: Intelligent UI element detection using advanced language models
- **Screenshot Management**: Automatic screenshot capture and storage
- **Error Handling**: Robust error handling with fallback mechanisms
- **JSON Repair**: Advanced JSON parsing with multiple repair strategies