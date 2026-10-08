# agent-by-langgraph

Learn langgraph

## 环境准备

- poetry.toml 设置 create = true 和 in-project = true

- 创建 conda 环境，例如 `conda create -n project-name-py314 python=3.14`

- poetry init 初始化项目依赖

- poetry env use {project-name-py314}/bin/python，以conda环境中的python解释器为基座，在项目根目录下创建虚拟环境.venv

- poetry install 安装依赖

## pyproject.toml配置注意事项

```toml
requires-python = ">=3.12,<4.0.0"

[[tool.poetry.source]]
name = "aliyun"
url = "https://mirrors.aliyun.com/pypi/simple/"
priority = "primary"
```
