from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()

# 1. Define tools and model
model = init_chat_model("deepseek-v4-flash", temperature=0)


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers."""
    return a / b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


# 单个工具调用级别的重试装饰器（只重试失败的那个工具，不影响同一节点里的其他工具）
import time
from functools import wraps


def retry(max_attempts: int = 3, initial_interval: float = 1.0):
    """在单个工具内部重试。"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"[retry] {func.__name__} 第 {attempt} 次失败: {e}")
                    if attempt == max_attempts:
                        raise
                    time.sleep(initial_interval)

        return wrapper

    return decorator


_weather_attempts = 0
_max_retries = 3


@tool
@retry(max_attempts=_max_retries, initial_interval=1.0)
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    global _weather_attempts
    _weather_attempts += 1

    # 模拟网络波动：前两次抛异常，第三次才成功
    if _weather_attempts < _max_retries:
        raise ConnectionError(f"网络波动，第 {_weather_attempts} 次调用失败")

    # For demonstration purposes, we'll return a static response.
    # In a real implementation, you would call a weather API here.
    return f"The weather in {city} is sunny with a high of 25°C."


tools = [multiply, add, divide, subtract, get_weather]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

# 2. Define state

from typing import Annotated, TypedDict

from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages

# import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int


# 3. Define model node

from langchain.messages import SystemMessage


def llm_call(state: MessagesState) -> MessagesState:
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# 4. Define tool node

from langchain.messages import ToolMessage


def tool_node(state: MessagesState):
    """Performs the tool call"""
    result = []

    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        print(f"[tool_node] 调用工具 {tool_call['name']}，结果: {observation}")
        result.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )
    return {"messages": result}


# Step 5: Define logic to determine whether to end

from typing import Literal

from langgraph.graph import END, START, StateGraph


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    # if the llm makes a tool call, then peformance the tool call, otherwise end the loop
    if last_message and last_message.tool_calls:
        return "tool_node"
    else:
        return END


# Step 6: Build agent


agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# from IPython.display import Image, display

# Show the agent in notebook
# display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# Invoke
from langchain.messages import HumanMessage
from rich import print as rprint

result = {
    "messages": [],
    "llm_calls": 0,
}

while True:
    user_input = input("请输入问题 (或输入 'exit' 退出): ")
    if user_input in ("exit", "quit", "退出"):
        break

    result["messages"].append(HumanMessage(content=user_input))
    # 每次 invoke 都把完整 history 传给模型
    response = agent.invoke({"messages": result["messages"]})
    # 更新为完整历史（包含模型回复和工具消息），供下一轮继续
    result["messages"] = response["messages"]
    # 累计每次 invoke 内的 LLM 调用次数
    result["llm_calls"] += response.get("llm_calls", 0)

for m in result["messages"]:
    m.pretty_print()

print(f"LLM calls: {result['llm_calls']}")

rprint("messages:", result["messages"])
