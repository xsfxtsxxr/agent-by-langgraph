import sqlite3
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

# 1. Define tools and model
model = init_chat_model("deepseek-v4-flash", temperature=0)


# ==========================================
# 1. 定义状态（这就是 Checkpointer 要存的东西）
# ==========================================
class TaskState(MessagesState):
    """继承 MessagesState 自带 messages 列表，再加一个任务进度字段"""

    task_stage: str  # 记录当前任务执行到哪一步


# ==========================================
# 2. 定义图的节点（模拟一个多步 Agent 任务）
# ==========================================
def planner_node(state: TaskState):
    """第一步：拆解任务"""
    print("  [节点] planner: 正在拆解任务...")
    response = model.invoke(
        [
            {
                "role": "system",
                "content": "你是一个任务规划助手。用一句话说明你要做什么。",
            },
            *state["messages"],
        ]
    )
    return {"messages": [response], "task_stage": "规划完成，等待执行"}


def executor_node(state: TaskState):
    """第二步：执行子任务（模拟工具调用）"""
    print("  [节点] executor: 正在执行数据查询...")
    # 模拟工具返回
    tool_result = "查询到 Q3 销售额为 1,234,567 元，环比增长 15%"
    return {
        "messages": [AIMessage(content=f"执行完毕。{tool_result}")],
        "task_stage": "执行完成，等待总结",
    }


def summarizer_node(state: TaskState):
    """第三步：生成总结"""
    print("  [节点] summarizer: 正在生成总结...")
    response = model.invoke(
        [
            {"role": "system", "content": "根据前面的对话，用简洁的语言总结任务结果。"},
            *state["messages"],
        ]
    )
    return {"messages": [response], "task_stage": "全部完成 ✅"}


# ==========================================
# 3. 路由逻辑（控制任务流向）
# ==========================================
def route_after_planner(state: TaskState) -> Literal["executor", "summarizer"]:
    """规划完去执行，否则直接总结"""
    if "执行" in state["task_stage"]:
        return "executor"
    return "summarizer"


# ==========================================
# 4. 构建图
# ==========================================
workflow = StateGraph(TaskState)
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("summarizer", summarizer_node)

workflow.add_edge(START, "planner")
workflow.add_conditional_edges("planner", route_after_planner)
workflow.add_edge("executor", "summarizer")
workflow.add_edge("summarizer", END)

# ==========================================
# 5. 关键：挂上 SQLite Checkpointer
# ==========================================
# 这会在当前目录创建 checkpoints.db 文件
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# 编译时注入 checkpointer
app = workflow.compile(checkpointer=checkpointer)

# ==========================================
# 6. 使用：模拟多轮任务交互
# ==========================================

"""
实际企业级项目常见做法是，每个用户或每个任务都有一个唯一的 thread_id。
from langchain_core.utils.uuid import uuid7
thread_id = str(uuid7())

config = {
    "configurable": {
        "thread_id": thread_id,     # 会话标识（可恢复历史）
        "user_id": user_id,         # 用户标识（权限、审计、个性化）
    }
}
"""

THREAD_ID = "sales_analysis_task_001"
config = {"configurable": {"thread_id": THREAD_ID}}


print("=" * 50)
print("第一次调用：启动任务")
print("=" * 50)
result_1 = app.invoke(
    {
        "messages": [HumanMessage(content="帮我分析一下 Q3 的销售数据")],
        "task_stage": "刚启动",
    },
    config=config,
)
print(f"\n📊 当前任务阶段: {result_1['task_stage']}")
print(f"💬 最后回复: {result_1['messages'][-1].content}")
print(f"📦 数据库里已自动存了 {len(result_1['messages'])} 条消息")

print("\n" + "=" * 50)
print("第二次调用：追问（同一个 thread_id，不传历史）")
print("=" * 50)
result_2 = app.invoke(
    {
        "messages": [HumanMessage(content="数据能再细化到每个产品线吗？")],
        # 注意：这里没传 task_stage，Checkpointer 会从数据库恢复之前的状态
    },
    config=config,
)
print(f"\n📊 当前任务阶段: {result_2['task_stage']}")
print(f"💬 最后回复: {result_2['messages'][-1].content}")
print(f"📦 总消息数（含历史）: {len(result_2['messages'])} 条")
