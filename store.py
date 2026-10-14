from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.memory import InMemoryStore

load_dotenv()

# 1. Define tools and model
model = init_chat_model("deepseek-v4-flash", temperature=0)


# ==========================================
# 1. 初始化长期记忆 Store
# ==========================================
# InMemoryStore：存在进程内存里，重启就丢
# 生产环境换成 PostgresStore，API 一行不改
store = InMemoryStore()


# ==========================================
# 2. 定义状态
# ==========================================
class AgentState(MessagesState):
    """继承 MessagesState 自带 messages 列表，再加一个任务进度字段"""

    user_id: str  # 用 user_id 做长期记忆的隔离键


# ==========================================
# 3. 定义节点
# ==========================================
def load_memory_node(state: AgentState):
    """
    每次对话开始时，从长期记忆加载用户偏好，
    注入到 System Prompt 里，让模型"认识"这个用户
    """
    user_id = state["user_id"]
    namespace = ("prefs", user_id)  # 命名空间：用 tuple 做层级隔离

    stored = store.get(namespace, "preferences")

    if stored and stored.value:
        prefs = stored.value
        print(f"  🧠 [长期记忆] 加载到用户 {user_id} 的偏好: {prefs}")

        system_prompt = "你是一个智能助手。"
        if prefs.get("language"):
            system_prompt += f" 用户偏好使用 {prefs['language']} 编程。"
        if prefs.get("style") == "concise":
            system_prompt += " 回答必须极简，不超过 3 句话。"
        elif prefs.get("style") == "detailed":
            system_prompt += " 回答要详细，带代码示例和原理解释。"

        # 把系统提示插到消息最前面
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        return {"messages": messages}
    else:
        print(f"  🧠 [长期记忆] 用户 {user_id} 暂无历史偏好，使用默认模式")
        return {}


def save_memory_node(state: AgentState):
    """
    每次对话结束后，从用户最新输入中提取偏好，更新长期记忆
    """
    user_id = state["user_id"]
    namespace = ("prefs", user_id)
    last_input = state["messages"][-1].content if state["messages"] else ""

    # 实际生产中这里应该用 LLM 做结构化提取
    # 这里用简单规则演示
    new_prefs = {}
    if "python" in last_input.lower():
        new_prefs["language"] = "Python"
    elif "typescript" in last_input.lower():
        new_prefs["language"] = "TypeScript"

    if "简洁" in last_input or "简短" in last_input or "极简" in last_input:
        new_prefs["style"] = "concise"
    elif "详细" in last_input or "展开" in last_input:
        new_prefs["style"] = "detailed"

    if new_prefs:
        # 读取旧偏好 → 合并 → 写回
        existing = store.get(namespace, "preferences")
        if existing:
            merged = {**existing.value, **new_prefs}
            store.put(namespace, "preferences", merged)
            print(f"  💾 [长期记忆] 更新偏好: {merged}")
        else:
            store.put(namespace, "preferences", new_prefs)
            print(f"  💾 [长期记忆] 首次保存偏好: {new_prefs}")

    return {}


def chat_node(state: AgentState):
    """正常调用大模型"""
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# ==========================================
# 4. 构建并编译图
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("load_memory", load_memory_node)
workflow.add_node("chat", chat_node)
workflow.add_node("save_memory", save_memory_node)

workflow.add_edge(START, "load_memory")
workflow.add_edge("load_memory", "chat")
workflow.add_edge("chat", "save_memory")
workflow.add_edge("save_memory", END)

app = workflow.compile()

# ==========================================
# 5. 演示：跨线程的长期记忆
# ==========================================
print("=" * 65)
print("📌 线程 001：Alice 首次对话，告诉 Agent 她的偏好")
print("=" * 65)
app.invoke(
    {
        "messages": [HumanMessage(content="你好，我喜欢用 Python，回答尽量简洁")],
        "user_id": "alice",
    },
    config={"configurable": {"thread_id": "thread_001"}},
)

print("\n" + "=" * 65)
print("📌 线程 002：Alice 开启全新对话（不同 thread_id）")
print("=" * 65)
result_2 = app.invoke(
    {
        "messages": [HumanMessage(content="帮我写个二分查找")],
        "user_id": "alice",  # 同一个 user_id
    },
    config={"configurable": {"thread_id": "thread_002"}},
)
print(f"\n🤖 最终回复:\n{result_2['messages'][-1].content}")

print("\n" + "=" * 65)
print("📌 线程 003：Bob 也来问（不同用户，无偏好）")
print("=" * 65)
result_3 = app.invoke(
    {"messages": [HumanMessage(content="帮我写个二分查找")], "user_id": "bob"},
    config={"configurable": {"thread_id": "thread_003"}},
)
print(f"\n🤖 最终回复:\n{result_3['messages'][-1].content}")
