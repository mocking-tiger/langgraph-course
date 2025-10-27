from dotenv import load_dotenv

load_dotenv()

from typing import List
from langchain_core.messages import BaseMessage,ToolMessage
from langgraph.graph import END, MessageGraph, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

from chains import first_responder, revisor
from tool_executor import execute_tools

# State 스키마 정의: messages 필드를 가진 State
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

MAX_ITERATIONS = 2
builder = StateGraph(State)

# 노드 함수들을 State 형식에 맞게 래핑
def draft_node(state: State):
    return {"messages": [first_responder.invoke(state)]}

def revise_node(state: State):
    return {"messages": [revisor.invoke(state)]}

builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)

builder.add_edge("draft","execute_tools")
builder.add_edge("execute_tools","revise")

def event_loop(state: State) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop,{END: END,"execute_tools": "execute_tools"})
builder.set_entry_point("draft")

graph = builder.compile()
# graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

def main():
    print("main함수 실행")
    from langchain_core.messages import HumanMessage

    res = graph.invoke({
        "messages": [
            HumanMessage(content="Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital.")
        ]
    })
    print(res["messages"][-1].tool_calls[0]["args"]["answer"])
    print(res)

if __name__ == "__main__":
    main()