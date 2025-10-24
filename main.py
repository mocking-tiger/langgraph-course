from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain

class MessageGraph(TypedDict):
  messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
  return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
  res = reflect_chain.invoke({"messages": state["messages"]})
  return {"messages": [HumanMessage(content=res.content)]}

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: MessageGraph):
  if len(state["messages"]) > 6:
    return END
  return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, path_map={REFLECT: REFLECT, END: END})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="flow.png")

def main():
  print("main함수 실행")

if __name__ == "__main__":
  main()