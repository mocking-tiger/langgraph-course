# LangGraph의 노드(Node) 정의 파일
# 이 파일은 그래프를 구성하는 노드들을 정의합니다.
# ReAct 패턴에서는 크게 2가지 노드가 필요합니다:
# 1. Agent Reasoning Node: LLM이 사용자 요청을 분석하고 다음 행동을 결정
# 2. Tool Node: 실제로 도구를 실행하는 노드

from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

# 에이전트에게 부여할 시스템 메시지 (역할 정의)
SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""

# Agent Reasoning Node: LLM이 현재 상태를 분석하고 응답을 생성하는 노드
# - 사용자 메시지와 이전 대화 내용을 분석
# - 필요하다면 도구 호출(tool_calls)을 포함한 응답 생성
# - 도구가 필요없다면 최종 답변 생성
def run_agent_reasoning(state: MessagesState) -> MessagesState:
  """
  Run the agent reasoning node.
  """
  response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
  return {"messages": [response]}

# Tool Node: LLM이 요청한 도구를 실제로 실행하는 노드
# - Agent가 tool_calls를 생성하면 이 노드가 실행됨
# - 도구 실행 결과를 메시지로 반환
tool_node = ToolNode(tools)