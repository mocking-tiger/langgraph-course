"""
LangGraph ReAct 패턴 구현

전체 흐름:
1. START → AGENT_REASON (Entry Point)
2. AGENT_REASON: LLM이 사용자 메시지를 분석하고 다음 행동 결정
3. should_continue: 조건부 분기
   - tool_calls가 있으면 → ACT (도구 실행)
   - tool_calls가 없으면 → END (종료)
4. ACT: 도구 실행 후 결과 생성
5. ACT → AGENT_REASON: 도구 실행 결과를 다시 Agent에게 전달
6. 2-5 반복 (더 이상 도구가 필요 없을 때까지)

이 패턴을 통해 Agent는 복잡한 문제를 단계적으로 해결할 수 있습니다.
"""

from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoning, tool_node

load_dotenv()

# 노드 이름 상수 정의
AGENT_REASON = "agent_reason"  # Agent 추론 노드
ACT = "act"                     # 도구 실행 노드
LAST = -1                       # 메시지 리스트의 마지막 인덱스

# 조건부 엣지: Agent가 다음에 어디로 갈지 결정하는 함수
def should_continue(state: MessagesState) -> str:
  """
  마지막 메시지에 tool_calls가 있는지 확인하여 다음 노드 결정

  - tool_calls가 있으면: ACT 노드로 이동 (도구 실행 필요)
  - tool_calls가 없으면: END로 이동 (최종 답변 완료)
  """
  if not state["messages"][LAST].tool_calls:
    return END
  return ACT

# StateGraph 생성: MessagesState를 상태로 사용
flow = StateGraph(MessagesState)

# 노드 추가 및 연결
flow.add_node(AGENT_REASON, run_agent_reasoning)  # Agent 추론 노드 추가
flow.set_entry_point(AGENT_REASON)                # 시작점을 AGENT_REASON으로 설정
flow.add_node(ACT, tool_node)                     # 도구 실행 노드 추가

# 조건부 엣지 추가: AGENT_REASON에서 나가는 엣지
# should_continue 함수의 반환값에 따라 END 또는 ACT로 이동
flow.add_conditional_edges(AGENT_REASON, should_continue,{END: END, ACT: ACT})

# 일반 엣지 추가: ACT 노드에서 AGENT_REASON으로 다시 돌아감 (순환)
# 도구 실행 결과를 Agent가 다시 분석할 수 있도록 함
flow.add_edge(ACT, AGENT_REASON)

# 그래프 컴파일: 실행 가능한 앱으로 변환
app = flow.compile()

# 그래프 시각화: flow.png 파일로 저장
# app.get_graph().draw_mermaid_png(output_file_path="flow.png")

def main():
    print("main함수 실행")
    # invoke: 동기 방식으로 그래프 실행 (ainvoke는 비동기 방식으로 await 필요)
    res = app.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo? List it and then triple it")]})
    print(res["messages"][LAST].content)

if __name__ == "__main__":
    main()
