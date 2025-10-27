# 환경 변수(.env 파일) 로드를 위한 모듈 임포트
from dotenv import load_dotenv
# .env 파일의 환경 변수를 로드 (예: API 키 등)
load_dotenv()

# LangGraph의 StateGraph와 END 임포트 (그래프 구성용)
from langgraph.graph import StateGraph, END
# 노드 이름 상수 임포트
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
# 모든 노드 함수 임포트
from graph.nodes import generate, grade_documents, retrieve, web_search
# LangGraph의 State 스키마 임포트
from graph.state import GraphState

# 조건부 엣지 함수: 문서 평가 결과에 따라 웹 검색 또는 답변 생성으로 분기
def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    # web_search 플래그가 True이면 (관련 없는 문서가 있으면)
    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        # 웹 검색 노드로 이동
        return WEBSEARCH
    # 모든 문서가 관련 있으면
    else:
        print("---DECISION: GENERATE---")
        # 답변 생성 노드로 이동
        return GENERATE

# StateGraph 인스턴스 생성 (GraphState 스키마 사용)
workflow = StateGraph(GraphState)

# 노드 추가: 각 노드는 State를 받아서 처리하고 업데이트된 State를 반환
workflow.add_node(RETRIEVE, retrieve)  # 문서 검색 노드
workflow.add_node(GRADE_DOCUMENTS, grade_documents)  # 문서 평가 노드
workflow.add_node(GENERATE, generate)  # 답변 생성 노드
workflow.add_node(WEBSEARCH, web_search)  # 웹 검색 노드

# 시작점 설정: RETRIEVE 노드부터 시작
workflow.set_entry_point(RETRIEVE)

# 엣지 추가: 노드 간 연결
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)  # 검색 후 문서 평가
# 조건부 엣지: 문서 평가 결과에 따라 웹 검색 또는 답변 생성으로 분기
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE})
workflow.add_edge(WEBSEARCH, GENERATE)  # 웹 검색 후 답변 생성
workflow.add_edge(GENERATE, END)  # 답변 생성 후 종료

# 그래프 컴파일 (실행 가능한 앱으로 변환)
app = workflow.compile()
# 그래프를 Mermaid PNG 이미지로 저장 (시각화)
app.get_graph().draw_mermaid_png(output_file_path="graph.png")