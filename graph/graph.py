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
# hallucination_grader 체인 임포트
from graph.chains.hallucination_grader import hallucination_grader
# answer_grader 체인 임포트
from graph.chains.answer_grader import answer_grader
# router 체인 임포트
from graph.chains.router import question_router, RouteQuery


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

# 조건부 엣지 함수: 생성된 답변의 품질을 평가하여 다음 단계 결정
# 1. Hallucination 체크: 답변이 문서에 근거하는가?
# 2. Answer 체크: 답변이 질문에 답하는가?
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    # State에서 질문, 문서, 생성된 답변 추출
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    retry_count = state.get("retry_count", 0)  # 재시도 횟수 가져오기 (기본값 0)

    # 최대 재시도 횟수 체크 (3회 제한) - 무한 루프 방지
    MAX_RETRIES = 3
    if retry_count >= MAX_RETRIES:
        print(f"---MAX RETRIES ({MAX_RETRIES}) REACHED, PROCEEDING WITH CURRENT GENERATION---")
        return "useful"  # 재시도 제한 도달 시 강제 종료

    # 1단계: Hallucination 체크 - 답변이 문서에 근거하는지 평가
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    # 답변이 문서에 근거하면 (hallucination 없음)
    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # 2단계: Answer 체크 - 답변이 질문에 답하는지 평가
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        # 답변이 질문에 답하면 → 성공 (END로 이동)
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        # 답변이 질문에 답하지 못하면 → 웹 검색 후 재시도
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    # 답변이 문서에 근거하지 않으면 (hallucination 발생) → 답변 재생성
    else:
        print(f"---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY ({retry_count + 1}/{MAX_RETRIES})---")
        return "not supported"

# 질문을 vectorstore 또는 websearch로 라우팅
def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE

# StateGraph 인스턴스 생성 (GraphState 스키마 사용)
workflow = StateGraph(GraphState)

# 노드 추가: 각 노드는 State를 받아서 처리하고 업데이트된 State를 반환
workflow.add_node(RETRIEVE, retrieve)  # 문서 검색 노드
workflow.add_node(GRADE_DOCUMENTS, grade_documents)  # 문서 평가 노드
workflow.add_node(GENERATE, generate)  # 답변 생성 노드
workflow.add_node(WEBSEARCH, web_search)  # 웹 검색 노드

# 시작점 설정: RETRIEVE 노드부터 시작
# workflow.set_entry_point(RETRIEVE)

# 질문을 vectorstore 또는 websearch로 라우팅
workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

# 엣지 추가: 노드 간 연결
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)  # 검색 후 문서 평가
# 조건부 엣지 1: 문서 평가 결과에 따라 웹 검색 또는 답변 생성으로 분기
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_to_generate, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE})
# 조건부 엣지 2: 생성된 답변 품질 평가 후 분기
# - "not supported" (hallucination): GENERATE로 돌아가서 재생성
# - "useful" (성공): END로 종료
# - "not useful" (질문에 답 못함): WEBSEARCH로 이동하여 추가 정보 수집
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,  # Hallucination → 재생성
        "useful": END,  # 성공 → 종료
        "not useful": WEBSEARCH,  # 질문에 답 못함 → 웹 검색
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)  # 웹 검색 후 답변 생성
workflow.add_edge(GENERATE, END)  # 답변 생성 후 종료 (기본 경로, 조건부 엣지에 의해 오버라이드됨)

# 그래프 컴파일 (실행 가능한 앱으로 변환)
app = workflow.compile()
# 그래프를 Mermaid PNG 이미지로 저장 (시각화)
app.get_graph().draw_mermaid_png(output_file_path="graph.png")