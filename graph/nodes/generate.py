# 타입 힌팅을 위한 Any, Dict 임포트
from typing import Any, Dict

# generation 체인 임포트 (문서와 질문으로 답변 생성)
from graph.chains.generation import generation_chain
# LangGraph의 State 스키마 임포트
from graph.state import GraphState


# LangGraph 노드 함수: 검색된 문서를 기반으로 최종 답변 생성
def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    # State에서 사용자 질문 추출
    question = state["question"]
    # State에서 검색 및 필터링된 문서 목록 추출
    documents = state["documents"]
    # State에서 재시도 횟수 가져오기 (기본값 0)
    retry_count = state.get("retry_count", 0)

    # generation_chain을 사용하여 문서와 질문을 바탕으로 답변 생성
    # context: 검색된 문서들, question: 사용자 질문
    generation = generation_chain.invoke({"context": documents, "question": question})

    # 재시도 횟수 증가 (hallucination 방지 재시도용)
    retry_count += 1

    # 문서, 질문, 생성된 답변, 재시도 횟수를 반환하여 State 업데이트
    return {"documents": documents, "question": question, "generation": generation, "retry_count": retry_count}