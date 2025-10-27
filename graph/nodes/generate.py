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

    # generation_chain을 사용하여 문서와 질문을 바탕으로 답변 생성
    # context: 검색된 문서들, question: 사용자 질문
    generation = generation_chain.invoke({"context": documents, "question": question})
    # 문서, 질문, 생성된 답변을 반환하여 State 업데이트
    return {"documents": documents, "question": question, "generation": generation}