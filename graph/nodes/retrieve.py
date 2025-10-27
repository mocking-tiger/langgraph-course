# 타입 힌팅을 위한 Any, Dict 임포트
from typing import Any, Dict
# LangGraph의 State 스키마 임포트
from graph.state import GraphState
# ingestion.py에서 생성한 벡터 DB retriever 임포트
from ingestion import retriever

# LangGraph 노드 함수: 벡터 DB에서 관련 문서 검색
def retrieve(state: GraphState) -> Dict[str, Any]:
    print('---RETRIEVE---')
    # State에서 사용자 질문 추출
    question = state['question']
    # retriever를 사용하여 질문과 유사한 문서 검색
    # 내부적으로 질문을 임베딩 벡터로 변환 후 유사도 검색 수행
    documents = retriever.invoke(question)
    # 검색된 문서와 질문을 반환하여 State 업데이트
    return {"documents": documents, "question": question}