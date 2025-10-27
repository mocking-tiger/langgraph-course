# 타입 힌팅을 위한 Any, Dict 임포트
from typing import Any, Dict

# retrieval_grader 체인 임포트 (문서 관련성 평가용)
from graph.chains.retrieval_grader import retrieval_grader
# LangGraph의 State 스키마 임포트
from graph.state import GraphState


# LangGraph 노드 함수: 검색된 문서들의 관련성을 평가하고 필터링
def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    # State에서 사용자 질문 추출
    question = state["question"]
    # State에서 검색된 문서 목록 추출
    documents = state["documents"]

    # 관련 있는 문서만 담을 빈 리스트
    filtered_docs = []
    # 웹 검색 필요 여부 플래그 (기본값: False)
    web_search = False
    # 각 문서를 순회하며 관련성 평가
    for d in documents:
        # retrieval_grader를 사용해 질문과 문서의 관련성 평가 (yes/no)
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        # 평가 결과 (yes 또는 no)
        grade = score.binary_score
        # 문서가 관련 있는 경우 ('yes')
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            # 필터링된 문서 리스트에 추가
            filtered_docs.append(d)
        # 문서가 관련 없는 경우 ('no')
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # 웹 검색 플래그를 True로 설정 (추가 정보 필요)
            web_search = True
            # 이 문서는 건너뛰고 다음 문서로
            continue
    # 필터링된 문서, 질문, 웹 검색 플래그를 반환하여 State 업데이트
    return {"documents": filtered_docs, "question": question, "web_search": web_search}