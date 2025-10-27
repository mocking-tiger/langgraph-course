# LangGraph 노드 이름 상수 정의 (문자열 오타 방지 및 코드 가독성 향상)
RETRIEVE = "retrieve"  # 벡터 DB에서 문서 검색 노드
GRADE_DOCUMENTS = "grade_documents"  # 검색된 문서의 관련성 평가 노드
GENERATE = "generate"  # 최종 답변 생성 노드
WEBSEARCH = "websearch"  # 웹 검색 노드 (관련 문서가 부족할 때)