# 타입 힌팅을 위한 Any, Dict 임포트
from typing import Any, Dict

# 환경 변수(.env 파일) 로드를 위한 모듈 임포트
from dotenv import load_dotenv
# .env 파일의 환경 변수를 로드 (예: Tavily API 키 등)
load_dotenv()
# Document 클래스 임포트 (langchain.schema는 deprecated, langchain_core 사용)
from langchain_core.documents import Document
# Tavily 웹 검색 도구 임포트 (AI 기반 검색 API)
from langchain_tavily import TavilySearch

# LangGraph의 State 스키마 임포트
from graph.state import GraphState

# Tavily 웹 검색 도구 인스턴스 생성 (최대 3개의 검색 결과 반환)
web_search_tool = TavilySearch(max_results=3)

# LangGraph 노드 함수: 웹 검색을 수행하여 추가 문서를 가져옴
def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    # State에서 사용자 질문 추출
    question = state["question"]
    # State에서 기존 문서 목록 추출 (Router에서 바로 온 경우 없을 수 있음)
    documents = state.get("documents")

    # Tavily API를 사용하여 질문에 대한 웹 검색 수행
    tavily_results = web_search_tool.invoke({"query": question})

    # Tavily 반환 형식 처리: 문자열이면 그대로 사용, 리스트면 content 추출
    if isinstance(tavily_results, str):
        # 문자열로 반환된 경우 (최신 TavilySearch 형식)
        joined_tavily_result = tavily_results
    elif isinstance(tavily_results, list):
        # 리스트로 반환된 경우 (구 버전 형식)
        joined_tavily_result = "\n".join(
            [tavily_result["content"] for tavily_result in tavily_results]
        )
    else:
        # 예상치 못한 형식인 경우 빈 문자열 사용
        joined_tavily_result = ""

    # 검색 결과를 LangChain Document 객체로 변환
    web_results = Document(page_content=joined_tavily_result)
    # 기존 문서 목록이 있으면 웹 검색 결과를 추가
    if documents is not None:
        documents.append(web_results)
    # 기존 문서 목록이 없으면 웹 검색 결과로만 새 리스트 생성
    else:
        documents = [web_results]
    # 업데이트된 문서 목록과 질문을 반환하여 State 업데이트
    return {"documents": documents, "question": question}


# 직접 실행 시 테스트
if __name__ == "__main__":
    web_search({"question": "agent memory", "documents":None})