# 환경 변수(.env 파일) 로드를 위한 모듈 임포트
from dotenv import load_dotenv
# .env 파일의 환경 변수를 로드 (예: Tavily API 키 등)
load_dotenv()

# Tavily 검색 도구 임포트 (웹 검색 API)
from langchain_tavily import TavilySearch
# 구조화된 도구를 정의하기 위한 클래스 임포트
from langchain_core.tools import StructuredTool
# LangGraph의 사전 구축된 도구 노드 임포트
from langgraph.prebuilt import ToolNode

# schemas.py에서 정의한 답변 및 수정 스키마 임포트
from schemas import AnswerQuestion, ReviseAnswer

# Tavily 검색 도구 인스턴스 생성 (최대 5개의 검색 결과 반환)
tavily_tool = TavilySearch(max_results=5)

# 여러 검색 쿼리를 실행하는 함수
def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    # search_queries 리스트의 각 쿼리를 딕셔너리 형태로 변환하여 batch 실행
    # tavily_tool.batch()는 여러 쿼리를 동시에 처리하여 검색 결과를 반환
    return tavily_tool.batch([{"query": query} for query in search_queries])

execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)