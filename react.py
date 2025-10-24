# ReAct 패턴을 위한 도구(Tool)와 LLM 설정 파일
# 이 파일은 LangGraph에서 사용할 도구들과 도구가 바인딩된 LLM을 정의합니다.

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

# 커스텀 도구: 숫자를 3배로 만드는 간단한 함수
@tool
def triple(num:float) -> float:
  """
  param num: a number to triple
  returns: the triple of the input number
  """
  return float(num) * 3

# 에이전트가 사용할 수 있는 도구 리스트
# - TavilySearch: 웹 검색 도구 (최대 1개 결과 반환)
# - triple: 숫자를 3배로 만드는 커스텀 도구
tools = [TavilySearch(max_results=1), triple]

# LLM 설정: GPT-4o-mini 모델에 도구들을 바인딩
# bind_tools()를 통해 LLM이 필요시 도구를 호출할 수 있도록 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)