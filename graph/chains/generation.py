# LangSmith 클라이언트 임포트 (LangChain Hub 프롬프트 템플릿 다운로드용)
from langsmith import Client
# 문자열 출력 파서 임포트 (LLM 출력을 문자열로 변환)
from langchain_core.output_parsers import StrOutputParser
# OpenAI의 ChatGPT 모델을 사용하기 위한 클래스 임포트
from langchain_openai import ChatOpenAI

# ChatGPT 4o-mini 모델 인스턴스 생성 (temperature=0으로 일관된 출력 보장)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# LangSmith 클라이언트 생성
client = Client()
# LangChain Hub에서 RAG용 프롬프트 템플릿 가져오기 (langchainhub는 deprecated, langsmith 사용)
prompt = client.pull_prompt("rlm/rag-prompt")

# Generation 체인: 프롬프트 → LLM → 문자열 파서
# 검색된 문서와 질문을 받아서 최종 답변을 생성
generation_chain = prompt | llm | StrOutputParser()