# 환경 변수(.env 파일) 로드를 위한 모듈 임포트
from dotenv import load_dotenv
# .env 파일의 환경 변수를 로드 (OpenAI API 키가 필요하므로 최상단에서 로드)
load_dotenv()

# Literal 타입 힌팅 임포트 (제한된 값만 허용)
from typing import Literal

# 프롬프트 템플릿을 생성하기 위한 클래스 임포트
from langchain_core.prompts import ChatPromptTemplate
# Pydantic 모델과 필드 정의를 위한 클래스 임포트 (데이터 검증 및 스키마 정의)
from pydantic import BaseModel, Field
# OpenAI의 ChatGPT 모델을 사용하기 위한 클래스 임포트
from langchain_openai import ChatOpenAI


# 라우팅 결과를 담는 스키마: 질문을 vectorstore 또는 websearch로 라우팅
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    # datasource 필드: "vectorstore" 또는 "websearch" 중 하나만 가능
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# ChatGPT 4o-mini 모델 인스턴스 생성 (temperature=0으로 일관된 라우팅 결정 보장)
# Structured Output API 지원 모델 사용
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# LLM이 RouteQuery 스키마 형식으로만 출력하도록 구조화
structured_llm_router = llm.with_structured_output(RouteQuery)

# 시스템 프롬프트: Router의 역할과 라우팅 기준을 정의
# - vectorstore: agents, prompt engineering, adversarial attacks 관련 질문
# - websearch: 그 외 모든 질문
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""
# Router 프롬프트 템플릿 생성: 시스템 메시지 + 사용자 질문
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 시스템 역할 정의
        ("human", "{question}"),  # 라우팅할 질문
    ]
)

# 최종 Question Router 체인: 프롬프트 → 구조화된 LLM → vectorstore/websearch 결정
question_router = route_prompt | structured_llm_router