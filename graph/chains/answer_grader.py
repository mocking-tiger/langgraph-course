# 프롬프트 템플릿을 생성하기 위한 클래스 임포트
from langchain_core.prompts import ChatPromptTemplate
# RunnableSequence 타입 힌팅용 임포트
from langchain_core.runnables import RunnableSequence
# OpenAI의 ChatGPT 모델을 사용하기 위한 클래스 임포트
from langchain_openai import ChatOpenAI
# Pydantic 모델과 필드 정의를 위한 클래스 임포트 (데이터 검증 및 스키마 정의)
from pydantic import BaseModel, Field


# Answer 평가 결과를 담는 스키마: LLM 생성 답변이 질문에 답하는지 평가
class GradeAnswer(BaseModel):

    # binary_score 필드: 답변이 질문에 답하는지 True/False로 평가
    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# ChatGPT 4o-mini 모델 인스턴스 생성 (temperature=0으로 일관된 평가 보장)
# Structured Output API 지원 모델 사용
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# LLM이 GradeAnswer 스키마 형식으로만 출력하도록 구조화
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# 시스템 프롬프트: Answer Grader의 역할과 평가 기준을 정의
# - LLM 생성 답변이 사용자 질문에 답하면 'yes'
# - 질문과 무관하거나 답하지 못하면 'no'
system = """You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
# Answer Grader 프롬프트 템플릿 생성: 시스템 메시지 + 사용자 입력 (질문과 생성된 답변)
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 시스템 역할 정의
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),  # 평가할 질문과 답변
    ]
)

# 최종 Answer Grader 체인: 프롬프트 → 구조화된 LLM → True/False 평가 결과
answer_grader: RunnableSequence = answer_prompt | structured_llm_grader