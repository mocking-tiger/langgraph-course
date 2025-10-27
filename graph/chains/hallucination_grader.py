# 프롬프트 템플릿을 생성하기 위한 클래스 임포트
from langchain_core.prompts import ChatPromptTemplate
# RunnableSequence 타입 힌팅용 임포트
from langchain_core.runnables import RunnableSequence
# OpenAI의 ChatGPT 모델을 사용하기 위한 클래스 임포트
from langchain_openai import ChatOpenAI
# Pydantic 모델과 필드 정의를 위한 클래스 임포트 (데이터 검증 및 스키마 정의)
from pydantic import BaseModel, Field

# ChatGPT 4o-mini 모델 인스턴스 생성 (temperature=0으로 일관된 평가 보장)
# Structured Output API 지원 모델 사용
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Hallucination 평가 결과를 담는 스키마: LLM 생성 답변이 문서에 근거하는지 평가
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    # binary_score 필드: 답변이 문서에 근거하는지 True/False로 평가
    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM이 GradeHallucinations 스키마 형식으로만 출력하도록 구조화
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 시스템 프롬프트: Hallucination Grader의 역할과 평가 기준을 정의
# - LLM 생성 답변이 검색된 문서에 근거하면 'yes'
# - 근거 없이 지어낸 내용이면 'no' (hallucination)
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
# Hallucination Grader 프롬프트 템플릿 생성: 시스템 메시지 + 사용자 입력 (문서와 생성된 답변)
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 시스템 역할 정의
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),  # 평가할 문서와 답변
    ]
)

# 최종 Hallucination Grader 체인: 프롬프트 → 구조화된 LLM → True/False 평가 결과
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader