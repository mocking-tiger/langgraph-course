# 프롬프트 템플릿을 생성하기 위한 클래스 임포트
from langchain_core.prompts import ChatPromptTemplate
# Pydantic 모델과 필드 정의를 위한 클래스 임포트 (데이터 검증 및 스키마 정의)
from pydantic import BaseModel, Field
# OpenAI의 ChatGPT 모델을 사용하기 위한 클래스 임포트
from langchain_openai import ChatOpenAI

# ChatGPT 4o-mini 모델 인스턴스 생성 (temperature=0으로 일관된 평가 보장)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 문서 평가 결과를 담는 스키마: 검색된 문서가 질문과 관련 있는지 이진 평가 (yes/no)
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    # binary_score 필드: 문서의 관련성을 'yes' 또는 'no'로 평가
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM이 GradeDocuments 스키마 형식으로만 출력하도록 구조화
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 시스템 프롬프트: Grader의 역할과 평가 기준을 정의
# - 문서에 키워드나 의미적 관련성이 있으면 'yes'
# - 관련성이 없으면 'no'
system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
# Grader 프롬프트 템플릿 생성: 시스템 메시지 + 사용자 입력 (문서와 질문)
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),  # 시스템 역할 정의
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),  # 평가할 문서와 질문
    ]
)

# 최종 Retrieval Grader 체인: 프롬프트 → 구조화된 LLM → yes/no 평가 결과
retrieval_grader = grade_prompt | structured_llm_grader