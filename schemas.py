# 타입 힌팅을 위한 List 타입 임포트
from typing import List
# Pydantic의 기본 모델과 필드 정의를 위한 클래스 임포트 (데이터 검증 및 스키마 정의에 사용)
from pydantic import BaseModel, Field

# Reflection 스키마: Actor가 자신의 답변을 비판적으로 평가한 결과를 담는 클래스
class Reflection(BaseModel):
    # missing 필드: 답변에서 빠진 내용에 대한 비판
    missing:str = Field(description="Critique of what is missing.")
    # superfluous 필드: 답변에서 불필요하거나 과한 내용에 대한 비판
    superfluous:str = Field(description="Critique of what is superfluous.")

# AnswerQuestion 스키마: Actor가 질문에 답변할 때 사용하는 전체 응답 구조를 정의하는 클래스
class AnswerQuestion(BaseModel):
    """Answer the question"""

    # answer 필드: 질문에 대한 약 250단어 분량의 상세한 답변
    answer:str = Field(description="~250 words detailed answer to the question.")
    # reflection 필드: 초기 답변에 대한 자기 성찰 (Reflection 객체)
    reflection:Reflection = Field(description="Your Reflection on the initial answer.")
    # search_queries 필드: 답변 개선을 위해 추가 조사가 필요한 1~3개의 검색 쿼리 목록
    search_queries:List[str] = Field(description="1~3 search queries for researching improvements to address the critique of your current answer.")