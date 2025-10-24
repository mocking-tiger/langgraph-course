"""
Reflection Agent 패턴을 위한 Chain 정의

Reflection 패턴:
1. Generate: 초기 콘텐츠 생성
2. Reflect: 생성된 콘텐츠를 평가하고 피드백 제공
3. Generate: 피드백을 받아 콘텐츠 개선
4. 2-3 반복 (만족스러운 결과가 나올 때까지)

이 파일은 트윗 작성을 위한 Reflection Agent의 두 가지 핵심 체인을 정의합니다.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Reflection 프롬프트: 생성된 트윗을 평가하고 개선 방향을 제시하는 역할
# - 바이럴 트위터 인플루언서 관점에서 트윗을 평가
# - 길이, 바이럴성, 스타일 등에 대한 구체적인 피드백 제공
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),  # 이전 대화 기록 (생성된 트윗 포함)
    ]
)

# Generation 프롬프트: 트윗을 생성하거나 피드백을 반영해 개선하는 역할
# - 초기 요청에 대해 트윗 생성
# - 피드백이 있으면 이를 반영하여 트윗을 개선
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),  # 이전 대화 기록 (피드백 포함)
    ]
)

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 두 가지 핵심 체인
generate_chain = generation_prompt | llm  # 트윗 생성/개선 체인
reflect_chain = reflection_prompt | llm   # 트윗 평가/피드백 체인