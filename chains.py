# 날짜/시간 처리를 위한 모듈 임포트
import datetime
# 환경 변수(.env 파일) 로드를 위한 모듈 임포트
from dotenv import load_dotenv
# .env 파일의 환경 변수를 로드 (예: API 키 등)
load_dotenv()

# LangChain의 출력 파서 임포트 (OpenAI 도구 출력을 JSON/Pydantic 형식으로 파싱)
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
# 사용자 메시지를 생성하기 위한 클래스 임포트
from langchain_core.messages import HumanMessage
# 프롬프트 템플릿과 메시지 플레이스홀더를 생성하기 위한 클래스 임포트
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# OpenAI의 ChatGPT 모델을 사용하기 위한 클래스 임포트
from langchain_openai import ChatOpenAI

# Actor 에이전트의 프롬프트 템플릿 정의
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        # 시스템 메시지: Actor의 역할과 지시사항 정의
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        # 대화 기록을 삽입할 플레이스홀더 (이전 메시지들이 여기에 들어감)
        MessagesPlaceholder(variable_name="messages"),
        # 최종 지시사항: 요구된 형식으로 답변하도록 지시
        ("system", "Answer the user's question above using the required format."),
    ]
# partial: 템플릿의 일부 변수를 미리 바인딩 (time 변수를 현재 시간으로 자동 설정)
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)
