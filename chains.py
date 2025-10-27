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

# schemas.py에서 정의한 답변 및 수정 스키마 임포트
from schemas import AnswerQuestion, ReviseAnswer

# ChatGPT 4o-mini 모델 인스턴스 생성 (temperature=0으로 일관된 출력 보장)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# JSON 형식의 도구 출력을 파싱하는 파서 (도구 호출 ID 포함)
parser = JsonOutputToolsParser(return_id=True)
# Pydantic 모델로 도구 출력을 파싱하는 파서 (AnswerQuestion 스키마 사용)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

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

# 첫 번째 응답자(First Responder) 프롬프트 템플릿: 초기 답변 생성을 위한 지시사항 설정
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

# 첫 번째 응답자 체인: 프롬프트와 LLM을 연결하고 AnswerQuestion 도구를 바인딩
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

# 수정자(Revisor)를 위한 지시사항: 새로운 정보를 활용해 답변을 개선하는 방법 정의
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

# 수정자(Revisor) 체인: 답변 수정을 위한 프롬프트와 LLM 연결, ReviseAnswer 도구 바인딩
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

# 메인 실행 블록: 직접 스크립트를 실행할 때만 동작
if __name__ == "__main__":
    # 테스트용 사용자 메시지 생성
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
        " list startups that do that and raised capital."
    )
    # 첫 번째 응답자 체인 구성: 프롬프트 -> LLM(도구 바인딩) -> Pydantic 파서
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )
    # 체인 실행 및 결과 출력
    res = chain.invoke(input={"messages": [human_message]})
    print(res)
