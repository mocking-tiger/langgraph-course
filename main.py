# 환경 변수(.env 파일) 로드를 위한 모듈 임포트
from dotenv import load_dotenv
# .env 파일의 환경 변수를 로드 (예: API 키 등)
load_dotenv()

# 컴파일된 LangGraph 앱 임포트 (Agentic RAG 그래프)
from graph.graph import app

# 메인 함수: Agentic RAG 그래프를 실행
def main():
    print("main함수 실행")
    # 그래프에 질문을 입력하여 실행
    # 플로우: RETRIEVE → GRADE_DOCUMENTS → (WEBSEARCH?) → GENERATE
    print(app.invoke({"question": "what is agent memory? answer in korean."}))

# 직접 실행 시 메인 함수 호출
if __name__ == "__main__":
    main()