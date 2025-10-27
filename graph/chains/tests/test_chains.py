# 환경 변수(.env 파일) 로드를 위한 모듈 임포트
from dotenv import load_dotenv

from graph.chains.router import RouteQuery, question_router
# .env 파일의 환경 변수를 로드 (예: API 키 등)
load_dotenv()

# retrieval_grader 체인과 결과 스키마 임포트
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
# generation 체인 임포트
from graph.chains.generation import generation_chain
# ingestion.py에서 생성한 벡터 DB retriever 임포트
from ingestion import retriever
# pprint 모듈 임포트
from pprint import pprint
# hallucination_grader 체인 임포트
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations


# 테스트 1: Retrieval Grader가 관련 있는 문서에 대해 'yes'를 반환하는지 검증
def test_retrival_grader_answer_yes() -> None:
    # 검색 쿼리: "agent memory"
    question = "agent memory"
    # retriever가 벡터 DB에서 "agent memory"와 유사한 문서들을 검색
    # 내부적으로 문자열을 임베딩 벡터로 변환 후 유사도 검색 수행
    docs = retriever.invoke(question)
    # 검색된 문서 중 두 번째 문서의 내용 추출
    doc_txt = docs[1].page_content

    # retrieval_grader에 질문과 문서를 전달하여 관련성 평가
    # question과 document가 모두 agent 관련이므로 'yes'가 예상됨
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    # 결과가 'yes'인지 확인 (관련 있는 문서이므로)
    assert res.binary_score == "yes"

# 테스트 2: Retrieval Grader가 관련 없는 문서에 대해 'no'를 반환하는지 검증
def test_retrival_grader_answer_no() -> None:
    # 검색 쿼리: "agent memory"로 agent 관련 문서 검색
    question = "agent memory"
    # retriever가 벡터 DB에서 agent 관련 문서들을 검색
    docs = retriever.invoke(question)
    # 검색된 문서 중 두 번째 문서의 내용 추출 (agent 관련 내용)
    doc_txt = docs[1].page_content

    # retrieval_grader에 전혀 다른 질문("pizza 만드는 법")과 agent 문서를 전달
    # 질문과 문서가 관련 없으므로 'no'가 예상됨
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    # 결과가 'no'인지 확인 (관련 없는 문서이므로)
    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score

def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"