from langchain.chat_models import init_chat_model
from typing_extensions import Annotated, TypedDict
from langchain.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel,Field
import json
import re

def evaluation_wrapper(agent,context_holder):
    def evaluation(input_query:str)->dict:
        """
        A wrapper function that LangSmith evaluation will call.
        inputs_dict must be in the format {"question": "..."}.
        """

        # 1. Run the agent
        # (This call internally triggers the 'prompt_with_context_and_rewrite_and_save' middleware)
        result = agent.invoke({"messages": [{"role": "user", "content": input_query}]})
        answer = result["messages"][-1].content

        # 2. Get the "hidden" retrieved docs
        retrieved_docs = context_holder.get_docs()

        # 3. Return in the format required by the evaluation tutorial
        return {
            "answer": answer,
            "documents": [d.page_content for d in retrieved_docs]
        }
    return evaluation


def extract_and_parse_json(text: str):
    """
    텍스에서 JSON 객체 부분만 강제로 추출하여 파싱하는 함수 (무적 파서)
    """
    try:
        # 1. 가장 기본적인 json.loads 시도
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # 2. 실패 시, 텍스트에서 첫 번째 '{' 와 마지막 '}' 사이를 찾음 (Regex)
            # re.DOTALL: 줄바꿈이 있어도 찾을 수 있게 함
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group()
                return json.loads(json_str)
            else:
                raise ValueError("No JSON object found in text")
        except Exception as e:
            # 3. 그래도 실패하면 에러 발생
            raise ValueError(f"Failed to parse JSON: {e}")

# Evaluator
def relevance(inputs: dict, outputs: dict) -> int:
    # 1. 모델 설정
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    
    # 2. 프롬프트 작성 
    relevance_instructions = """You are an impartial evaluator. Your task is to assess the relevance of a provided ANSWER to a given QUESTION using a 1-5 score.

    You will be given a QUESTION and an ANSWER. Here is the grading criteria:
    - **1 (Poor):** The ANSWER is completely off-topic.
    - **2 (Fair):** The ANSWER is tangentially related.
    - **3 (Average):** The ANSWER partially addresses the QUESTION.
    - **4 (Good):** The ANSWER directly addresses the QUESTION.
    - **5 (Excellent):** The ANSWER addresses the QUESTION's intent perfectly.

    First, analyze the question's intent and the answer's content.
    Finally, provide your score from 1 to 5.

    IMPORTANT: You must return a valid JSON object with the keys "explanation" (string) and "relevant" (integer).
    DO NOT output any text before or after the JSON. Only the JSON object.
    
    Example Output:
    {{
        "explanation": "The answer directly addresses...",
        "relevant": 5
    }}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", relevance_instructions),
        # 여기 있는 {question}과 {answer}는 변수이므로 하나만 씁니다.
        ("human", "QUESTION: {question}\nANSWER: {answer}")
    ])

    # 3. 체인 연결 (파서 없이 모델까지만)
    chain = prompt | model 

    try:
        if not outputs or "answer" not in outputs:
            return 0
            
        # (1) LLM 실행 (Raw Text 받기)
        raw_response = chain.invoke({
            "question": inputs['query'],
            "answer": outputs['answer']
        })
        
        response_text = raw_response.content
        
        # (2) 강제 JSON 추출 및 파싱
        result_dict = extract_and_parse_json(response_text)
        
        # (3) 결과 반환
        return result_dict.get("relevant", 1) # 키가 없으면 기본 1점

    except Exception as e:
        # (4) 에러 발생 시 로그 출력 후 기본 점수 반환
        print(f"Evaluation failed (returning default score 1). Error: {e}")
        return 1

def groundedness(inputs: dict, outputs: dict) -> int:
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    grounded_instructions = """You are an impartial evaluator. Your task is to assess whether an ANSWER is "grounded in" a set of provided CONTEXTS using a 1-5 score.

    You will be given a set of CONTEXTS and an ANSWER. Here are the grading criteria:
    - **1 (Not Grounded):** The ANSWER contains significant information or claims that are NOT supported by the CONTEXTS (i.e., hallucination).
    - **2 (Poorly Grounded):** The ANSWER contains some claims that are not supported, or significantly misrepresents the CONTEXTS.
    - **3 (Partially Grounded):** The ANSWER is mostly supported by the CONTEXTS, but may contain minor claims or details not found in the CONTEXTS.
    - **4 (Well Grounded):** The ANSWER is almost entirely supported by the CONTEXTS, with only very minor embellishments.
    - **5 (Fully Grounded):** Every single claim in the ANSWER is explicitly supported by the provided CONTEXTS.

    First, break down the ANSWER into individual claims. Second, for each claim, check if it is supported by the CONTEXTS. Finally, provide your score from 1 to 5.

    IMPORTANT: You must return a valid JSON object with the keys "explanation" (string) and "grounded" (integer).
    DO NOT output any text before or after the JSON. Only the JSON object.

    Example Output:
    {{
        "explanation": "Every claim in the answer is supported by the provided contexts...",
        "grounded": 5
    }}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", grounded_instructions),
        ("human", "CONTEXTS: {contexts}\n\nANSWER: {answer}")
    ])

    chain = prompt | model

    try:
        if not outputs.get("documents"):
            return 1

        doc_string = "\n\n".join(outputs["documents"])
        
        raw_response = chain.invoke({
            "contexts": doc_string,
            "answer": outputs["answer"]
        })
        
        result_dict = extract_and_parse_json(raw_response.content)
        return result_dict.get("grounded", 1)

    except Exception:
        return 1


def retrieval_relevance(inputs: dict, outputs: dict) -> int:
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    retrieval_relevance_instructions = """You are an impartial evaluator. Your task is to assess the relevance of a set of retrieved CONTEXTS to a given QUESTION using a 1-5 score.

    You will be given a QUESTION and a set of CONTEXTS. Here are the grading criteria:
    - **1 (Poor):** ALL retrieved CONTEXTS are completely irrelevant to the QUESTION.
    - **2 (Fair):** Most CONTEXTS are irrelevant, but one or two might be tangentially related.
    - **3 (Average):** Some CONTEXTS are relevant to the QUESTION, but many are irrelevant or contain noise.
    - **4. (Good):** Most CONTEXTS are relevant and helpful for answering the QUESTION.
    - **5 (Excellent):** ALL retrieved CONTEXTS are highly relevant and crucial for answering the QUESTION.

    First, analyze the QUESTION's intent. Second, examine each CONTEXT for its relevance. Finally, provide your score from 1 to 5 based on the overall relevance of the set.

    IMPORTANT: You must return a valid JSON object with the keys "explanation" (string) and "retrieval" (integer).
    DO NOT output any text before or after the JSON. Only the JSON object.

    Example Output:
    {{
        "explanation": "The retrieved contexts contain the specific information required to answer the question...",
        "retrieval": 5
    }}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", retrieval_relevance_instructions),
        ("human", "CONTEXTS: {contexts}\n\nQUESTION: {question}")
    ])

    chain = prompt | model

    try:
        if not outputs.get("documents"):
            return 1

        doc_string = "\n\n".join(outputs["documents"])

        raw_response = chain.invoke({
            "contexts": doc_string,
            "question": inputs['query']
        })
        
        result_dict = extract_and_parse_json(raw_response.content)
        return result_dict.get("retrieval", 1)

    except Exception:
        return 1