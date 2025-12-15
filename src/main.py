import os
import sys
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langsmith import Client
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from modules.data_loading import data_loader
from modules.data_splitting import token_text_splitter, recursive_character_text_splitter
from modules.embedding import get_embedding_model
from modules.vector_store import create_vectorstore,load_vectorstore
from modules.middleware import create_middleware
from modules.evaluation import *

def main():
    load_dotenv()

    pdf_path = "./data/paper" # 데이터 폴더 경로
    base_directory="vector_stores" #vector store 폴더 경로
    result_dir="results"
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #하이퍼파라미터 지정
    chunk_size=1000
    chunk_overlap=200
    separators=["\n\n", "\n", " ", ""]
    #Separator를 어떻게 정했는지에 대한 설명 텍스트
    sep_details="default"
    splitter="recursive"

    embedding_model=get_embedding_model()

    file_name=f"faiss_cs{chunk_size}_ol{chunk_overlap}_{sep_details}_{splitter}"
    file_path=os.path.join(base_directory,file_name)

    #vector store가 이미 존재하면 새로 만들지 않고 로드
    vector_store=load_vectorstore(embedding_model,file_path)

    if vector_store is None:
        # Step 1: 논문 PDF 로드
        document_stream=data_loader(pdf_path)
        
        #Step 2: 데이터를 Chunk 단위로 Split
        #RecursiveCharacterTextSplitter 활용
        chunk_stream=recursive_character_text_splitter(document_stream,chunk_size,chunk_overlap,separators)

        #TokenTextSplitter 활용
        #chunk_stream=token_text_splitter(document_stream,chunk_size,chunk_overlap)

        #Step 3: Chunk를 Vector Store에 저장
        vector_store=create_vectorstore(chunk_stream,embedding_model,file_path)
    else:
        print(f"Vector store is loaded and ready from {file_path}")

    retriever_k4=vector_store.as_retriever({
        "search_kwargs": {"k":4}
    })
    retriever_k10=vector_store.as_retriever({
        "search_kwargs": {"k":10}
    })
    
    model=ChatGroq(model="llama-3.3-70b-versatile",
    temperature=0)
    RAG_middleware,context_holder=create_middleware(retriever_k4, retriever_k10, model)
    agent=create_agent(model, tools=[], middleware=[RAG_middleware])

    #여기부턴 시연용 코드
    #query에 원하는 질문 입력하면 콘솔에 출력 가능
    '''
    query=["Could you tell me how to understand RNA sequencing?","Explain about Blood Transcriptome."]
    for i,question in enumerate(query,1):
        for step in agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()
    '''

    #여기부턴 Evaluation 코드
    #시연 시 이 아래는 주석 처리
    
    evaluation_func = evaluation_wrapper(agent,context_holder)
    def target(inputs:dict)->dict:
        return evaluation_func(inputs["query"])

    query_dataset=[
        {
            "inputs": {"query": "Which RNA-seq read alignment tools are commonly used in bulk RNA-seq clinical studies?"},
        },
        {
            "inputs": {"query": "Explain about Blood Transcriptome."},
        },
    ]

    client=Client()
    dataset_name="queries"
    if client.has_dataset(dataset_name=dataset_name):
        client.delete_dataset(dataset_name=dataset_name)
    dataset=client.create_dataset(dataset_name=dataset_name)
    client.create_examples(
    dataset_id=dataset.id,
    examples=query_dataset
    )

    experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[relevance,groundedness,retrieval_relevance],
    experiment_prefix="rag-doc-relevance",
    metadata={"version": "none"},
    )

    df=experiment_results.to_pandas()
    csv_file="rag_evaluation_result.csv"
    csv_filename=os.path.join(result_dir,csv_file)
    if os.path.exists(csv_filename):
        try:
            os.remove(csv_filename)
        except PermissionError:
            print(f"Error")

    df.to_csv(csv_filename, index=False, encoding="utf-8-sig")
    
    print(f"Results saved to: {csv_filename}")
    

if __name__ == "__main__":
    main()