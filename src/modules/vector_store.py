import os
from langchain_community.vectorstores import FAISS

def create_vectorstore(chunk_stream, embedding_model,file_path, batch_size=50):
    vector_store = None
    batch_docs = []
    
    print(f"---Creating FAISS DB(Embedding Chunks)... (Batch size: {batch_size})...---")

    # chunk를 하나씩 받아서 배치 리스트에 추가
    for i, chunk in enumerate(chunk_stream):
        batch_docs.append(chunk)

        # batch가 꽉 차면 처리 시작
        if len(batch_docs) >= batch_size:
            if vector_store is None:
                # 첫 번째 batch는 DB를 생성
                vector_store = FAISS.from_documents(documents=batch_docs, embedding=embedding_model)
            else:
                # 두 번째 batch부터는 기존 DB에 추가
                vector_store.add_documents(batch_docs)
            
            print(f"Processed {i + 1} chunks so far...")
            batch_docs = []

    # batch 처리가 끝난 후 남은 데이터 삽입
    if batch_docs:
        if vector_store is None:
            vector_store = FAISS.from_documents(batch_docs, embedding_model)
        else:
            vector_store.add_documents(batch_docs)
        print(f"Processed final {len(batch_docs)} chunks.")

    # 재사용을 위해 저장
    if vector_store:
        vector_store.save_local(file_path)
        print(f"Vector store saved successfully to '{file_path}'")
    else:
        print("Warning: No documents were processed. Vector store was not created.")
    
    return vector_store

def load_vectorstore(embedding_model,file_path):
    if not os.path.exists(file_path):
        return None
    
    print(f"Loading existing vector store from '{file_path}'...")
    return FAISS.load_local(file_path, embedding_model, allow_dangerous_deserialization=True)