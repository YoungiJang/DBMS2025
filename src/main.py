import os
from modules.data_loading import data_loader
from modules.data_splitting import token_text_splitter, recursive_character_text_splitter
from modules.embedding import get_embedding_model
from modules.vector_store import create_vectorstore,load_vectorstore
from modules.vector_store_test import test_search

def main():
    pdf_path = "./data/papers" # 데이터 폴더 경로
    base_directory="vector_stores" #vector store 폴더 경로
    if not os.path.exists(base_directory):
        os.makedirs(base_directory)

    #하이퍼파라미터 지정
    chunk_size=1000
    chunk_overlap=200
    separators=["\n\n", "\n", " ", ""]
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
    
    #Vector store 테스트를 위한 코드
    k=1
    query="Tell me about independence"
    test_search(vector_store,query,k)


if __name__ == "__main__":
    main()