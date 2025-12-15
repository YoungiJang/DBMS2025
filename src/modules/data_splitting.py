from langchain_text_splitters import TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

def token_text_splitter(document,chunk_size,chunk_overlap):
    
    #Text Splitter 결정
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size,      # chunk당 최대 글자 수
        chunk_overlap=chunk_overlap  # 문맥 유지를 위한 overlap
    )

    print("Initializing document splitter...")

    for doc in document:
        chunks = text_splitter.split_documents([doc])
        for chunk in chunks:
            yield chunk  # split된 chunk를 하나씩 다음 단계로 전달



def recursive_character_text_splitter(document,chunk_size,chunk_overlap,separators):
    
    # Text Splitter 결정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # chunk 당 최대 글자 수
        chunk_overlap=chunk_overlap, #문맥 유지를 위한 overlap
        separators=separators       
    )

    print("Initializing document splitter...")

    for doc in document:
        chunks = text_splitter.split_documents([doc])
        
        for chunk in chunks:
            yield chunk  # split된 chunk를 하나씩 다음 단계로 전달