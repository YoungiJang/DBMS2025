from langchain_community.vectorstores import FAISS

def test_search(vector_store,query,k):
    print(f"Query: {query}")
    retrieved_chunks=vector_store.similarity_search(query,k)
    print(f"Answer: {retrieved_chunks[0].page_content}")