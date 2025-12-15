from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import json
import re

class RAGContextHolder:
    def __init__(self):
        # A variable to store the most recently retrieved docs
        self.last_retrieved_docs = []

    def set_docs(self, docs: list[Document]):
        """Called by the middleware to save the retrieved docs"""
        self.last_retrieved_docs = docs

    def get_docs(self) -> list[Document]:
        """Called by the evaluation function to get the saved docs"""
        return self.last_retrieved_docs


def create_middleware(retriever_k4, retriever_k10,model):
    context_holder=RAGContextHolder()
    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """Inject context into state messages."""
        last_query = request.state["messages"][-1].content
        print(f"\n--- [Middleware] Original Query: '{last_query}' ---")

        # 2. Request to rewrite the query
        rewrite_system_msg = """
            You are an expert assistant for literature retrieval.

            Given a user's question, do the following:
            1. Rewrite the question into an optimized query for vector database similarity search over RNA-seq research papers.
            2. Decide whether the rewritten query is an aggregation-style query that requires retrieving evidence from multiple papers.

            Return your output strictly in the following JSON format:
            {{
            "rewritten_query": "...",
            "is_aggregation": true or false
            }}
            """

        # Make a template
        rewrite_template = ChatPromptTemplate(
            [
                ("system", rewrite_system_msg),
                ("human", "{user_input}")
            ]
        )

        # Fill in the template with the query content
        rewrite_prompt_value = rewrite_template.invoke(
            {
                "user_input": last_query,
            }
        )

        # rewrite_response = model.invoke(rewrite_prompt_value.messages)
        rewrite_response = model.invoke(rewrite_prompt_value.messages)

        # import json
        raw = rewrite_response.content.strip()

        # JSON 블록만 추출
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if not match:
            raise ValueError(f"Rewrite LLM did not return JSON:\n{raw}")

        rewrite_result = json.loads(match.group())

        rewritten_query = rewrite_result["rewritten_query"]
        is_aggregation = rewrite_result["is_aggregation"]
        print(f"--- [Middleware] Rewritten Query: '{rewritten_query}' ---")

        if is_aggregation:
            retriever = retriever_k10
            print(f"--- [Middleware] Using retriever_k10 for aggregation query ---")
        else:
            retriever = retriever_k4
            print(f"--- [Middleware] Using retriever_k4 for non-aggregation query ---")

        retrieved_docs = retriever.invoke(last_query) # using the faiss vector store from our own dataset
        
        context_holder.set_docs(retrieved_docs)
        print(f"--- [Middleware] Saved {len(retrieved_docs)} docs to Context Holder ---")

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        print(f"--- [Middleware] Retrieved {len(retrieved_docs)} docs ---")

        system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        "\n\n--- CONTEXT ---"
        f"\n{docs_content}"
        "\n--- END CONTEXT ---"
        )

        return system_message
    return prompt_with_context,context_holder