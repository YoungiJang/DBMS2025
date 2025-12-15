from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


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


def create_middleware(retriever,model):
    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """Inject context into state messages."""
        last_query = request.state["messages"][-1].content
        print(f"\n--- [Middleware] Original Query: '{last_query}' ---")

        # 2. Request to rewrite the query
        rewrite_system_msg = """You are an expert query assistant. Your task is to rewrite the user's question into an optimized query for a vector database search. Your rewritten query will be used for similarity search.
        Only output the rewritten query."""

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

        rewrite_response = model.invoke(rewrite_prompt_value.messages)

        rewritten_query = rewrite_response.content
        print(f"--- [Middleware] Rewritten Query: '{rewritten_query}' ---")

        retrieved_docs = retriever.invoke(last_query) # using the faiss vector store from our own dataset
        context_holder=RAGContextHolder()
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
    return prompt_with_context