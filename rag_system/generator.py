# rag_system/generator.py
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from rag_system.retriever_faiss import FAISSRetriever
from config import GROQ_API_KEY, GROQ_MODEL

class RAGGenerator:
    def __init__(self):
        self.retriever = FAISSRetriever()
        self.llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)

    def answer_question(self, query, return_score=False):
        retrieved = self.retriever.retrieve(query, top_k=1)
        if not retrieved:
            return ("No data available", 0.0) if return_score else "No data available"

        top = retrieved[0]
        context = f"Context:\nQ: {top['question']}\nA: {top['answer']}\n\nUser Question: {query}\nAnswer clearly:"

        response = self.llm([HumanMessage(content=context)])
        answer = response.content

        if return_score:
            return answer, top["score"]
        return answer