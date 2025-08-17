import streamlit as st
import time
from rag_system.generator import RAGGenerator
from fine_tuned_system.model import FineTunedModel

st.set_page_config(page_title="Financial Q&A Assistant", layout="wide")

@st.cache_resource
def load_models():
    return RAGGenerator(), FineTunedModel()

rag_model, ft_model = load_models()

st.title("📊 Globex Innovations — Financial Q&A Assistant")
st.write("Ask questions about **Globex Innovations** financial data (2023–2024).")

mode = st.sidebar.radio("Choose QA System:", ("RAG (FAISS + Groq)", "Fine-Tuned"))

query = st.text_input("Enter your question:")

if st.button("Get Answer") and query.strip():
    start = time.time()
    if mode.startswith("RAG"):
        answer, score = rag_model.answer_question(query, return_score=True)
        method = "RAG (FAISS + Groq)"
    else:
        answer, score = ft_model.answer_question(query, return_score=True)
        method = "Fine-Tuned QA" if ft_model._loaded else "Fine-Tuned QA"
    elapsed = round((time.time() - start) * 1000, 2)

    st.subheader("Answer")
    st.success(answer)

    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{score:.3f}")
    c2.metric("Method", method)
    c3.metric("Response Time", f"{elapsed} ms")

    with st.expander("🔎 Retrieved Context"):
        try:
            top = rag_model.retriever.retrieve(query, top_k=1)
            if top:
                top = top[0]
                st.info(f"Q: {top['question']}")
                st.write(f"A: {top['answer']}")
                st.caption(f"Similarity score: {top['score']:.3f}")
        except Exception as e:
            st.error(f"Context unavailable: {e}")