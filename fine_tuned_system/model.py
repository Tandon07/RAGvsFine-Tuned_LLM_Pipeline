# fine_tuned_system/model.py
import os
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from rag_system.retriever_faiss import FAISSRetriever
from config import FINETUNED_MODEL_PATH

class FineTunedModel:
    def __init__(self, model_path=FINETUNED_MODEL_PATH):
        self.model_path = model_path
        self._loaded = False
        self._init_model_or_fallback()

    def _init_model_or_fallback(self):
        if os.path.exists(self.model_path) and os.listdir(self.model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_path)
            self.qa_pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)
            self._loaded = True
        else:
            self.qa_pipeline = None
            self.retriever = FAISSRetriever()
            self._loaded = False

    def answer_question(self, question, context=None, return_score=False):
        if self._loaded:
            context = context or "Globex Innovations financial data for 2023 and 2024."
            res = self.qa_pipeline(question=question, context=context)
            answer, score = res.get("answer", ""), float(res.get("score", 0.0))
        else:
            retrieved = self.retriever.retrieve(question, top_k=1)
            if not retrieved:
                return ("No data available", 0.0) if return_score else "No data available"
            top = retrieved[0]
            answer, score = top["answer"], float(top.get("score", 0.0))
        return (answer, score) if return_score else answer