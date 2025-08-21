# config.py
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Default system for CLI
USE_SYSTEM = "rag"   # options: "rag", "finetuned"

# Data paths
RAW_DATA_PATH = "data/raw/Company_data.docx"
PROCESSED_DATA_PATH = "data/processed/qa_pairs.json"

# Fine-tuned model path
FINETUNED_MODEL_PATH = "fine_tuned_system/distilbert-qa-lora"

# FAISS embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "artifacts/rag/faiss_index"

# Groq LLM
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # loaded from .env
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_API_KEY=st.secrets['GROQ_API_KEY']

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Please set it in your .env file.")
