# rag_system/data_processor.py
import re
import json
import os
from docx import Document
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

class DataProcessor:
    def __init__(self, raw_path=RAW_DATA_PATH, out_path=PROCESSED_DATA_PATH):
        self.raw_path = raw_path
        self.out_path = out_path
        self.data = []

    def load_docx(self, filepath=None):
        filepath = filepath or self.raw_path
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        doc = Document(filepath)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return text

    def parse_qa(self, text):
        pattern = r"Q:\s*(.*?)\s*A:\s*(.*?)\s*(?=Q:|$)"
        matches = re.findall(pattern, text, re.DOTALL)
        self.data = [{"question": q.strip(), "answer": a.strip()} for q, a in matches]
        return self.data

    def save(self, filepath=None):
        filepath = filepath or self.out_path
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def run(self):
        text = self.load_docx()
        self.parse_qa(text)
        self.save()
        return self.data

if __name__ == "__main__":
    processor = DataProcessor()
    data = processor.run()
    print(f"✅ Processed {len(data)} Q&A pairs → {PROCESSED_DATA_PATH}")