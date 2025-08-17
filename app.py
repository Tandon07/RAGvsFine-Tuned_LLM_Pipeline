# app.py
from config import USE_SYSTEM
from rag_system.generator import RAGGenerator
from fine_tuned_system.model import FineTunedModel

def main():
    q = input("Enter your question: ").strip()
    if not q:
        return print("No input provided")

    if USE_SYSTEM == "rag":
        rag = RAGGenerator()
        ans = rag.answer_question(q)
    elif USE_SYSTEM == "finetuned":
        ft = FineTunedModel()
        ans = ft.answer_question(q)
    else:
        raise ValueError("Invalid USE_SYSTEM in config.py")

    print("\nAnswer:", ans)

if __name__ == "__main__":
    main()