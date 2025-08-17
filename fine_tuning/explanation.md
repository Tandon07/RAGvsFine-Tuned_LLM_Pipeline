---

# Fine-Tuning QA Model with LoRA — Explanation

This script fine-tunes a **Question Answering (QA) model** using **LoRA (Low-Rank Adaptation)** on custom financial Q\&A data. Below is a step-by-step explanation of how the code works.

---

## 1. Dataset Preparation

* The code starts by importing the required libraries: `pandas`, `datasets`, `transformers`, and `peft`.
* It also imports the **raw Q\&A data** from `rag_system.data_processor`.

```python
questions, answers = parse_data(file_content)
```

* Since we are doing **extractive QA**, we need one long context block. The script creates this by joining all answers into a single string.
* Each question-answer pair is then structured into the Hugging Face **QA dataset format**:

```json
{
  "question": "...",
  "context": "...",
  "answers": {"text": ["..."], "answer_start": [<position>]}
}
```

* Finally, the dataset is split into training and test sets using an **90/10 split**.

---

## 2. Tokenization and Preprocessing

* The tokenizer (from `config.BASE_QA_MODEL`) converts text into token IDs.
* During preprocessing, the script carefully computes the **start and end token positions** of each answer in the context.

Why this matters:

* QA models need to know where the answer is located inside the context.
* If the answer is not found inside the truncated context, the positions are set to `(0,0)`.

This ensures the model learns the correct mapping between question → answer span.

---

## 3. Model Setup with LoRA (PEFT)

* Instead of fully fine-tuning the model (which is heavy), we use **LoRA adapters** via the **PEFT library**.

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_lin", "v_lin"], # layers in DistilBERT
    bias="none",
    task_type=TaskType.QUESTION_ANS
)
```

* LoRA trains **only a small set of additional parameters**, while keeping the base model frozen.
* This makes fine-tuning much faster and lighter while still adapting the model to our dataset.

---

## 4. Training

* Training is handled by Hugging Face’s **Trainer API**.

* Key configurations:

  * **Learning rate** = `2e-5`
  * **Batch size** = `8`
  * **Epochs** = `5` (higher since dataset is small)
  * **Weight decay** = `0.01`

* The model trains on the training set and evaluates on the test set after each epoch.

---

## 5. Saving Fine-Tuned Model

* After training, only the **LoRA adapters** (not the full model) are saved:

```python
peft_model.save_pretrained(config.FINE_TUNED_ADAPTER_PATH)
tokenizer.save_pretrained(config.FINE_TUNED_ADAPTER_PATH)
```

* This makes storage and future loading lightweight.
* To use the fine-tuned model later, we just need to load the base model + adapters.

---

## 🔑 Key Takeaways

1. **Dataset creation** → Custom Q\&A pairs structured into Hugging Face format.
2. **Preprocessing** → Maps answers into token positions for supervised training.
3. **LoRA (PEFT)** → Efficient fine-tuning with very few additional parameters.
4. **Trainer API** → Handles training loop, evaluation, and logging.
5. **Adapters saved** → Storage-friendly, easy to re-load and deploy.

This script is a **lightweight yet powerful fine-tuning pipeline** for QA tasks, ideal when working with **small datasets and limited compute**.

---