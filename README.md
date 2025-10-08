# AlpaCare LoRA Fine-Tuning on LLaMA-1B

This repository provides code, notebooks, and scripts for fine-tuning a **LLaMA-1B** model on the **AlpaCare-MedInstruct-52k** medical instruction dataset using **LoRA (Low-Rank Adaptation)**. The project includes preprocessing, training, evaluation, and inference pipelines.

---

## ⚡ Project Overview

- **Base Model:** LLaMA-1B Instruct  
- **Adapter Method:** LoRA (Low-Rank Adaptation)  
- **Dataset:** [lavita/AlpaCare-MedInstruct-52k](https://huggingface.co/lavita/AlpaCare-MedInstruct-52k)  
- **Training Type:** FP16 fine-tuning  
- **Evaluation:** Human evaluation with rubric and safety scoring  

**Disclaimer:** This model is for research purposes only. Outputs **are not guaranteed to be clinically accurate**. Do not use this model for real medical advice or patient care.

---


---

## 📥 Dataset

- **Source**: Hugging Face
- **Name**: `lavita/AlpaCare-MedInstruct-52k`
- **Fields**: `instruction`, `input`, `output`
- **Splits**:
  - Train: 2000 samples
  - Validation: 100 samples
  - Test: 100 samples

---

## 🧹 Preprocessing

- Removed rows with missing `instruction` or `output`.
- Dropped duplicate samples across `instruction`, `input`, and `output`.
- Cleaned text:
  - Stripped whitespace
  - Collapsed multiple spaces and newlines
  - Standardized quotes (`“” → "`, `‘’ → '`)
  - Removed non-printable/unicode characters
- Filtered out samples with very short instructions/outputs (≤ 5 characters).

---

## 🔢 Tokenization & LoRA Fine-Tuning

**Tokenization:**
- Prompts formatted as:
- Truncated sequences to **max length = 512 tokens**.
- Masked prompt tokens so loss is computed only on response tokens.
- Added EOS token.

**LoRA Fine-Tuning:**
- Config: rank (`r`) = 16, alpha = 32, dropout = 0.05
- Targeted attention projection layers (`q_proj` and `v_proj`)
- Training:
- FP16
- 5 epochs
- Gradient accumulation
- Only adapter weights updated; base model frozen

---

## 🧪 Model Evaluation

- **Metrics**: Accuracy, Relevance, Completeness, Safety (human evaluation)
- Evaluated on **100 test samples** from AlpaCare-MedInstruct-52k
- Optional: Human evaluation spreadsheet processed via Python:
- Computes per-prompt, per-evaluator, and overall average scores
- Processed results saved for reporting

**Disclaimer:** Evaluation outputs are not a substitute for professional medical advice.

---

## 🛡 Safety & Mitigation

- Data filtered to remove low-quality, incomplete, or duplicate samples
- Human evaluation includes safety scoring
- Model outputs should **not be relied upon for real-world medical decisions**
- Research-only use; implement further validation before deployment

---

## 🚀 Usage

### 1. Fine-Tuning (Colab)
1. Open `notebooks/colab-finetune.ipynb`.
2. Install dependencies from `requirements.txt`.
3. Mount Google Drive.
4. Load and preprocess dataset.
5. Fine-tune the model with LoRA.
6. Save adapter weights and tokenizer to Drive.

### 2. Inference
1. Open `inference_demo.ipynb`.
2. Mount Drive and load base model + LoRA adapter.
3. Run sample prompts.
4. Outputs include **disclaimer** in every response.

---

## 📦 Requirements

```txt
transformers==4.44.0
torch>=2.0
peft
datasets
accelerate
bitsandbytes

