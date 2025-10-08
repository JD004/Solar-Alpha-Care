# AlpaCare LoRA Fine-Tuning on LLaMA-1B

This repository contains code, notebooks, and scripts for fine-tuning a LLaMA-1B model on the **AlpaCare-MedInstruct-52k** medical instruction dataset using **LoRA (Low-Rank Adaptation)**. It also includes inference scripts and evaluation pipelines.

---

## ⚡ Project Overview

- **Base Model**: LLaMA-1B Instruct
- **Adapter Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: [lavita/AlpaCare-MedInstruct-52k](https://huggingface.co/lavita/AlpaCare-MedInstruct-52k)
- **Training Type**: FP16 fine-tuning
- **Inference**: Evaluate on sample prompts with the LoRA adapter applied.

**Disclaimer:** This model and dataset are intended for research purposes only. Outputs are **not guaranteed to be clinically accurate**. Do not rely on this model for medical advice or patient care.

---

## 🗂 Repository Structure

