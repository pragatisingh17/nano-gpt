# 🚀 NanoGPT from Scratch in PyTorch

A lightweight, character-level implementation of a decoder-only Transformer model inspired by Andrej Karpathy's `nanoGPT` and the foundational *"Attention Is All You Need"* paper.

## 🧠 What This Project Does
This project builds a miniature Large Language Model (LLM) completely from scratch using Python and PyTorch. It reads a text dataset (Tiny Shakespeare), tokenizes characters into numerical IDs, passes them through multi-head causal self-attention layers, and trains the model to generate brand-new, stylized text character by character.

## ⚙️ Key Architectural Components
* **Causal Self-Attention:** Computes Queries ($Q$), Keys ($K$), and Values ($V$) with a lower-triangular mask (`tril`) to prevent the model from looking into the future.
* **Multi-Head Attention:** Runs multiple attention heads in parallel to capture different contextual relationships in text.
* **Transformer Blocks:** Combines communication (attention) and computation (feed-forward neural networks) with Layer Normalization and residual connections ($x + \text{sublayer}(x)$).
* **Text Generation:** Samples from predicted probability distributions using `torch.multinomial` to generate novel text sequences.

## 📁 Repository Structure
* `model.py` — Contains the neural network architecture (`Head`, `MultiHeadAttention`, `FeedForward`, `Block`, and `GPTLanguageModel`).
* `train.py` — Handles dataset downloading, batch generation, the training loop, optimizer configuration, and text generation.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/nano-gpt.git](https://github.com/YOUR_USERNAME/nano-gpt.git)
   cd nano-gpt
