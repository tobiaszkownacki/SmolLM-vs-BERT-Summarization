# Abstractive vs. Extractive Text Summarization: SmolLM vs. BERT

## Project Overview
This project implements and evaluates two distinct approaches to text summarization using the XSum (Extreme Summarization) dataset. The goal is to compare traditional encoder-only architectures against modern, lightweight decoder-only Large Language Models (LLMs) in the context of generating concise, one-sentence summaries.

The study compares:
1. **Extractive Summarization:** Using BERT to select key sentences from the source.
2. **Abstractive Summarization:** Using SmolLM3-3B (fine-tuned) to generate new content.

## Dataset
The project utilizes the **EdinburghNLP/xsum** dataset, consisting of BBC news articles and their professional one-sentence summaries. Due to hardware constraints, a subset was used for this experiment:
* Training: 5,000 samples
* Validation: 500 samples
* Test: 1,000 samples

## Methodology

### 1. Extractive Approach (BERT)
* **Model:** `bert-large-uncased`
* **Library:** `bert-extractive-summarizer`
* **Technique:** K-means clustering of sentence embeddings to identify the most representative sentences. Two configurations were tested: standard layer usage and hidden layer concatenation.

### 2. Abstractive Approach (SmolLM)
* **Model:** `Hugging FaceTB/SmolLM3-3B`
* **Technique:** Supervised Fine-Tuning (SFT) using Low-Rank Adaptation (LoRA).
* **Optimization:** 8-bit quantization (BitsAndBytes) and `bfloat16` precision to reduce VRAM usage.
* **Prompt Engineering:** Models were tested in Zero-shot, Five-shot, and Fine-tuned settings using a "Professional BBC Editor" system prompt.

#### LoRA Configuration
* Rank (r): 8
* Alpha: 16
* Dropout: 0.05
* Target Modules: all-linear

## Results
The evaluation was performed using ROUGE (lexical overlap) and BERTScore (semantic similarity) metrics. The abstractive approach significantly outperformed the extractive method, as XSum requires heavy paraphrasing rather than simple sentence selection.

| Model | Setup | ROUGE-1 | ROUGE-L | BERTScore (F1) |
| :--- | :--- | :--- | :--- | :--- |
| **BERT** | Extractive (Standard) | 0.1694 | 0.1229 | 0.8548 |
| **SmolLM** | Zero-shot | 0.2359 | 0.1654 | 0.8707 |
| **SmolLM** | **Fine-tuned (LoRA)** | **0.3615** | **0.2891** | **0.9032** |

Notably, the fine-tuned SmolLM3-3B (3B parameters) achieved ROUGE-1 scores comparable to larger models like Llama-3-8B (0.37) on this specific task.
