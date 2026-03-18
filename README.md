# Cyberbullying Detection System
### Multi-Label Toxic Comment Classification using DistilBERT + Explainable AI

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-91.72%25-green)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.99-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What Does This Project Do?

This AI system reads any online comment and instantly detects:

| Label | Example |
|-------|---------|
| `toxic` | General harmful language |
| `severe_toxic` | Extreme abuse |
| `obscene` | Vulgar language |
| `threat` | "I will find you" |
| `insult` | Personal attacks |
| `identity_hate` | Hate speech against groups |

One comment can have **multiple labels at once**.  
The system also **explains why** a comment was flagged — in plain English.
```
Input:  "You are so stupid and ugly, I hate you!"
Output: TOXIC (0.92) | INSULT (0.85) | OBSCENE (0.62)
Why:    "Personal attack using degrading language targeting individual."
```

---

## Technical Overview

| Detail | Value |
|--------|-------|
| Model | DistilBERT (distilbert-base-uncased) — 66M parameters |
| Task | Multi-label classification (6 labels simultaneously) |
| Dataset | Jigsaw Toxic Comment Classification (155,000 samples) |
| Loss | BCEWithLogitsLoss with class weights [1, 9, 2, 30, 2, 10] |
| Optimizer | AdamW (lr=2e-5, weight_decay=0.01) |
| Scheduler | Linear warmup (10% of steps) |
| Training | 3 epochs, early stopping patience=2, FP16 |
| Hardware | NVIDIA RTX 4070 Super (12GB VRAM) |
| Batch Size | 48 per device |
| Max Length | 128 tokens |
| Explainability | Groq API (llama-3.1-8b-instant) + local fallback |

---

## Pipeline
```
INPUT TEXT
    |
    v
+----------------------+
| clean_text()         |  lowercase, remove URLs, remove special chars
+----------------------+
    |
    v
+----------------------+
| DistilBERT Tokenizer |  WordPiece, [CLS]+tokens+[SEP], max_len=128
+----------------------+
    |
    v
+----------------------+
| 6 Transformer Layers |  12 attention heads each, 768-dim hidden states
| (MHA + FFN)          |
+----------------------+
    |
    v
+----------------------+
| [CLS] Token Pooling  |  768-dim sequence representation
+----------------------+
    |
    v
+----------------------+
| Linear (768 → 6)     |  one logit per toxicity label
| + Sigmoid()          |  independent probability per label (0.0 – 1.0)
+----------------------+
    |
    v
6 SCORES → flagged if score >= 0.5
    |
    v
+----------------------+
| Groq LLM Explanation |  "Why is this toxic?" in 7-10 words
| (+ local fallback)   |
+----------------------+
    |
    v
FINAL OUTPUT: {is_toxic, labels, scores, explanation}

Tensor shapes:
  Input     → [batch, 128]
  Embedding → [batch, 128, 768]
  CLS Token → [batch, 768]
  Output    → [batch, 6]
```

---

## Project Structure
```
cyberbullying-detection/
├── train_modal.py        # Training pipeline (GPU-optimized)
├── predict.py            # Evaluation + all charts/metrics
├── llm.py                # Real-time prediction + LLM explanations
├── best_model/           # Saved fine-tuned DistilBERT weights
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer_config.json
├── .env                  # GROQ_API_KEY (never commit this!)
├── .gitignore
├── requirements.txt
├── train.csv             # Jigsaw dataset (download separately)
├── confusion_matrices.png
├── roc_curves.png
├── f1_scores.png
├── metrics_heatmap.png
└── final_dashboard.png
```

---

## Setup

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/cyberbullying-detection.git
cd cyberbullying-detection
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies
```bash
pip install torch transformers pandas scikit-learn matplotlib seaborn python-dotenv requests
```

### Step 4 — Download the dataset
Download `train.csv` from [Kaggle Jigsaw Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) and place it in the project root.

### Step 5 — Setup Groq API (optional, for explanations)
```bash
# Create .env file in project root:
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at: https://console.groq.com

---

## How to Run

### Train the model (requires GPU)
```bash
python train_modal.py
# Trains for 3 epochs (~20 min on RTX 4070 Super)
# Saves best model to ./best_model/
```

### Evaluate + generate all charts
```bash
python predict.py
# Outputs: confusion_matrices.png, roc_curves.png,
#          f1_scores.png, metrics_heatmap.png, final_dashboard.png
```

### Real-time interactive prediction
```bash
python llm.py
# Type any comment → instant classification + AI explanation
# Commands:
#   'api'  → enable Groq LLM mode
#   'fast' → local mode (no API)
#   'exit' → quit
```

---

## Results

### Overall Metrics

| Metric | Score |
|--------|-------|
| Subset Accuracy (Exact Match) | **91.72%** |
| Hamming Loss | 0.0184 |
| F1 Micro | **0.7610** |
| F1 Macro | 0.6791 |
| Precision Micro | 0.7132 |
| Recall Micro | 0.8157 |
| ROC-AUC Micro | **0.9901** |
| ROC-AUC Macro | 0.9896 |

### Per-Label Metrics

| Label | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| toxic | 0.87 | 0.76 | **0.81** | 0.984 |
| severe_toxic | 0.34 | **0.94** | 0.50 | 0.994 |
| obscene | 0.82 | 0.87 | **0.85** | 0.992 |
| threat | 0.47 | 0.77 | 0.58 | 0.987 |
| insult | 0.64 | 0.85 | 0.73 | 0.988 |
| identity_hate | 0.48 | 0.82 | 0.60 | 0.993 |

> **Note:** Low precision on `severe_toxic` and `threat` is **intentional**.  
> In content moderation, missing a real threat (false negative) is far worse  
> than a false alarm. Class weights are designed to maximize recall on  
> dangerous minority categories.

### vs Baseline

| Method | F1-Micro | ROC-AUC |
|--------|----------|---------|
| Keyword Blacklist | ~0.35 | ~0.70 |
| TF-IDF + Logistic Regression | 0.63 | 0.89 |
| **DistilBERT (ours)** | **0.76** | **0.99** |

---

## Limitations

- English only — no multilingual support yet
- Context-blind: classifies single comments, no conversation history
- Adversarial bypass: character substitution (st\*pid) can fool the model
- LLM explanations are post-hoc, not mechanistic token-level XAI
- Static model — does not learn from new data without retraining

---

## Future Work

- [ ] Per-label threshold optimization via Precision-Recall curves
- [ ] XLM-RoBERTa for multilingual (100+ languages) support
- [ ] SHAP / attention visualization for true token-level explainability
- [ ] FastAPI + Docker deployment for production use
- [ ] Active learning from moderator feedback
- [ ] Adversarial training with leetspeak / obfuscation examples
- [ ] Drift detection and automated monitoring

---

## Tech Stack

![DistilBERT](https://img.shields.io/badge/DistilBERT-HuggingFace-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-red)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-orange)
![Groq](https://img.shields.io/badge/Groq-LLaMA--3.1-purple)
![Seaborn](https://img.shields.io/badge/Seaborn-Matplotlib-blue)

---

## License

MIT License — free to use, modify, and distribute.

---

*Built with PyTorch + HuggingFace Transformers | Dataset: Jigsaw Toxic Comment Classification Challenge*
