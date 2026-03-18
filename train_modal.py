import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import re
import torch
import gc
from torch import nn
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,Trainer,TrainingArguments, EarlyStoppingCallback)

# ============================
# CUDA SETUP
# ============================
torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
print(f"🚀 GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================
# LOAD & CLEAN DATA
# ============================
data = pd.read_csv("train.csv").sample(155000, random_state=42)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-zA-Z ]", "", re.sub(r"http\S+", "", text.lower())).strip()

data["comment_text"] = data["comment_text"].apply(clean_text)
data = data[data["comment_text"].str.len() > 0]

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
X = data["comment_text"]
y = data[labels]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# ============================
# TOKENIZE
# ============================
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

train_enc = tokenizer(X_train.tolist(), truncation=True, padding="max_length", max_length=128)
val_enc = tokenizer(X_val.tolist(), truncation=True, padding="max_length", max_length=128)

del X_train, X_val
gc.collect()
torch.cuda.empty_cache()

# ============================
# DATASET CLASS
# ============================
class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.values

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ToxicDataset(train_enc, y_train)
val_dataset = ToxicDataset(val_enc, y_val)

# ============================
# MODEL (RTX 4070 Super Optimized)
# ============================
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=6,
    problem_type="multi_label_classification"
).to(device)

# ============================
# TRAINER WITH CLASS WEIGHTS
# ============================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").to(device)
        logits = model(**inputs).logits
        
        # Class weights for imbalanced data
        weights = torch.tensor([1.0, 9.0, 2.0, 30.0, 2.0, 10.0], device=device)
        loss = nn.BCEWithLogitsLoss(pos_weight=weights)(logits, labels)
        
        return (loss, model(**inputs)) if return_outputs else loss

# ============================
# TRAINING ARGS (Optimized for RTX 4070 Super 12GB)
# ============================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=48,      # 48 fits in 12GB
    per_device_eval_batch_size=48,
    gradient_accumulation_steps=1,          # No need to accumulate with 12GB
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                                    # RTX 4070 Super supports FP16
    dataloader_pin_memory=True,
    logging_steps=100,
    save_total_limit=1,
    report_to="none",
    remove_unused_columns=False,
)

# ============================
# TRAIN
# ============================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("🔥 Training started...")
trainer.train()

# ============================
# SAVE MODEL
# ============================
trainer.save_model("best_model")
tokenizer.save_pretrained("best_model")
print("✅ Model saved to ./best_model/")

# Cleanup
torch.cuda.empty_cache()
print("✅ Done!")