# ============================
# 0. IMPORTS
# ============================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import re
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("=" * 60)
print("📊 STARTING EVALUATION & PREDICTION PIPELINE")
print("=" * 60)

# ============================
# 1. TEXT CLEANING FUNCTION
# ============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# ============================
# 2. RELOAD SAME DATA & SPLIT
# ============================
data = pd.read_csv("train.csv")
data = data.sample(155000, random_state=42)   # Same as training!
data["comment_text"] = data["comment_text"].apply(clean_text)

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

X = data["comment_text"]
y = data[labels]

# SAME random_state=42 so we get SAME split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

print(f"✅ Data Loaded: {len(X_train)} train, {len(X_val)} val samples")

# ============================
# 3. LOAD SAVED MODEL
# ============================
model = AutoModelForSequenceClassification.from_pretrained("best_model")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model.eval()

print("✅ Saved Model Loaded Successfully!")

# ============================
# 4. PREDICTION FUNCTION
# ============================
def predict_toxicity(text, threshold=0.5):
    cleaned = clean_text(text)
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().numpy()

    results = {}
    for i, label in enumerate(labels):
        results[label] = {
            "score": round(float(probs[i]), 4),
            "flagged": bool(probs[i] >= threshold)
        }

    overall_toxic = any(r["flagged"] for r in results.values())
    max_score = max(probs)

    return {
        "original_text": text,
        "is_toxic": overall_toxic,
        "max_toxicity_score": round(float(max_score), 4),
        "details": results
    }

# ============================
# 5. BATCH PREDICTION ON VAL SET
# ============================
print("\n🔥 Running Predictions on Validation Set...")

all_preds = []
all_probs = []
batch_size = 32

for i in range(0, len(X_val), batch_size):
    batch_texts = X_val.iloc[i:i+batch_size].tolist()

    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).numpy()

    all_probs.append(probs)
    preds = (probs >= 0.5).astype(int)
    all_preds.append(preds)

    if (i // batch_size) % 20 == 0:
        print(f"   Processed {i + len(batch_texts)}/{len(X_val)} samples...")

all_preds = np.vstack(all_preds)
all_probs = np.vstack(all_probs)
y_true = y_val.values

print("✅ Predictions Done!")

# ============================
# 6. OVERALL METRICS
# ============================
print("\n" + "=" * 60)
print("📊 OVERALL METRICS")
print("=" * 60)

subset_acc = accuracy_score(y_true, all_preds)
print(f"\n🎯 Subset Accuracy (Exact Match): {subset_acc:.4f}")

sample_acc = np.mean(np.sum(y_true == all_preds, axis=1) / len(labels))
print(f"🎯 Sample-wise Accuracy:          {sample_acc:.4f}")

f1_micro = f1_score(y_true, all_preds, average="micro")
f1_macro = f1_score(y_true, all_preds, average="macro")
f1_weighted = f1_score(y_true, all_preds, average="weighted")
f1_samples = f1_score(y_true, all_preds, average="samples", zero_division=0)

print(f"\n📈 F1 Score (Micro):    {f1_micro:.4f}")
print(f"📈 F1 Score (Macro):    {f1_macro:.4f}")
print(f"📈 F1 Score (Weighted): {f1_weighted:.4f}")
print(f"📈 F1 Score (Samples):  {f1_samples:.4f}")

precision_micro = precision_score(y_true, all_preds, average="micro")
recall_micro = recall_score(y_true, all_preds, average="micro")
precision_macro = precision_score(y_true, all_preds, average="macro", zero_division=0)
recall_macro = recall_score(y_true, all_preds, average="macro", zero_division=0)

print(f"\n✅ Precision (Micro): {precision_micro:.4f}")
print(f"✅ Recall    (Micro): {recall_micro:.4f}")
print(f"✅ Precision (Macro): {precision_macro:.4f}")
print(f"✅ Recall    (Macro): {recall_macro:.4f}")

try:
    roc_auc_micro = roc_auc_score(y_true, all_probs, average="micro")
    roc_auc_macro = roc_auc_score(y_true, all_probs, average="macro")
    print(f"\n🏆 ROC-AUC (Micro): {roc_auc_micro:.4f}")
    print(f"🏆 ROC-AUC (Macro): {roc_auc_macro:.4f}")
except:
    print("\n⚠️ ROC-AUC could not be computed")

# ============================
# 7. PER-LABEL METRICS
# ============================
print("\n" + "=" * 60)
print("📊 PER-LABEL METRICS")
print("=" * 60)

per_label_data = []

for i, label in enumerate(labels):
    acc = accuracy_score(y_true[:, i], all_preds[:, i])
    f1 = f1_score(y_true[:, i], all_preds[:, i], zero_division=0)
    prec = precision_score(y_true[:, i], all_preds[:, i], zero_division=0)
    rec = recall_score(y_true[:, i], all_preds[:, i], zero_division=0)

    try:
        auc_score = roc_auc_score(y_true[:, i], all_probs[:, i])
    except:
        auc_score = 0.0

    per_label_data.append({
        "Label": label,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc_score, 4)
    })

    print(f"\n📌 {label.upper()}")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {auc_score:.4f}")

metrics_df = pd.DataFrame(per_label_data)
print("\n📋 METRICS SUMMARY TABLE:")
print(metrics_df.to_string(index=False))

# ============================
# 8. CLASSIFICATION REPORT
# ============================
print("\n" + "=" * 60)
print("📋 CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_true, all_preds, target_names=labels, zero_division=0))

# ============================
# 9. CONFUSION MATRICES
# ============================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Confusion Matrix - Per Label", fontsize=16, fontweight='bold')

for idx, label in enumerate(labels):
    ax = axes[idx // 3, idx % 3]
    cm = confusion_matrix(y_true[:, idx], all_preds[:, idx])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Not " + label, label],
                yticklabels=["Not " + label, label])
    ax.set_title(f"{label.upper()}", fontsize=12, fontweight='bold')
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Confusion Matrices Saved!")

# ============================
# 10. F1 SCORE BAR CHART
# ============================
f1_scores = [f1_score(y_true[:, i], all_preds[:, i], zero_division=0) for i in range(len(labels))]

plt.figure(figsize=(10, 6))
colors = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
bars = plt.bar(labels, f1_scores, color=colors, edgecolor='black')

for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

plt.title("F1 Score Per Label", fontsize=14, fontweight='bold')
plt.ylabel("F1 Score")
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("f1_scores.png", dpi=150)
plt.show()
print("✅ F1 Bar Chart Saved!")

# ============================
# 11. ROC CURVES
# ============================
plt.figure(figsize=(10, 8))

for i, label in enumerate(labels):
    try:
        fpr, tpr, _ = roc_curve(y_true[:, i], all_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC={roc_auc_val:.3f})")
    except:
        pass

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.title("ROC Curves", fontsize=14, fontweight='bold')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.show()
print("✅ ROC Curves Saved!")

# ============================
# 12. TEST PREDICTIONS
# ============================
print("\n" + "=" * 60)
print("🧪 SINGLE COMMENT PREDICTIONS")
print("=" * 60)

test_comments = [
    "You are so stupid and ugly, I hate you!",
    "Thank you for helping me with my homework.",
    "I will kill you if you don't shut up",
    "Great video, very informative!",
    "You're a worthless piece of garbage, die!",
    "Let's meet at the park tomorrow.",
    "All people from that country are terrorists",
    "I love this community, everyone is so supportive!"
]

for comment in test_comments:
    result = predict_toxicity(comment)
    print(f"\n💬 \"{comment}\"")
    print(f"   🚨 Toxic: {'YES ❌' if result['is_toxic'] else 'NO ✅'}")
    print(f"   📊 Max Score: {result['max_toxicity_score']}")
    for label, info in result["details"].items():
        flag = "🔴" if info["flagged"] else "🟢"
        print(f"      {flag} {label:15s} → {info['score']:.4f}")
    print("-" * 50)

# ============================
# 13. SAVE TO CSV
# ============================
records = []
for comment in test_comments:
    r = predict_toxicity(comment)
    record = {"comment": comment, "is_toxic": r["is_toxic"], "max_score": r["max_toxicity_score"]}
    for label, info in r["details"].items():
        record[f"{label}_score"] = info["score"]
    records.append(record)

pd.DataFrame(records).to_csv("predictions_output.csv", index=False)
print("\n✅ Predictions Saved: predictions_output.csv")

# ============================
# 14. FINAL SUMMARY
# ============================
print("\n" + "=" * 60)
print("🏆 FINAL SUMMARY")
print("=" * 60)
print(f"""
🎯 Subset Accuracy:    {subset_acc:.4f}
📈 F1 (Micro):         {f1_micro:.4f}
📈 F1 (Macro):         {f1_macro:.4f}
✅ Precision (Micro):  {precision_micro:.4f}
✅ Recall (Micro):     {recall_micro:.4f}
""")
print("✅ ALL DONE! 🎉")