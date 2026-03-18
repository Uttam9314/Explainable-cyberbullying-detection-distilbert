# ============================
# 0. IMPORTS
# ============================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import re
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

# ============================
# 1. LOAD API KEY SAFELY
# ============================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("=" * 60)
    print("WARNING: GROQ_API_KEY not found in .env file!")
    print("   Will use LOCAL explanations (no API)")
    print("   To enable API:")
    print("   1. Create .env file in project folder")
    print("   2. Add: GROQ_API_KEY=your_key_here")
    print("=" * 60)
    API_AVAILABLE = False
else:
    print("[OK] Groq API Key Loaded Safely")
    API_AVAILABLE = True

# ============================
# 2. LOAD MODEL
# ============================
print("\nLoading Toxic Classifier...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    toxic_model = AutoModelForSequenceClassification.from_pretrained(
        "best_model"
    ).to(device)
    toxic_tokenizer = AutoTokenizer.from_pretrained("best_model")
    print(f"[OK] Model Loaded on {device}")
except Exception as e:
    print(f"[ERROR] Loading model failed: {e}")
    print("   Make sure 'best_model/' folder exists!")
    exit()

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

print("=" * 60)

# ============================
# 3. TEXT CLEANING
# ============================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================
# 4. TOXICITY PREDICTION
# ============================
def predict_toxic(text):
    cleaned = clean_text(text)

    if not cleaned:
        empty_probs = {label: 0.0 for label in labels}
        return False, "none", [], empty_probs

    inputs = toxic_tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    toxic_model.eval()
    with torch.no_grad():
        outputs = toxic_model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().squeeze().numpy()

    if probs.ndim == 0:
        probs = [float(probs)]

    detected = [
        (labels[i], float(probs[i]))
        for i in range(len(labels))
        if probs[i] >= 0.5
    ]
    detected.sort(key=lambda x: x[1], reverse=True)

    all_probs = {
        labels[i]: round(float(probs[i]), 4)
        for i in range(len(labels))
    }

    is_toxic = len(detected) > 0
    primary_label = detected[0][0] if detected else "none"

    return is_toxic, primary_label, detected, all_probs

# ============================
# 5. GROQ API EXPLANATION
# ============================
def get_groq_explanation(comment, primary_label):
    if not API_AVAILABLE:
        return quick_explanation(primary_label)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""Explain why this comment is classified as '{primary_label}' in exactly 7-10 words.

Comment: "{comment}"

Rules:
- Use 7-10 words only
- Be direct and clear
- Explain the toxicity reason

Explanation:"""

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are a content moderation expert. Give brief 7-10 word explanations only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 30
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )

        if response.status_code == 401:
            return "API Error: Invalid API key. Check .env file."
        if response.status_code == 429:
            return "API Error: Rate limit. Wait and try again."
        if response.status_code != 200:
            return f"API Error: Status {response.status_code}."

        result = response.json()
        explanation = result["choices"][0]["message"]["content"].strip()
        explanation = explanation.strip('"').strip("'").strip()
        words = explanation.split()

        if len(words) > 10:
            explanation = " ".join(words[:10])
            if not explanation.endswith("."):
                explanation += "."
        elif len(words) < 3:
            explanation = quick_explanation(primary_label)

        return explanation

    except requests.exceptions.Timeout:
        return "API Timeout. " + quick_explanation(primary_label)
    except requests.exceptions.ConnectionError:
        return "No internet. " + quick_explanation(primary_label)
    except Exception:
        return quick_explanation(primary_label)

# ============================
# 6. LOCAL FALLBACK EXPLANATIONS
# ============================
def quick_explanation(primary_label):
    explanations = {
        "toxic": "Contains harmful and offensive language targeting others.",
        "severe_toxic": "Extremely harmful and abusive content detected.",
        "obscene": "Contains vulgar and inappropriate language.",
        "threat": "Contains direct violence or harm threats.",
        "insult": "Personal attack aimed at demeaning an individual.",
        "identity_hate": "Hate speech targeting a specific group.",
        "none": "Clean comment, no toxicity detected."
    }
    return explanations.get(primary_label, "Content analyzed, no clear category.")

# ============================
# 7. MAIN ANALYSIS FUNCTION
# ============================
def analyze_comment(comment, use_api=True):
    print(f"\n{'=' * 60}")
    print(f"COMMENT: \"{comment}\"")
    print(f"{'=' * 60}")

    # Predict
    is_toxic, primary_label, detected, all_probs = predict_toxic(comment)

    # --- RESULT FIRST (highest probability label) ---
    print(f"\nRESULT: ", end="")
    if is_toxic:
        top_label = detected[0][0]
        top_score = detected[0][1]
        print(f"TOXIC --> {top_label.upper()} ({top_score:.4f})")
        if len(detected) > 1:
            other = ", ".join([f"{d[0]}({d[1]:.4f})" for d in detected[1:]])
            print(f"   Also flagged: {other}")
    else:
        print("NOT TOXIC")

    # --- TOXICITY SCORES (all 6 labels) ---
    print(f"\nTOXICITY SCORES:")
    for label, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        bar_filled = int(prob * 30)
        bar = "#" * bar_filled + "-" * (30 - bar_filled)
        if prob >= 0.5:
            status = "[FLAGGED]"
        else:
            status = "[      ]"
        print(f"  {status} {label:15s}: {prob:.4f} [{bar}]")

    # --- EXPLANATION LAST ---
    print(f"\nEXPLANATION:")
    if use_api and is_toxic and API_AVAILABLE:
        explanation = get_groq_explanation(comment, primary_label)
        print(f"   {explanation}")
    else:
        print(f"   {quick_explanation(primary_label)}")

    print(f"{'=' * 60}")

    return is_toxic, primary_label

# ============================
# 8. INTERACTIVE MODE (No batch test)
# ============================
def main():
    print("\n" + "=" * 60)
    print("   CYBERBULLYING DETECTION SYSTEM")
    print("=" * 60)
    print(f"   Model:  DistilBERT (Multi-Label)")
    print(f"   Device: {device}")
    print(f"   API:    {'Groq Connected' if API_AVAILABLE else 'Local Mode'}")
    print("=" * 60)
    print("\nCommands:")
    print("   Type any comment --> Analyze it")
    print("   'api'            --> Enable API mode")
    print("   'fast'           --> Local mode (no API)")
    print("   'exit'           --> Quit")
    print("=" * 60)

    use_api = API_AVAILABLE

    while True:
        try:
            comment = input("\nYour comment: ").strip()

            if not comment:
                print("Empty input. Type something!")
                continue

            if comment.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            if comment.lower() == "fast":
                use_api = False
                print("Local mode ON (no API calls)")
                continue

            if comment.lower() == "api":
                if API_AVAILABLE:
                    use_api = True
                    print("API mode ON")
                else:
                    print("No API key found. Create .env file first!")
                continue

            analyze_comment(comment, use_api=use_api)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

        except Exception as e:
            print(f"Error: {e}")
            print("Try again with a different comment.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU Memory Cleaned")

# ============================
# 9. RUN
# ============================
if __name__ == "__main__":
    main()