"""Offline intent classification with the Laya English checkpoint.

Usage:
    HF_HUB_OFFLINE=1 python classify_intents.py "I forgot my password"
"""
import os
import sys

import laya

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

# Replace with your own intents: {label: short description}.
INTENTS = {
    "reset_password": "user wants to reset or recover their password",
    "cancel_subscription": "user wants to cancel their plan or subscription",
    "track_order": "user asks where their order or package is",
    "refund_request": "user wants money back for a purchase",
    "talk_to_human": "user asks for a human agent",
    "other": "anything else",
}

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What is the customer's intent?",
        "criteria": INTENTS,
    }
}

# vegaml=False loads the classic encoder checkpoint from MODEL_DIR (no Hub access).
agent = laya.Agent(MODEL_DIR, vegaml=False)


def classify(text):
    # predict_tournament splits large label sets into groups of 16 so every label fits the
    # 512-token budget; with <=16 labels it is a single normal predict call.
    result = laya.predict_tournament(agent, text, QUESTIONS)
    return result["answers"]["intent"]


if __name__ == "__main__":
    texts = sys.argv[1:] or ["I forgot my password and can't log in"]
    for t in texts:
        a = classify(t)
        print(f"{t!r} -> {a['choice']} (confidence {a.get('confidence')})")
