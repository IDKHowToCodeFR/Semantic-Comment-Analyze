"""NLP engine for semantic analysis using transformer models."""

import os
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

DEFAULT_INTENT_LABELS = [
    "Bug Report",
    "Complaint",
    "Feature Request",
    "Praise",
    "Question",
]


def get_intent_labels() -> list[str]:
    mapping_path = os.path.join(os.path.dirname(__file__), "label_mapping.txt")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return DEFAULT_INTENT_LABELS


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_classifier_head():
    head_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../data/local_model_head.pkl")
    )
    if not os.path.exists(head_path):
        raise RuntimeError(
            "data/local_model_head.pkl not found. Run train_model.py first."
        )
    return joblib.load(head_path)


@lru_cache(maxsize=1)
def get_ner_model():
    return pipeline("ner", aggregation_strategy="simple", device=-1)


@lru_cache(maxsize=1)
def get_sentiment_model():
    return pipeline("sentiment-analysis", device=-1)


def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    # SentenceTransformer handles empty strings gracefully
    safe_texts = [str(t) if t else "" for t in texts]
    return model.encode(safe_texts, batch_size=32)


def classify_intent(text: str, threshold: float = 0.5) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Input empty")
    return classify_intents_batch([text], threshold)[0]


def classify_intents_batch(
    texts: list[str], threshold: float = 0.5, embeddings=None
) -> list[dict[str, Any]]:
    valid_texts, valid_indices = [], []
    for i, t in enumerate(texts):
        if t and str(t).strip() and str(t) != "nan":
            valid_texts.append(str(t))
            valid_indices.append(i)

    results = [
        {
            "labels": ["INVALID_INPUT"],
            "scores": [0.0],
            "all_intents": {},
            "top_intent": "INVALID_INPUT",
            "top_confidence": 0.0,
        }
        for _ in texts
    ]
    if not valid_texts:
        return results

    clf = get_classifier_head()
    labels = clf.classes_

    if embeddings is not None:
        embs = np.array([embeddings[i] for i in valid_indices])
    else:
        model = get_embedding_model()
        embs = model.encode(valid_texts, batch_size=32)

    probs_batch = clf.predict_proba(embs)

    for idx, probs in zip(valid_indices, probs_batch):
        sorted_pairs = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
        filtered = [p for p in sorted_pairs if p[1] >= threshold]
        if not filtered:
            filtered = [sorted_pairs[0]]

        results[idx] = {
            "labels": [str(item[0]) for item in filtered[:3]],
            "scores": [float(item[1]) for item in filtered[:3]],
            "all_intents": {str(item[0]): float(item[1]) for item in sorted_pairs},
            "top_intent": str(filtered[0][0]),
            "top_confidence": float(filtered[0][1]),
        }
    return results


def explain_intent(text: str) -> list[dict[str, Any]]:
    words = text.split()
    if not words:
        return []

    orig_res = classify_intents_batch([text])[0]
    orig_intents = orig_res.get("all_intents", {})

    masked_texts = []
    for i in range(len(words)):
        masked = words[:i] + words[i + 1 :]
        masked_texts.append(" ".join(masked))

    masked_results = classify_intents_batch(masked_texts)

    contributions = []
    for i, res in enumerate(masked_results):
        masked_intents = res.get("all_intents", {})

        best_intent = None
        max_drop = 0.0

        for intent, orig_prob in orig_intents.items():
            masked_prob = masked_intents.get(intent, 0.0)
            drop = orig_prob - masked_prob
            if drop > max_drop:
                max_drop = drop
                best_intent = intent

        contributions.append(
            {
                "word": words[i],
                "contribution": float(max_drop),
                "primary_intent": best_intent if best_intent else "None",
            }
        )

    return contributions


def analyze_sentiment(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}
    return get_sentiment_model()(text, truncation=True, max_length=512)[0]


def analyze_sentiments_batch(texts: list[str]) -> list[dict[str, Any]]:
    valid_texts = [
        str(t) if t and str(t).strip() and str(t) != "nan" else "Neutral" for t in texts
    ]
    model = get_sentiment_model()
    return model(valid_texts, truncation=True, max_length=512, batch_size=32)


def extract_entities(text: str) -> list[dict[str, Any]]:
    if not text or not text.strip():
        return []
    return get_ner_model()(text[:2000])


def extract_entities_batch(texts: list[str]) -> list[list[dict[str, Any]]]:
    valid_texts = [
        str(t) if t and str(t).strip() and str(t) != "nan" else "" for t in texts
    ]
    ner = get_ner_model()
    # Batch process ignoring empty strings where possible
    batch_results = ner([t[:2000] for t in valid_texts], batch_size=32)
    return batch_results
