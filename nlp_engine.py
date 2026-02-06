"""NLP engine for semantic analysis using transformer models."""

from typing import Dict, Any
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

INTENT_LABELS = [
    "Bug Report",
    "Feature Request",
    "Question",
    "Praise",
    "Complaint",
    "General Feedback"
]


@lru_cache(maxsize=1)
def load_intent_classifier() -> pipeline:
    """Load and cache BART-MNLI for zero-shot classification."""
    try:
        return pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load intent classifier: {e}")


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """Load and cache MiniLM sentence embeddings."""
    try:
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}")


def classify_intent(text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Classify comment intent using zero-shot BART-MNLI."""
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")
    
    classifier = load_intent_classifier()
    result = classifier(text, candidate_labels=INTENT_LABELS, multi_label=False)
    
    filtered = [
        (label, score)
        for label, score in zip(result['labels'], result['scores'])
        if score >= threshold
    ]
    if not filtered:
        filtered = [(result['labels'][0], result['scores'][0])]
    
    return {
        'labels': [item[0] for item in filtered[:3]],
        'scores': [item[1] for item in filtered[:3]],
        'top_intent': filtered[0][0],
        'top_confidence': filtered[0][1]
    }


def analyze_sentiment(text: str) -> Dict[str, float]:
    """Analyze sentiment using embedding similarity to anchor phrases."""
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")
    
    model = load_embedding_model()
    anchors = [
        "This is excellent, amazing, and wonderful!",
        "This is terrible, awful, and horrible!",
        "This is a neutral statement without emotion."
    ]
    
    embeddings = model.encode([text] + anchors)
    text_emb = embeddings[0]
    
    similarities = [
        np.dot(text_emb, embeddings[i]) /
        (np.linalg.norm(text_emb) * np.linalg.norm(embeddings[i]))
        for i in range(1, 4)
    ]
    
    total = sum(similarities)
    return {
        'positive': float(similarities[0] / total),
        'negative': float(similarities[1] / total),
        'neutral': float(similarities[2] / total),
        'compound': float((similarities[0] - similarities[1]) / total)
    }


def compute_similarity(text: str, anchor: str) -> float:
    """Calculate cosine similarity between two texts."""
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty")
    if not anchor or not anchor.strip():
        raise ValueError("Anchor comment cannot be empty")
    
    model = load_embedding_model()
    embeddings = model.encode([text, anchor])
    return float(
        np.dot(embeddings[0], embeddings[1]) /
        (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    )
