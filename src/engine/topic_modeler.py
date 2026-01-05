"""Unsupervised topic modeling for batch comments."""

import numpy as np
from sklearn.cluster import KMeans


def discover_topics(embeddings: np.ndarray, texts: list, n_clusters: int = 5) -> list:
    """Cluster texts using pre-computed embeddings and KMeans."""
    valid_indices = []
    for i, t in enumerate(texts):
        if t and str(t).strip() and str(t) != "nan":
            valid_indices.append(i)

    if not valid_indices:
        return ["Unknown"] * len(texts)

    valid_embs = np.array([embeddings[i] for i in valid_indices])

    if len(valid_embs) < n_clusters:
        n_clusters = max(1, len(valid_embs))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(valid_embs)

    results = ["No Topic"] * len(texts)
    for i, emb_idx in enumerate(valid_indices):
        cluster_id = kmeans.labels_[i]
        results[emb_idx] = f"Topic_{cluster_id}"

    return results
