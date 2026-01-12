import csv
import io
from src.engine import nlp_engine, topic_modeler


def process_single_comment(comment: str) -> dict:
    if not comment or not comment.strip():
        return {"error": "Empty comment"}

    intent_result = nlp_engine.classify_intent(comment)
    sentiment_result = nlp_engine.analyze_sentiment(comment)
    entities = nlp_engine.extract_entities(comment)

    return {
        "intent": intent_result["top_intent"],
        "confidence": intent_result["top_confidence"],
        "sentiment": sentiment_result["label"],
        "sentiment_score": sentiment_result["score"],
        "entities": [e["word"] for e in entities],
    }


def process_csv(file_obj, target_column="text") -> str:
    try:
        text_data = file_obj.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return f"Error: Failed to decode file. {e!s}"

    reader = csv.DictReader(io.StringIO(text_data))

    if not reader.fieldnames:
        return "Error: Invalid CSV format or empty file."

    if target_column not in reader.fieldnames:
        return f"Error: CSV must contain a '{target_column}' column. Found: {', '.join(reader.fieldnames)}"

    rows = list(reader)
    if not rows:
        return "Error: CSV is empty."

    texts = [row.get(target_column, "") for row in rows]

    # Compute embeddings exactly ONCE
    embeddings = nlp_engine.get_embeddings_batch(texts)

    intent_results = nlp_engine.classify_intents_batch(texts, embeddings=embeddings)
    topics = topic_modeler.discover_topics(embeddings, texts)

    sentiment_results = nlp_engine.analyze_sentiments_batch(texts)
    entities_results = nlp_engine.extract_entities_batch(texts)

    for i, row in enumerate(rows):
        row["intent"] = intent_results[i]["top_intent"]
        row["confidence"] = intent_results[i]["top_confidence"]
        row["sentiment"] = sentiment_results[i]["label"]
        row["sentiment_score"] = sentiment_results[i]["score"]
        row["entities"] = ", ".join([e["word"] for e in entities_results[i]])
        row["topic"] = topics[i]

    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return output.getvalue()
