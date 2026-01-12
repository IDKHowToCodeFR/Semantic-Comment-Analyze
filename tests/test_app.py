import pytest
import io
import csv
from src.engine import nlp_engine
from src.data import data_handler
from src.engine import topic_modeler

def test_intent_classification():
    result = nlp_engine.classify_intent("The app crashed when I tried to log in.")
    assert "labels" in result
    assert "scores" in result
    assert result["top_intent"] in nlp_engine.get_intent_labels()

def test_sentiment_analysis():
    result = nlp_engine.analyze_sentiment("This is a fantastic update, I love it!")
    assert result["label"] in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    assert 0.0 <= result["score"] <= 1.0

def test_ner_extraction():
    result = nlp_engine.extract_entities("John flew to New York.")
    assert isinstance(result, list)
    if result:
        assert "word" in result[0]
        assert "entity_group" in result[0]

def test_topic_modeling():
    texts = [
        "Login is broken", 
        "Cannot sign in to my account", 
        "The dashboard looks great", 
        "Love the new UI"
    ]
    embeddings = nlp_engine.get_embeddings_batch(texts)
    topics = topic_modeler.discover_topics(embeddings, texts, n_clusters=2)
    assert len(topics) == len(texts)
    assert topics[0] == topics[1]

def test_data_handler_single():
    res = data_handler.process_single_comment("The button doesn't work.")
    assert "intent" in res
    assert "sentiment" in res
    assert "entities" in res

def test_data_handler_csv():
    csv_content = "comment\nI love this app!\nIt crashes sometimes.\n"
    file_obj = io.BytesIO(csv_content.encode('utf-8'))
    
    output_str = data_handler.process_csv(file_obj, target_column="comment")
    
    reader = csv.DictReader(io.StringIO(output_str))
    rows = list(reader)
    
    assert len(rows) == 2
    assert "intent" in rows[0]
    assert "sentiment" in rows[0]
    assert "topic" in rows[0]
    assert "entities" in rows[0]
