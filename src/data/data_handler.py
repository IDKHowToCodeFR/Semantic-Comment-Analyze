import csv
import io
from src.engine import nlp_engine


from typing import Generator
import codecs


from src.engine import evaluation


def stream_csv(file_obj, target_column="text") -> Generator[str, None, None]:
    try:
        decoded_file = codecs.iterdecode(file_obj, "utf-8", errors="replace")
        reader = csv.DictReader(decoded_file)
    except Exception as e:
        yield f"Error: Failed to decode file. {e!s}\n"
        return

    if not reader.fieldnames:
        yield "Error: Invalid CSV format or empty file.\n"
        return

    if target_column not in reader.fieldnames:
        yield f"Error: CSV must contain a '{target_column}' column. Found: {', '.join(reader.fieldnames)}\n"
        return

    original_fieldnames_upper = [f.upper() for f in reader.fieldnames]
    reader.fieldnames = original_fieldnames_upper
    target_column_upper = target_column.upper()

    output = io.StringIO()
    new_fields = original_fieldnames_upper + ["INTENT", "CONFIDENCE", "SENTIMENT", "SENTIMENT_SCORE", "TONE", "URGENCY", "RECOMMENDED_ACTION"]
    writer = csv.DictWriter(output, fieldnames=new_fields)
    writer.writeheader()
    yield output.getvalue()
    
    chunk = []
    for row in reader:
        chunk.append(row)
        if len(chunk) == 32:
            yield _process_chunk(chunk, target_column_upper, writer, output)
            chunk = []
            
    if chunk:
        yield _process_chunk(chunk, target_column_upper, writer, output)


def _process_chunk(chunk: list[dict], target_column: str, writer: csv.DictWriter, output: io.StringIO) -> str:
    texts = [row.get(target_column, "") for row in chunk]
    
    embeddings = nlp_engine.get_embeddings_batch(texts)
    intent_results = nlp_engine.classify_intents_batch(texts, embeddings=embeddings)
    sentiment_results = nlp_engine.analyze_sentiments_batch(texts)
    
    for i, row in enumerate(chunk):
        intent = intent_results[i]["top_intent"]
        sentiment = sentiment_results[i]
        
        ctx = evaluation.evaluate_business_context(intent, sentiment)
        
        row["INTENT"] = intent
        row["CONFIDENCE"] = f"{intent_results[i]['top_confidence']:.2f}"
        row["SENTIMENT"] = sentiment["label"]
        row["SENTIMENT_SCORE"] = f"{sentiment['score']:.2f}"
        row["TONE"] = ctx["tone"]
        row["URGENCY"] = ctx["urgency"]
        row["RECOMMENDED_ACTION"] = ctx["recommended_action"]
        
    output.seek(0)
    output.truncate(0)
    writer.writerows(chunk)
    return output.getvalue()
