"""Data handling utilities for CSV processing and batch analysis."""

from typing import Tuple, Optional, Any

import pandas as pd

from nlp_engine import classify_intent, analyze_sentiment


def process_batch(
    data_file: Optional[Any],
    target_column: str,
    threshold: float
) -> Tuple[pd.DataFrame, str]:
    """Process CSV file with semantic analysis on specified column."""
    if data_file is None:
        raise ValueError("No CSV file uploaded")
    
    try:
        data = pd.read_csv(data_file.name)
    except pd.errors.ParserError as e:
        raise ValueError(f"Malformed CSV: {e}")
    
    if data.empty:
        raise ValueError("CSV file is empty")
    if target_column not in data.columns:
        available = ', '.join(data.columns)
        raise ValueError(
            f"Column '{target_column}' not found. Available: {available}"
        )
    
    data['Sentiment_Score'] = 0.0
    data['Predicted_Intent'] = ""
    data['Confidence'] = 0.0
    data['Positive_Score'] = 0.0
    data['Neutral_Score'] = 0.0
    data['Negative_Score'] = 0.0
    
    for idx, row in data.iterrows():
        comment = str(row[target_column])
        
        if not comment or not comment.strip() or comment == 'nan':
            data.at[idx, 'Predicted_Intent'] = "INVALID_INPUT"
            continue
        
        try:
            intent = classify_intent(comment, threshold)
            data.at[idx, 'Predicted_Intent'] = intent['top_intent']
            data.at[idx, 'Confidence'] = intent['top_confidence']
            
            sentiment = analyze_sentiment(comment)
            data.at[idx, 'Sentiment_Score'] = sentiment['compound']
            data.at[idx, 'Positive_Score'] = sentiment['positive']
            data.at[idx, 'Neutral_Score'] = sentiment['neutral']
            data.at[idx, 'Negative_Score'] = sentiment['negative']
        except Exception as e:
            error_msg = str(e)[:50]
            data.at[idx, 'Predicted_Intent'] = f"ERROR: {error_msg}"
    
    output_path = "processed_comments.csv"
    data.to_csv(output_path, index=False)
    return data, output_path


def get_batch_stats(data: pd.DataFrame) -> Tuple[int, int, float]:
    """Calculate statistics for processed batch data."""
    total = len(data)
    valid = len(data[data['Predicted_Intent'] != 'INVALID_INPUT'])
    avg_conf = data[data['Confidence'] > 0]['Confidence'].mean()
    return total, valid, avg_conf
