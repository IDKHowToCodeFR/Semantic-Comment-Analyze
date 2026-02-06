"""Gradio interface components for semantic comment analysis."""

from typing import Dict, Tuple, Optional, Any

import pandas as pd
import gradio as gr
import plotly.graph_objects as go

from nlp_engine import classify_intent, analyze_sentiment, compute_similarity
from data_handler import process_batch, get_batch_stats


def analyze_text(
    text: str,
    anchor: str,
    threshold: float
) -> Tuple[Dict[str, float], go.Figure, str]:
    """Perform complete analysis on a single comment."""
    try:
        intent = classify_intent(text, threshold)
        intent_dict = {
            label: score
            for label, score in zip(intent['labels'], intent['scores'])
        }
        
        sentiment = analyze_sentiment(text)
        fig = go.Figure(data=[
            go.Bar(
                x=['Positive', 'Neutral', 'Negative'],
                y=[
                    sentiment['positive'],
                    sentiment['neutral'],
                    sentiment['negative']
                ],
                marker_color=['#2ecc71', '#95a5a6', '#e74c3c'],
                text=[
                    f"{v:.2%}"
                    for v in [
                        sentiment['positive'],
                        sentiment['neutral'],
                        sentiment['negative']
                    ]
                ],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="Sentiment Polarity",
            xaxis_title="Sentiment",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400,
            showlegend=False
        )
        
        similarity = 0.0
        if anchor and anchor.strip():
            similarity = compute_similarity(text, anchor)
        
        summary = f"""## Analysis Summary

**Top Intent:** {intent['top_intent']} ({intent['top_confidence']:.2%})

**Sentiment:**
Positive: {sentiment['positive']:.2%} | Neutral: {sentiment['neutral']:.2%} | Negative: {sentiment['negative']:.2%}
Compound: {sentiment['compound']:.3f}

**Similarity to Anchor:** {similarity:.2%}

**All Intents:**
{chr(10).join([f"- {label}: {score:.2%}" for label, score in zip(intent['labels'], intent['scores'])])}"""
        
        return intent_dict, fig, summary
    except ValueError as e:
        return {}, go.Figure(), f"INPUT ERROR: {e}"
    except Exception as e:
        return {}, go.Figure(), f"ANALYSIS ERROR: {e}"


def process_csv(
    data_file: Optional[Any],
    column: str,
    threshold: float
) -> Tuple[pd.DataFrame, Optional[str], str]:
    """Wrapper for batch processing with error handling."""
    try:
        data, output_path = process_batch(data_file, column, threshold)
        total, valid, avg_conf = get_batch_stats(data)
        
        status = f"""Processing Complete

- Total rows: {total}
- Valid analyses: {valid}
- Average confidence: {avg_conf:.2%}"""
        
        return data.head(100), output_path, status
    except ValueError as e:
        return pd.DataFrame(), None, f"INPUT ERROR: {e}"
    except Exception as e:
        return pd.DataFrame(), None, f"PROCESSING ERROR: {e}"


def create_interface() -> gr.Blocks:
    """Create Gradio application interface."""
    with gr.Blocks(
        title="Semantic Comment Analysis",
        theme=gr.themes.Soft()
    ) as interface:
        gr.Markdown("""# Semantic Comment Analysis
Analyze customer feedback with ML-powered intent classification and sentiment scoring.""")
        
        with gr.Tabs():
            with gr.TabItem("Single Analysis"):
                gr.Markdown(
                    "Analyze individual comments with intent, sentiment, "
                    "and similarity metrics."
                )
                
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Comment",
                            placeholder="Enter comment...",
                            lines=5
                        )
                        anchor_input = gr.Textbox(
                            label="Anchor (Optional)",
                            placeholder="Reference comment...",
                            lines=3
                        )
                        confidence = gr.Slider(
                            0.0,
                            1.0,
                            0.5,
                            0.05,
                            label="Confidence Threshold"
                        )
                        analyze_btn = gr.Button(
                            "Analyze",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=3):
                        intent_out = gr.Label(
                            label="Top 3 Intents",
                            num_top_classes=3
                        )
                        sentiment_plot = gr.Plot(label="Sentiment Distribution")
                        summary = gr.Markdown()
                
                analyze_btn.click(
                    fn=analyze_text,
                    inputs=[text_input, anchor_input, confidence],
                    outputs=[intent_out, sentiment_plot, summary]
                )
            
            with gr.TabItem("Batch Processing"):
                gr.Markdown("Upload CSV and analyze multiple comments at once.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        csv_upload = gr.File(
                            label="Upload CSV",
                            file_types=[".csv"],
                            type="filepath"
                        )
                        column_input = gr.Textbox(
                            label="Column Name",
                            value="comment"
                        )
                        batch_confidence = gr.Slider(
                            0.0,
                            1.0,
                            0.5,
                            0.05,
                            label="Confidence Threshold"
                        )
                        process_btn = gr.Button(
                            "Process",
                            variant="primary",
                            size="lg"
                        )
                        gr.Markdown("**Note:** First row must be headers")
                    
                    with gr.Column(scale=2):
                        status = gr.Markdown("Upload CSV to begin...")
                        preview = gr.Dataframe(
                            label="Preview (First 100)",
                            wrap=True,
                            max_height=400
                        )
                        download = gr.File(label="Download Results")
                
                process_btn.click(
                    fn=process_csv,
                    inputs=[csv_upload, column_input, batch_confidence],
                    outputs=[preview, download, status]
                )
        
        gr.Markdown(
            """---
**Models:** BART-MNLI (intent) | MiniLM-L6-v2 (embeddings)"""
        )
    
    return interface
