# Semantic Comment Analysis

A production-ready application for semantic analysis of customer feedback using transformer-based NLP models. Provides automated intent classification, sentiment analysis, and semantic similarity scoring.

## Overview

This application leverages state-of-the-art natural language processing to analyze customer comments, support tickets, or social media feedback. It classifies intents into predefined categories, scores sentiment polarity, and computes semantic similarity against reference comments.

### Key Features

- **Intent Classification**: Zero-shot classification using BART-MNLI to identify comment types (Bug Report, Feature Request, Question, Praise, Complaint, General Feedback)
- **Sentiment Analysis**: Embedding-based sentiment scoring with positive, neutral, and negative components
- **Similarity Computation**: Cosine similarity measurement between comments using sentence embeddings
- **Dual Processing Modes**: Single text analysis and batch CSV processing
- **Interactive UI**: Web-based Gradio interface for easy interaction

## Architecture

The application follows a modular design pattern with clear separation of concerns:

```
Semantic/
├── nlp_engine.py      # NLP model management and inference
├── data_handler.py    # CSV processing and batch operations
├── interface.py       # Gradio UI components and callbacks
├── main.py           # Application entry point
├── requirements.txt  # Python dependencies
└── sample_comments.csv  # Example test data
```

### Module Responsibilities

**nlp_engine.py**
- Model loading and caching (BART-MNLI, MiniLM)
- Intent classification logic
- Sentiment analysis computation
- Semantic similarity calculation

**data_handler.py**
- CSV file validation and parsing
- Batch processing workflows
- Result aggregation and statistics

**interface.py**
- Gradio component definitions
- UI layout and styling
- Callback function orchestration

**main.py**
- Application initialization
- Model pre-loading
- Interface launch configuration

## Setup

### Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for batch processing)
- Internet connection for initial model downloads

### Installation

1. Clone or download the project directory

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** First-time setup will download transformer models (approximately 1.7GB total). Models are cached locally for subsequent runs.

## Usage

### Starting the Application

```bash
python main.py
```

The application will:
1. Load ML models into memory
2. Launch a local web server
3. Open automatically at `http://127.0.0.1:7860`

### Single Text Analysis

1. Navigate to the "Single Analysis" tab
2. Enter the comment text to analyze
3. (Optional) Enter a reference "anchor" comment for similarity comparison
4. Adjust the confidence threshold slider (default: 0.5)
5. Click "Analyze" to process

**Output:**
- Top 3 predicted intents with confidence scores
- Interactive sentiment polarity chart
- Detailed analysis summary with all metrics

### Batch CSV Processing

1. Navigate to the "Batch Processing" tab
2. Upload a CSV file (first row must contain column headers)
3. Specify the column name containing comments (e.g., "comment", "text", "feedback")
4. Adjust the confidence threshold if needed
5. Click "Process" to analyze all rows

**Output:**
- Preview of first 100 processed rows
- Downloadable CSV with appended analysis columns:
  - `Sentiment_Score`: Compound sentiment (-1 to 1)
  - `Predicted_Intent`: Primary intent category
  - `Confidence`: Classification confidence (0 to 1)
  - `Positive_Score`: Positive sentiment component
  - `Neutral_Score`: Neutral sentiment component
  - `Negative_Score`: Negative sentiment component

### CSV Format Example

```csv
comment,user_id
The app crashes when I try to export data,user_001
Love the new design! Much cleaner interface,user_002
How do I reset my password?,user_003
```

## Technical Details

### Models

**Intent Classification:**
- Model: `facebook/bart-large-mnli`
- Task: Zero-shot classification
- Size: ~1.6GB
- Inference: CPU-optimized

**Semantic Embeddings:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Task: Sentence embeddings
- Size: ~90MB
- Inference: CPU-optimized

### Performance

- **First run:** 2-3 minutes (model downloads)
- **Subsequent runs:** 10-15 seconds (cached models)
- **Single analysis:** <1 second per comment
- **Batch processing:** ~2-3 seconds per comment

### Customization

**Modifying Intent Labels:**

Edit `INTENT_LABELS` in `nlp_engine.py`:
```python
INTENT_LABELS = [
    "Your Custom",
    "Intent Categories",
    "Go Here"
]
```

**Changing Sentiment Anchors:**

Edit anchor phrases in `analyze_sentiment()` function in `nlp_engine.py`.

## Development

### Project Structure Details

The modular architecture enables:
- Independent testing of components
- Easy swapping of ML models
- UI redesign without touching business logic
- Parallel development across modules

### Adding New Features

**New Analysis Type:**
1. Add inference function to `nlp_engine.py`
2. Update `interface.py` to display results
3. Modify `data_handler.py` if CSV processing needed

**UI Modifications:**
1. Edit only `interface.py`
2. Business logic remains untouched

### Code Quality

- PEP 8 compliant formatting
- Type hints on all function signatures
- Comprehensive docstrings
- Modular, testable architecture

## Troubleshooting

**Models fail to load:**
- Ensure stable internet connection
- Check available disk space (2GB minimum)
- Try manual download: `huggingface-cli download facebook/bart-large-mnli`

**CSV processing errors:**
- Verify first row contains headers
- Check column name spelling
- Ensure CSV is properly formatted (UTF-8 encoding recommended)

**Performance issues:**
- Reduce batch size for large CSV files
- Close other memory-intensive applications
- Consider GPU acceleration for large-scale deployments

## License

This project is provided as-is for educational and commercial use.

## Support

For issues or questions, verify:
1. All dependencies installed correctly
2. Python version 3.8+
3. Models downloaded successfully
4. CSV file format matches requirements

---

**Last Updated:** 2026-02-06  
**Version:** 2.0.0 (Modular Architecture)
