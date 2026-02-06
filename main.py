"""Main entry point for semantic comment analysis application."""

from nlp_engine import load_intent_classifier, load_embedding_model
from interface import create_interface


def main():
    """Initialize models and launch application."""
    print("Loading models...")
    try:
        load_intent_classifier()
        load_embedding_model()
        print("Models loaded successfully")
    except Exception as e:
        print(f"Model pre-load failed: {e}")
        print("Models will load on first use")
    
    app = create_interface()
    app.launch(share=False, show_error=True)


if __name__ == "__main__":
    main()
