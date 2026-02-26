import io
import logging
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.data import data_handler
from src.engine import evaluation, nlp_engine

logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic NLP Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str
    threshold: float = 0.5


@app.post("/api/analyze")
def analyze_text(request: AnalyzeRequest):
    try:
        intent = nlp_engine.classify_intent(request.text, request.threshold)
        sentiment = nlp_engine.analyze_sentiment(request.text)
        explainability = nlp_engine.explain_intent(request.text)

        # Ponytail Ultra features
        tone = evaluation.analyze_tone(intent["top_intent"], sentiment)
        urgency = evaluation.calculate_urgency(intent["top_intent"], sentiment)
        action = evaluation.recommend_action(intent["top_intent"], urgency, sentiment)

        return {
            "intent": intent,
            "sentiment": sentiment,
            "explainability": explainability,
            "tone": tone,
            "urgency": urgency,
            "recommended_action": action,
        }
    except Exception as e:
        logger.exception("Error during analysis")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/batch")
def process_batch(
    file: UploadFile = File(...),  # noqa: B008
    threshold: float = Form(0.5),
    targetColumn: str = Form("text"),
):
    try:
        generator = data_handler.stream_csv(file.file, target_column=targetColumn)
        return StreamingResponse(generator, media_type="text/csv")
    except Exception as e:
        logger.exception("Error during batch processing")
        raise HTTPException(status_code=500, detail=str(e))


# Mount frontend
frontend_dist = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../frontend/dist"
)
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    logger.warning("Frontend dist not found. Please build the frontend.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
