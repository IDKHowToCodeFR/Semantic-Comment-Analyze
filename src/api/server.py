import asyncio
import io
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.data import data_handler
from src.engine import evaluation, nlp_engine

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    nlp_engine.get_embedding_model()
    nlp_engine.get_classifier_head()
    nlp_engine.get_ner_model()
    nlp_engine.get_sentiment_model()
    nlp_engine.get_irony_model()
    yield

app = FastAPI(title="Semantic NLP Platform API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str
    threshold: float = 0.5
    include_explanation: bool = False


@app.post("/api/analyze")
async def analyze_text(request: AnalyzeRequest):
    try:
        ml_results = await nlp_engine.analyze_full(
            request.text, 
            request.threshold, 
            request.include_explanation
        )

        # Ponytail Ultra features
        business_context = evaluation.evaluate_business_context(
            ml_results["intent"]["top_intent"], 
            ml_results["sentiment"]
        )

        return {
            **ml_results,
            **business_context,
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
