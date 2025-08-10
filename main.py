from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://meeting-summarizer-frontend.netlify.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY. Please set it in .env")

class AnalysisRequest(BaseModel):
    datasheet_text: str
    log_text: str = None
    query: str = None

async def call_groq_api(datasheet_text: str, log_text: str, query: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    prompt = f"""
    You are an engineering assistant. Given the datasheet content and optional device logs:
    1. Identify relevant commands and register settings.
    2. If logs are provided, match errors with possible causes from the datasheet.
    3. Generate example commands/code to fix or configure the device.
    4. Keep output in a structured JSON format: {{ "commands": [...], "errors": [...], "suggestions": [...] }}

    Datasheet:
    {datasheet_text}

    Logs:
    {log_text if log_text else "No logs provided"}

    User query:
    {query if query else "No specific query"}
    """

    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

@app.post("/analyze_device")
async def analyze_device(req: AnalysisRequest):
    logging.info("Received analysis request")
    try:
        result = await call_groq_api(req.datasheet_text, req.log_text, req.query)
        return {
            "analysis": result["choices"][0]["message"]["content"]
        }
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
