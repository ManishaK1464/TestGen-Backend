from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Allow CORS from your frontend domain (update after frontend deployed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://meeting-summarizer-frontend.netlify.app/"],  # Replace "*" with your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY. Please set it in .env")

class SummarizeRequest(BaseModel):
    meeting_text: str

async def call_groq_api(meeting_text: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": f"Summarize and give action items:\n{meeting_text}"}
        ]
    }
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    logging.info(f"Received meeting text (first 100 chars): {req.meeting_text[:100]}")
    try:
        result = await call_groq_api(req.meeting_text)
        return result
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error while calling Groq API: {e}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
