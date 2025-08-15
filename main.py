from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv
import logging
import json
import uuid
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(title="IEM Test Case Generator")

# Simple CORS: allow any origin (works with Netlify)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://testgen-frontend.netlify.app", "http://localhost:5173" ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Testcase(BaseModel):
    id: str
    title: str
    description: str
    steps: str = ""
    expected_result: str = ""
    priority: str = "Medium"
    status: str = "Open"

class GenerateRequest(BaseModel):
    requirement_description: str

# Call Groq API asynchronously
async def call_groq_api(prompt: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}],
    }
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

# Endpoint to generate test cases
@app.post("/generate-testcases")
async def generate_testcases(req: GenerateRequest = Body(...)):
    logging.info(f"Generating testcases for requirement: {req.requirement_description}")

    prompt = f"""
You are a QA engineer. Given the software requirement below, generate 5 example test cases.

Each test case should have the following fields in JSON:
- id: unique string ID (can be numeric or UUID)
- title: short descriptive title
- description: detailed description of what to test
- steps: ordered steps to execute the test
- expected_result: what the expected outcome should be
- priority: High, Medium, or Low
- status: Open, In Progress, or Closed

Return the response as a JSON array of these test case objects ONLY, no extra text.

Requirement:
{req.requirement_description}
"""

    try:
        result = await call_groq_api(prompt)
        content = result["choices"][0]["message"]["content"]

        # Try to parse JSON from AI response
        try:
            testcases_raw = json.loads(content)
            testcases = []
            for tc in testcases_raw:
                testcases.append({
                    "id": tc.get("id") or str(uuid.uuid4()),
                    "title": tc.get("title") or "No Title",
                    "description": tc.get("description") or "",
                    "steps": tc.get("steps") or "",
                    "expected_result": tc.get("expected_result") or "",
                    "priority": tc.get("priority") if tc.get("priority") in ["High", "Medium", "Low"] else "Medium",
                    "status": tc.get("status") if tc.get("status") in ["Open", "In Progress", "Closed"] else "Open",
                })
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(f"Failed to parse JSON from Groq response: {e}")
            testcases = [{
                "id": str(uuid.uuid4()),
                "title": "Parsing Error - raw output",
                "description": content,
                "steps": "",
                "expected_result": "",
                "priority": "Medium",
                "status": "Open"
            }]

        return {"testcases": testcases}

    except Exception as e:
        logging.error(f"Error generating testcases: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
