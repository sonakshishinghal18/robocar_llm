from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

SYSTEM_PROMPT = """You control a robot car. Convert the user instruction into a JSON array of commands.
Each command: {"cmd": "F"|"B"|"L"|"R"|"S", "duration": milliseconds}
F=forward, B=backward, L=left turn, R=right turn, S=stop.
Default duration: 1000ms for moves, 600ms for turns, 300ms for stops.
Respond with ONLY the JSON array, no explanation, no markdown backticks.
Example: [{"cmd":"F","duration":2000},{"cmd":"L","duration":600},{"cmd":"S","duration":300}]"""

class Prompt(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "Robocar API online"}

@app.get("/relay")
async def relay(ip: str, v: str):
    """Relay motor command to ESP32 — allows HTTPS page to control HTTP device"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"http://{ip}/cmd?v={v}")
            return Response(content=res.text, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping(ip: str):
    """Check if ESP32 is reachable"""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            res = await client.get(f"http://{ip}/")
            return {"status": "online", "response": res.text}
    except Exception as e:
        return {"status": "offline", "error": str(e)}

@app.post("/drive")
async def drive(p: Prompt):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")

    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": p.text}
                ],
                "temperature": 0.2
            }
        )
        data = res.json()
        content = data["choices"][0]["message"]["content"]
        return {"result": content}
