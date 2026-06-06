from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import httpx
import os
import json
from typing import Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY   = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

DRIVE_PROMPT = """You control a robot car. Convert the user instruction into a JSON array of commands.
Each command: {"cmd": "F"|"B"|"L"|"R"|"S", "duration": milliseconds}
F=forward, B=backward, L=left turn, R=right turn, S=stop.
Default duration: 1000ms for moves, 600ms for turns, 300ms for stops.
Respond with ONLY the JSON array, no explanation, no markdown backticks.
Example: [{"cmd":"F","duration":2000},{"cmd":"L","duration":600},{"cmd":"S","duration":300}]"""

VISION_PROMPT = """You are a funny Hindi-speaking robot car. Look at the image. Reply ONLY JSON:
{"cmd":"F/B/L/R/S","duration":300-1500,"narration":"Hindi sentence"}

Rules:
- Clear path → F, 1000ms
- Obstacle ahead → L or R (pick open side), 600ms
- Very close obstacle → S, 500ms
- Narration: 1 short Hindi sentence, first person, mix these styles randomly:
  * Bollywood drama: "अरे बाप रे! सामने दीवार!"
  * Sarcastic: "वाह क्या रास्ता है, बिल्कुल नहीं है!"  
  * Curious: "ये क्या पड़ा है आगे? चलो देखते हैं!"
  * Confident: "रास्ता साफ है भाई, फुल स्पीड!"
- Be creative, funny, dramatic. Never repeat same narration.
- ONLY JSON, no markdown."""

class Prompt(BaseModel):
    text: str

class VisionRequest(BaseModel):
    image: str

connected_peers: Dict[str, WebSocket] = {}

@app.get("/")
def root():
    return {"status": "Robocar API online v1.3.0"}

@app.get("/relay")
async def relay(ip: str, v: str):
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"http://{ip}/cmd?v={v}")
            return Response(content=res.text, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/drive")
async def drive(p: Prompt):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": DRIVE_PROMPT},
                    {"role": "user", "content": p.text}
                ],
                "temperature": 0.2
            }
        )
        data = res.json()
        content = data["choices"][0]["message"]["content"]
        return {"result": content}

@app.post("/vision")
async def vision(req: VisionRequest):
    """Analyze camera frame using Gemini 2.5 Flash-Lite"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    async with httpx.AsyncClient(timeout=10) as client:
        res = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [
                        {"text": VISION_PROMPT},
                        {"inline_data": {"mime_type": "image/jpeg", "data": req.image}}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.9,
                    "maxOutputTokens": 80
                }
            }
        )
        data = res.json()
        print("Gemini:", json.dumps(data)[:300])
        if "candidates" not in data:
            raise HTTPException(status_code=500, detail=f"Gemini error: {json.dumps(data)[:200]}")
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        clean = raw.replace("```json", "").replace("```", "").strip()
        try:
            result = json.loads(clean)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail=f"Non-JSON: {clean[:100]}")
        return result

@app.websocket("/signal")
async def websocket_signal(websocket: WebSocket):
    await websocket.accept()
    peer_id = None
    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            msg_type = data.get("type")
            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif msg_type == "motor":
                for pid, peer_ws in connected_peers.items():
                    if pid != peer_id:
                        try:
                            await peer_ws.send_text(json.dumps({
                                "type": "motor",
                                "cmd": data.get("cmd")
                            }))
                        except:
                            pass
            elif msg_type == "register":
                peer_id = data.get("id")
                connected_peers[peer_id] = websocket
                await websocket.send_text(json.dumps({"type": "registered", "id": peer_id}))
            elif msg_type == "signal":
                target_id = data.get("target")
                if target_id in connected_peers:
                    await connected_peers[target_id].send_text(json.dumps({
                        "type": "signal",
                        "from": peer_id,
                        "data": data.get("data")
                    }))
    except WebSocketDisconnect:
        if peer_id and peer_id in connected_peers:
            del connected_peers[peer_id]
