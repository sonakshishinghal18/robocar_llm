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

VISION_PROMPT = """You are a robot car brain. Camera faces forward only — you cannot see left/right directly.
Reply ONLY JSON: {"cmd":"F/B/L/R","duration":800,"narration":"Hindi"}
Note: Never use S — code automatically stops after every command.

DECISION RULES (follow in order):

1. CLEAR PATH → F
   Floor visible for >50% of bottom half, no object blocking center → cmd F

2. OBSTACLE AHEAD → use image EDGES to decide turn direction:
   - Look at LEFT edge of image: is it open/bright/floor visible?
   - Look at RIGHT edge of image: is it open/bright/floor visible?
   - More open on LEFT edge → cmd L
   - More open on RIGHT edge → cmd R
   - Both edges blocked → cmd B (back up to create space)

3. VERY CLOSE OBSTACLE (fills >70% of frame) → cmd S

4. STUCK/UNSURE → cmd B (backing up always creates new options)

5. NEVER stay stopped — after S, always follow with L, R or B next call

DURATION: Always 300ms for all commands.

NARRATION: 1 short Hindi sentence, mix randomly:
- Bollywood: "अरे बाप रे! सामने कुर्सी है, दाएं जाता हूं!"
- Sarcastic: "किसने यहाँ सोफा रख दिया? बाएं निकलता हूं!"
- Curious: "दाईं तरफ जगह दिख रही है, चलो वहाँ!"
- Confident: "रास्ता साफ है, फुल स्पीड आगे!"
- Name the object you see (chair, wall, sofa, door, table, person, box)

ONLY JSON. No markdown."""

class Prompt(BaseModel):
    text: str

class VisionRequest(BaseModel):
    image: str

connected_peers: Dict[str, WebSocket] = {}

@app.get("/")
def root():
    return {"status": "Robocar API online v1.3.2"}

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
