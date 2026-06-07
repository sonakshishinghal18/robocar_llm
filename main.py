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

VISION_PROMPT = """You are a robot car brain. Camera faces forward (standard 1x lens).
Reply ONLY JSON:
{"cmd":"F/B/L/R","duration":200-1500,"speak":true/false,"narration":"Hindi or empty"}

MOVEMENT RULES:
1. Clear path, no obstacle visible → cmd F, duration 1200-1500
2. Something far ahead (>1 meter away) → cmd F, duration 600-800
3. Obstacle at medium distance (35cm-1m) → decide turn direction:
   - Left 20% of image more open → cmd L, duration 400
   - Right 20% of image more open → cmd R, duration 400
   - Both sides blocked → cmd B, duration 400
4. Obstacle very close (<35cm, fills >60% of frame) → cmd B, duration 300
5. Unsure → cmd B, duration 300
Note: Code automatically stops after every command. Never use S.

SPEAK FIELD — you decide freely:
- speak: true → when you see something new, interesting, funny, or dangerous
  Examples: new object appeared, path changed, something unexpected, direction changed
- speak: false → boring/repetitive (same clear path, same obstacle still there)
- You are free to decide. Be creative about when to talk.

NARRATION (only when speak:true):
- 1 short Hindi sentence in Devanagari, fun personality:
  * Bollywood: "अरे बाप रे! ये तो सोफा है!"
  * Sarcastic: "किसने यहाँ जूता रख दिया भाई?"
  * Curious: "ये क्या चीज़ है? देखते हैं!"
  * Confident: "रास्ता साफ है, चल पड़े!"
  * Funny: "मैं तो फंस गया भाई, पीछे हटता हूं!"
- Name specific objects. Be creative, never repeat.
- When speak:false, set narration to ""

ONLY JSON. No markdown."""

class Prompt(BaseModel):
    text: str

class VisionRequest(BaseModel):
    image: str

class TTSRequest(BaseModel):
    text: str

connected_peers: Dict[str, WebSocket] = {}

@app.get("/")
def root():
    return {"status": "Robocar API online v1.5.0"}

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
    """Analyze camera frame using Gemini 2.5 Flash"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    async with httpx.AsyncClient(timeout=10) as client:
        res = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
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
                    "maxOutputTokens": 300,
                    "thinkingConfig": {
                        "thinkingBudget": 0
                    }
                }
            }
        )
        data = res.json()
        print("Vision:", json.dumps(data)[:300])
        if "candidates" not in data:
            raise HTTPException(status_code=500, detail=f"Gemini error: {json.dumps(data)[:200]}")
        raw = data["candidates"][0]["content"]["parts"][0]["text"]
        clean = raw.replace("```json", "").replace("```", "").strip()
        try:
            result = json.loads(clean)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail=f"Non-JSON: {clean[:100]}")
        return result

@app.post("/tts")
async def tts(req: TTSRequest):
    """Generate Hindi speech audio using Gemini Flash TTS"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")
    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"role": "user", "parts": [{"text": req.text}]}],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": "Puck"
                            }
                        }
                    }
                }
            }
        )
        data = res.json()
        print("TTS:", json.dumps(data)[:200])
        if "candidates" not in data:
            raise HTTPException(status_code=500, detail=f"TTS error: {json.dumps(data)[:200]}")
        audio_data = data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        mime_type = data["candidates"][0]["content"]["parts"][0]["inlineData"]["mimeType"]
        return {"audio": audio_data, "mimeType": mime_type}

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
