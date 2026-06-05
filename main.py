from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import httpx
import os
import json
import uuid
from typing import Dict

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

# WebSocket signaling for WebRTC
connected_peers: Dict[str, WebSocket] = {}

@app.get("/")
def root():
    return {"status": "Robocar API online"}

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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": p.text}
                ],
                "temperature": 0.2
            }
        )
        data = res.json()
        content = data["choices"][0]["message"]["content"]
        return {"result": content}

@app.websocket("/signal")
async def websocket_signal(websocket: WebSocket):
    await websocket.accept()
    peer_id = None
    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "register":
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
