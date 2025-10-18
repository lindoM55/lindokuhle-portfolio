from typing import Optional, Union, List, Dict, Any
import os
import asyncio
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


class OllamaRequest(BaseModel):
    prompt: Optional[str] = None
    # Using the installed Gemma model
    model: Optional[str] = "gemma3:270m"
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7  # Add some creativity while staying focused
    # You can send a chat-style payload with messages if your Ollama model supports it
    messages: Optional[List[Dict[str, Any]]] = None

# Website context for the AI to use
WEBSITE_CONTEXT = """
KEY FACTS ABOUT LINDOKUHLE:

Education:
• Currently a First Year ICT Student at Durban University of Technology
• Focus on Information Technology and Software Development

Programming Skills:
• Front-End Development: HTML and CSS
• Programming Languages: Learning C++
• Development Approach: Full-stack development aspirant
• Learning Style: Multi-language parallel learning approach

Professional Goals:
• Aspiring Full-Stack Developer
• Interested in Cybersecurity and System Protection
• Focused on understanding app development and system architecture

Unique Attributes:
• Uses innovative stress management technique: switching between programming languages
• Learning strategy: Treats web development as a continuous journey
• Problem-solving approach: When faced with challenges in one language, temporarily switches to another
• Motivation: Driven by curiosity about how applications work and systems are protected

Career Interests:
• Web Development (Both Front-end and Back-end)
• Software Development
• Cybersecurity
• System Protection and Security

Learning Philosophy:
"For me, my curiosity drives my passion and my passion determines my dream. When one language becomes challenging, I switch to another as a stress relief technique, making the learning journey more manageable."
"""


app = FastAPI(title="Ollama-FastAPI bridge")

# Allow CORS for local development. Adjust origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default Ollama endpoint. You can override with the OLLAMA_URL env var.
# Many Ollama installs expose an HTTP API at http://localhost:11434/api/generate
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")


async def call_ollama(payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    """Call the local Ollama HTTP API and return a consolidated response.

    Ollama often streams incremental JSON objects (NDJSON). This function
    collects them, extracts the `response`/`text` fields when present and
    returns a single aggregated text plus the final parsed object when available.
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(DEFAULT_OLLAMA_URL, json=payload)
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama at {DEFAULT_OLLAMA_URL}. Is Ollama running?")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Ollama request error: {str(e)}")

    if resp.status_code != 200:
        # Try to return any JSON error message from Ollama
        try:
            data = resp.json()
        except Exception:
            data = {"status_code": resp.status_code, "text": resp.text}
        raise HTTPException(status_code=502, detail={"ollama_response": data})

    text = resp.text
    # If Ollama returned NDJSON (multiple JSON objects separated by newlines), parse each
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) > 1:
        collected = []
        last_obj = None
        for line in lines:
            try:
                obj = json.loads(line)
                last_obj = obj
                # prefer common fields used by Ollama
                if isinstance(obj, dict) and 'response' in obj and obj['response']:
                    collected.append(str(obj['response']))
                elif isinstance(obj, dict) and 'text' in obj and obj['text']:
                    collected.append(str(obj['text']))
            except Exception:
                # ignore parse errors for individual lines
                continue
        joined = ''.join(collected)
        result = {"text": joined, "raw_lines": lines}
        if last_obj:
            result["last"] = last_obj
        return result

    # Single JSON or plain text
    try:
        parsed = resp.json()
        return parsed
    except Exception:
        return {"text": text}


@app.post("/api/ollama")
async def forward_to_ollama(req: OllamaRequest):
    """Accepts a prompt (or chat messages) and forwards to the local Ollama model.

    Example payloads:
    - {"prompt": "Hello"}
    - {"model": "my-model", "prompt": "Summarize..."}
    - {"messages": [{"role":"user","content":"Hi"}], "model": "chat-model"}
    """
    if not req.prompt and not req.messages:
        raise HTTPException(status_code=400, detail="Provide a 'prompt' string or 'messages' list.")

    # Prepare payload to Ollama with website context and strict instructions
    enhanced_prompt = f"""You are an AI assistant for Lindokuhle's portfolio. Your primary goal is to answer questions based on the provided context. Follow these rules precisely:

1. GREETING RULE:
   - IF the user's question is a simple greeting (e.g., "hi", "hello", "hey"), THEN respond with:
     "Hi! I'm Lindokuhle's AI assistant. I can tell you about Lindokuhle's programming skills, education at DUT, and journey in technology. What would you like to know?"
   - DO NOT use the greeting for specific questions.

2. QUESTION-ANSWERING RULE:
   - FOR ALL OTHER questions, provide a direct and specific answer based on the context below.
   - Start your response by directly addressing the user's question. For example, if asked "How does Lindokuhle handle challenges?", start with "Lindokuhle handles programming challenges by...".

3. CONTEXT-ONLY RULE:
   - You MUST base all answers on the following `KEY FACTS`:
{WEBSITE_CONTEXT}
   - If the information is not in the context, politely state that and offer to answer a question you *can* answer (e.g., "While I don't have information on that, I can tell you about Lindokuhle's programming skills.").

4. TOPIC-SPECIFIC GUIDANCE:
   - For questions about "Programming Skills," focus on HTML, CSS, C++, and the multi-language learning approach.
   - For questions about "Learning Style," explain the unique stress management technique of switching between languages.
   - For questions about "Education," mention the ICT program at Durban University of Technology.

User question: {req.prompt}

Your task is to analyze the user's question and decide whether to give the greeting or a specific answer from the context.
"""

    # Prepare payload to Ollama. Different Ollama versions/models may expect different fields.
    payload: Dict[str, Any] = {"model": req.model}
    if req.messages:
        # Some chat-style models expect 'messages'
        payload["messages"] = req.messages
    else:
        payload["prompt"] = enhanced_prompt
    if req.max_tokens:
        payload["max_tokens"] = req.max_tokens

    # Make the call with a reasonable timeout
    result = await call_ollama(payload, timeout=30)
    return {"ok": True, "ollama": result}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Mount static files AFTER registering API routes so they don't interfere
app.mount("/", StaticFiles(directory=".", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
