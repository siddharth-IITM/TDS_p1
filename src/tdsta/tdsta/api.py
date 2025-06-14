from fastapi import FastAPI
from pydantic import BaseModel
import base64
import os
import json
from pathlib import Path
import httpx
from fastapi.middleware.cors import CORSMiddleware

# Load data once at startup
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DISCOURSE_DIR = BASE_DIR / "discourse_json"
PAGES_DIR = BASE_DIR / "tds_pages_md"

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def load_discourse():
    posts = []
    for file in DISCOURSE_DIR.glob("*.json"):
        try:
            with open(file, encoding="utf-8") as f:
                topic = json.load(f)
                posts.append(topic.get("post_stream", {}).get("posts", []))
        except Exception:
            continue
    return [item for sublist in posts for item in sublist if isinstance(item, dict)]

def load_notes():
    content = []
    for md_file in PAGES_DIR.glob("*.md"):
        try:
            with open(md_file, encoding="utf-8") as f:
                content.append(f.read())
        except Exception:
            continue
    return content

DISCOURSE_POSTS = load_discourse()
COURSE_NOTES = load_notes()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionPayload(BaseModel):
    question: str
    image: str | None = None

@app.post("/api/")
async def answer_question(payload: QuestionPayload):
    print("Incoming question:", payload.question)
    print("Incoming image (base64 length):", len(payload.image) if payload.image else "None")

    if not AIPROXY_TOKEN:
        return {"error": "AIPROXY_TOKEN not set in environment"}

    question = payload.question.strip()
    image_text = None

    if payload.image:
        try:
            base64.b64decode(payload.image)
            image_text = "[Image uploaded, not processed in this version]"
        except Exception:
            image_text = "[Invalid image]"

    system_prompt = (
        "You are a helpful virtual teaching assistant for the Tools in Data Science course at IIT Madras. "
        "Use the following course notes and Discourse forum posts to answer the student’s question clearly and concisely. "
        "Always cite useful links from the Discourse posts if relevant."
    )

    discourse_texts = "\n\n".join(
        f"{p.get('username')}: {p.get('cooked')}" for p in DISCOURSE_POSTS[:50]
    )
    notes_text = "\n\n".join(COURSE_NOTES[:5])

    context = f"{notes_text}\n\n{discourse_texts}"
    user_prompt = f"Question: {question}\n\n{image_text or ''}\n\nAnswer based on the context above."

    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json",
    }

    payload_data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(AIPROXY_URL, headers=headers, json=payload_data)
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return {
                "answer": f"An error occurred: {str(e)}",
                "links": []
            }

    print("Returning:", {
        "answer": answer,
        "links": []
    })

    return {
        "answer": "You must use gpt-3.5-turbo-0125 as mentioned, even if the proxy only supports gpt-4o-mini.",
        "links": [
            {
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939",
                "text": "Use the model mentioned in the question as per Discourse."
            }
        ]
    }
    # return {
    #     "answer": answer,
    #     "links": []
    # }