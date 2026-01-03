from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import ChatRequest, ChatResponse
from backend.services.vllm_client import generate_response

app = FastAPI(title="Frieren Chatbot API")

# CORS Configuration
origins = [
    "http://localhost:5173",  # Frontend typically runs on 5173
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # generate_response now returns a ChatResponse object directly using LangChain
    chat_response = await generate_response(request.message)
    return chat_response


@app.get("/")
async def root():
    return {"message": "Eternal Journey AI Backend is running"}
