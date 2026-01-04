from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    VLLM_API_URL: str = "http://localhost:8000/v1"
    MODEL_NAME: str = "./models/llm/Qwen3-4B-Instruct-2507"

    # Database settings
    DB_PATH: str = "./data/chat_history.db"

    # Memory settings
    MAX_RECENT_MESSAGES: int = 10  # 프롬프트에 포함할 최근 메시지 개수
    SUMMARIZE_THRESHOLD: int = 20  # 요약 트리거 임계값

    class Config:
        env_file = ".env"

settings = Settings()
